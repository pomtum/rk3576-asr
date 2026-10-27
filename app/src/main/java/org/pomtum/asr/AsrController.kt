package org.pomtum.asr

import android.annotation.SuppressLint
import android.content.res.AssetManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Handler
import android.os.Looper
import android.util.Log
import java.util.concurrent.Executors

/**
 * 识别流程：加载某种语言的模型、开关麦克风、把识别结果回调到主线程。
 * 模型加载在后台线程；识别在录音线程里同步进行（一块解码约 0.1 s，录音缓冲留 2 s）。
 */
class AsrController(private val assets: AssetManager, private val listener: Listener) {

    enum class State { LOADING, READY, LISTENING, ERROR }

    interface Listener {
        fun onState(state: State, language: Language, detail: String? = null)
        fun onPartial(text: String)
        fun onSentence(text: String)
    }

    enum class Language(val assetDir: String) { ENGLISH("models/en"), CHINESE("models/zh") }

    private val main = Handler(Looper.getMainLooper())
    private val loader = Executors.newSingleThreadExecutor()
    @Volatile private var asr: NpuAsr? = null
    @Volatile private var listening = false
    private var recordThread: Thread? = null
    var language = Language.ENGLISH
        private set
    var state = State.LOADING
        private set

    fun hasModel(lang: Language): Boolean =
        runCatching { assets.list(lang.assetDir)?.contains("encoder.rknn") == true }.getOrDefault(false)

    fun load(lang: Language) {
        stop()
        language = lang
        post(State.LOADING)
        loader.execute {
            asr?.close()
            asr = null
            if (!hasModel(lang)) {
                post(State.ERROR, "missing-model")
                return@execute
            }
            try {
                val t0 = System.nanoTime()
                asr = NpuAsr(assets, lang.assetDir)
                Log.i(TAG, "loaded ${lang.name} in ${(System.nanoTime() - t0) / 1_000_000} ms")
                post(State.READY)
            } catch (t: Throwable) {
                Log.e(TAG, "load failed", t)
                post(State.ERROR, t.message)
            }
        }
    }

    @SuppressLint("MissingPermission")  // 调用方先确认已获得录音权限
    fun start(beforeRecording: () -> Unit) {
        val engine = asr ?: return
        if (listening || state != State.READY) return
        listening = true
        post(State.LISTENING)
        recordThread = Thread({ recordLoop(engine, beforeRecording) }, "asr-record").apply { start() }
    }

    fun stop() {
        if (!listening) return
        listening = false
        recordThread?.join(2000)
        recordThread = null
        if (asr != null) post(State.READY)
    }

    fun release() {
        stop()
        loader.execute { asr?.close(); asr = null }
        loader.shutdown()
    }

    @SuppressLint("MissingPermission")
    private fun recordLoop(engine: NpuAsr, beforeRecording: () -> Unit) {
        val rate = 16000
        val minBuf = AudioRecord.getMinBufferSize(rate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val rec = AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, rate,
            AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, maxOf(minBuf, rate * 4))
        if (rec.state != AudioRecord.STATE_INITIALIZED) {
            listening = false
            post(State.ERROR, "mic")
            return
        }
        beforeRecording()  // 提示音放完再开录，避免把提示音识别进去
        val buf = ShortArray(rate / 10)
        val samples = FloatArray(buf.size)
        var lastPartial = ""
        try {
            rec.startRecording()
            while (listening) {
                val n = rec.read(buf, 0, buf.size)
                if (n <= 0) continue
                for (i in 0 until n) samples[i] = buf[i] / 32768f
                engine.accept(samples, n)
                lastPartial = deliver(engine, lastPartial)
            }
            // 停止时补 0.6 s 静音，把最后半块也解出来，再把这句当作完整的一句
            engine.accept(FloatArray(rate * 6 / 10))
            val tail = engine.text.trim()
            if (tail.isNotEmpty()) main.post { listener.onSentence(tail) }
            engine.reset()
            main.post { listener.onPartial("") }
        } catch (t: Throwable) {
            Log.e(TAG, "recognition failed", t)
            listening = false
            post(State.ERROR, t.message)
        } finally {
            rec.stop()
            rec.release()
        }
    }

    private fun deliver(engine: NpuAsr, lastPartial: String): String {
        val partial = engine.text.trim()
        if (engine.isEndpoint) {
            if (partial.isNotEmpty()) main.post { listener.onSentence(partial) }
            engine.reset()
            if (lastPartial.isNotEmpty()) main.post { listener.onPartial("") }
            return ""
        }
        if (partial != lastPartial) main.post { listener.onPartial(partial) }
        return partial
    }

    private fun post(s: State, detail: String? = null) {
        state = s
        val lang = language
        main.post { listener.onState(s, lang, detail) }
    }

    companion object { private const val TAG = "PomTumASR" }
}

package org.pomtum.asr

import android.Manifest
import android.app.Activity
import android.app.AlertDialog
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import android.content.pm.PackageManager
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Bundle
import android.util.TypedValue
import android.view.HapticFeedbackConstants
import android.view.InputDevice
import android.view.KeyEvent
import android.view.View
import android.view.WindowInsets
import android.view.WindowManager
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import org.pomtum.asr.AsrController.Language
import org.pomtum.asr.AsrController.State

/**
 * 主界面：按机身左侧的 AI 键开始/停止听写（屏幕上没有开始按钮，底部只显示状态）；
 * 上方字幕区显示上一句（读屏播报）和正在说的话（不播报）；下方是全部记录，可复制、分享、清空。
 * 界面按读屏优先设计：状态和每句完整的话通过 live region 播报，开始/停止有震动和提示音。
 */
class MainActivity : Activity(), AsrController.Listener {
    private lateinit var controller: AsrController
    private lateinit var store: TranscriptStore
    private lateinit var status: TextView
    private lateinit var lastSentence: TextView
    private lateinit var partial: TextView
    private lateinit var history: LinearLayout
    private lateinit var historyScroll: ScrollView
    private lateinit var emptyHint: TextView
    private lateinit var keyBar: LinearLayout
    private lateinit var keyIcon: ImageView
    private lateinit var keyLabel: TextView
    private lateinit var language: RadioGroup
    private val tones by lazy { ToneGenerator(AudioManager.STREAM_MUSIC, 70) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val root = findViewById<View>(R.id.root)
        // 全面屏：让内容避开状态栏、导航栏和挖孔
        root.setOnApplyWindowInsetsListener { v, insets ->
            val bars = insets.getInsets(WindowInsets.Type.systemBars() or WindowInsets.Type.displayCutout())
            v.setPadding(v.paddingLeft, bars.top, v.paddingRight, bars.bottom)
            insets
        }
        status = findViewById(R.id.status)
        lastSentence = findViewById(R.id.last_sentence)
        partial = findViewById(R.id.partial)
        history = findViewById(R.id.history)
        historyScroll = findViewById(R.id.history_scroll)
        emptyHint = findViewById(R.id.empty_hint)
        keyBar = findViewById(R.id.key_bar)
        keyIcon = findViewById(R.id.key_icon)
        keyLabel = findViewById(R.id.key_label)
        language = findViewById(R.id.language)

        store = TranscriptStore(filesDir)
        store.sentences.forEach { addHistoryItem(it) }
        updateEmpty()
        historyScroll.post { historyScroll.fullScroll(View.FOCUS_DOWN) }

        controller = AsrController(assets, this)
        val saved = getPreferences(MODE_PRIVATE).getString(PREF_LANG, Language.ENGLISH.name)
        val lang = runCatching { Language.valueOf(saved!!) }.getOrDefault(Language.ENGLISH)
        language.check(if (lang == Language.CHINESE) R.id.lang_zh else R.id.lang_en)
        language.setOnCheckedChangeListener { _, id ->
            val l = if (id == R.id.lang_zh) Language.CHINESE else Language.ENGLISH
            if (l == controller.language) return@setOnCheckedChangeListener
            getPreferences(MODE_PRIVATE).edit().putString(PREF_LANG, l.name).apply()
            controller.load(l)
        }

        findViewById<Button>(R.id.copy).setOnClickListener { copyAll() }
        findViewById<Button>(R.id.share).setOnClickListener { shareAll() }
        findViewById<Button>(R.id.clear).setOnClickListener { confirmClear() }

        controller.load(lang)
    }

    override fun onStop() {
        // 离开界面就停止听写，不在后台录音
        controller.stop()
        super.onStop()
    }

    override fun onDestroy() {
        controller.release()
        tones.release()
        super.onDestroy()
    }

    // ---- AI 键：机身左侧，内核设备 adc-keys-ai 上报 KEY_A，安卓里是 KEYCODE_A ----

    // 几个 ADC 按键设备在安卓里被合成同一个设备（事件报的设备名可能是 adc-keys-vol-down），
    // 所以不按设备名判断：来自非全键盘设备的 KEYCODE_A 就是 AI 键，外接键盘（全键盘）的 A 不算
    private fun isAiKey(keyCode: Int, event: KeyEvent) =
        keyCode == KeyEvent.KEYCODE_A && event.device?.keyboardType != InputDevice.KEYBOARD_TYPE_ALPHABETIC

    override fun onKeyDown(keyCode: Int, event: KeyEvent): Boolean {
        if (!isAiKey(keyCode, event)) return super.onKeyDown(keyCode, event)
        if (event.repeatCount == 0) toggleListening()  // 长按不重复切换
        return true
    }

    override fun onKeyUp(keyCode: Int, event: KeyEvent): Boolean =
        isAiKey(keyCode, event) || super.onKeyUp(keyCode, event)

    private fun toggleListening() {
        if (controller.state == State.LISTENING) {
            keyBar.performHapticFeedback(HapticFeedbackConstants.REJECT)
            controller.stop()
            tones.startTone(ToneGenerator.TONE_PROP_ACK, 120)
            return
        }
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), REQ_MIC)
            return
        }
        startListening()
    }

    private fun startListening() {
        if (controller.state != State.READY) return
        keyBar.performHapticFeedback(HapticFeedbackConstants.CONFIRM)
        controller.start {
            tones.startTone(ToneGenerator.TONE_PROP_BEEP, 120)
            Thread.sleep(200)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        if (requestCode != REQ_MIC) return
        if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) startListening()
        else status.text = getString(R.string.status_need_permission)
    }

    // ---- AsrController.Listener ----

    override fun onState(state: State, lang: Language, detail: String?) {
        val langName = getString(if (lang == Language.CHINESE) R.string.lang_zh else R.string.lang_en)
        status.text = when (state) {
            State.LOADING -> getString(R.string.status_loading, langName)
            State.READY -> getString(R.string.status_ready, langName)
            State.LISTENING -> getString(R.string.status_listening, langName)
            State.ERROR -> when (detail) {
                "missing-model" -> getString(R.string.status_missing_model, langName)
                "mic" -> getString(R.string.status_mic_error)
                else -> getString(R.string.status_error, detail ?: "")
            }
        }
        val live = state == State.LISTENING
        emptyHint.setText(if (live) R.string.hint_listening else R.string.empty_hint)
        keyLabel.setText(if (live) R.string.key_stop else R.string.key_start)
        keyBar.setBackgroundResource(
            when {
                live -> R.drawable.mic_live_bg
                state == State.READY -> R.drawable.mic_idle_bg
                else -> R.drawable.mic_disabled_bg
            })
        keyLabel.setTextColor(getColor(if (live) R.color.on_mic_live else R.color.on_mic_idle))
        keyIcon.setImageResource(if (live) R.drawable.ic_stop else R.drawable.ic_mic)
        keyBar.alpha = if (state == State.READY || live) 1f else 0.6f
        language.isEnabled = state != State.LOADING
        for (i in 0 until language.childCount) language.getChildAt(i).isEnabled = state != State.LOADING
        // 听写时保持屏幕常亮
        if (live) window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        else window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

    override fun onPartial(text: String) {
        partial.text = text
        updateEmpty()
    }

    override fun onSentence(text: String) {
        lastSentence.text = text   // live region：读屏会念出这一句
        partial.text = ""
        store.add(text)
        addHistoryItem(text)
        updateEmpty()
        historyScroll.post { historyScroll.fullScroll(View.FOCUS_DOWN) }
    }

    // ---- 记录 ----

    private fun addHistoryItem(text: String) {
        val tv = TextView(this).apply {
            this.text = text
            setTextColor(getColor(R.color.text))
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 20f)
            setPadding(0, dp(8), 0, dp(8))
            setTextIsSelectable(false)
        }
        history.addView(tv)
    }

    private fun updateEmpty() {
        // 字幕区为空时显示提示语；没有记录时隐藏“记录”标题
        val captionEmpty = lastSentence.text.isEmpty() && partial.text.isEmpty()
        emptyHint.visibility = if (captionEmpty) View.VISIBLE else View.GONE
        lastSentence.visibility = if (lastSentence.text.isEmpty()) View.GONE else View.VISIBLE
        partial.visibility = if (partial.text.isEmpty()) View.GONE else View.VISIBLE
        val title = findViewById<TextView>(R.id.history_title)
        title.visibility = if (store.sentences.isEmpty()) View.INVISIBLE else View.VISIBLE
        title.text = "${getString(R.string.history)} · ${getString(R.string.sentence_count, store.sentences.size)}"
    }

    private fun copyAll() {
        if (store.sentences.isEmpty()) return toast(R.string.nothing_to_copy)
        getSystemService(ClipboardManager::class.java)
            .setPrimaryClip(ClipData.newPlainText(getString(R.string.app_name), store.allText()))
        toast(R.string.copied)
    }

    private fun shareAll() {
        if (store.sentences.isEmpty()) return toast(R.string.nothing_to_copy)
        val send = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_TEXT, store.allText())
        }
        startActivity(Intent.createChooser(send, getString(R.string.share_title)))
    }

    private fun confirmClear() {
        if (store.sentences.isEmpty()) return toast(R.string.nothing_to_copy)
        AlertDialog.Builder(this)
            .setTitle(R.string.clear_confirm_title)
            .setMessage(getString(R.string.clear_confirm_message, store.sentences.size))
            .setPositiveButton(R.string.clear_confirm_ok) { _, _ ->
                store.clear()
                history.removeAllViews()
                lastSentence.text = ""
                updateEmpty()
                toast(R.string.cleared)
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    private fun toast(res: Int) = Toast.makeText(this, res, Toast.LENGTH_SHORT).show()
    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    companion object {
        private const val REQ_MIC = 1
        private const val PREF_LANG = "language"
    }
}

package org.pomtum.asr

import android.content.res.AssetManager
import java.io.Closeable

/**
 * NPU 流式识别（自有 C 实现：零拷贝 RKNN 编码器 + 贪心搜索）。
 * 模型在 assets 的 [assetDir] 下：encoder/decoder/joiner.rknn、tokens.txt、encoder_layout.txt。
 * 非线程安全：同一时刻只能有一个线程调用。
 */
class NpuAsr(assets: AssetManager, assetDir: String) : Closeable {
    // 持有 AssetManager，保证内存映射的模型在识别器存活期间有效
    private val assetManager = assets
    private var handle: Long = nativeCreate(assetManager, assetDir)

    fun accept(samples: FloatArray, n: Int = samples.size) {
        check(nativeAccept(handle, samples, n)) { "NPU decode failed" }
    }

    val text: String get() = nativeText(handle)
    val isEndpoint: Boolean get() = nativeIsEndpoint(handle)
    fun reset() = nativeReset(handle)

    override fun close() {
        if (handle != 0L) nativeDestroy(handle)
        handle = 0L
    }

    companion object {
        init {
            System.loadLibrary("npuasr")
        }

        @JvmStatic private external fun nativeCreate(assets: AssetManager, dir: String): Long
        @JvmStatic private external fun nativeDestroy(h: Long)
        @JvmStatic private external fun nativeAccept(h: Long, samples: FloatArray, n: Int): Boolean
        @JvmStatic private external fun nativeText(h: Long): String
        @JvmStatic private external fun nativeIsEndpoint(h: Long): Boolean
        @JvmStatic private external fun nativeReset(h: Long)
    }
}

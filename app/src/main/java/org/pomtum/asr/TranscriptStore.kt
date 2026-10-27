package org.pomtum.asr

import java.io.File

/** 识别记录：一行一句，保存在 app 私有目录，重启后还在 */
class TranscriptStore(dir: File) {
    private val file = File(dir, "transcript.txt")
    val sentences: MutableList<String> = mutableListOf()

    init {
        if (file.exists()) sentences += file.readLines().filter { it.isNotBlank() }
    }

    fun add(text: String) {
        val line = text.replace('\n', ' ').trim()
        if (line.isEmpty()) return
        sentences += line
        file.appendText(line + "\n")
    }

    fun clear() {
        sentences.clear()
        file.delete()
    }

    fun allText(): String = sentences.joinToString("\n")
}

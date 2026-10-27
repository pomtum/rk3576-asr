// 流式识别：fbank -> 零拷贝 RKNN 编码器 -> RKNN 解码器/joiner 贪心搜索，带断句。
#pragma once

#include <stddef.h>

// 一个语言模型的全部数据；各缓冲区须在识别器销毁前一直有效（可直接指向 APK assets 的内存映射）
typedef struct {
    const void *encoder, *decoder, *joiner;
    size_t encoder_len, decoder_len, joiner_len;
    const char *tokens;   // tokens.txt 内容，'\0' 结尾
    const char *layout;   // encoder_layout.txt 内容，'\0' 结尾
} ModelData;

typedef struct Recognizer Recognizer;

Recognizer *recognizer_create(const ModelData *m, char *err, size_t err_len);
void recognizer_destroy(Recognizer *r);
// 送入 16 kHz [-1,1] 样本，凑够一块就解码
int recognizer_accept(Recognizer *r, const float *samples, size_t n);
// 输入结束：补静音并解完剩余部分
int recognizer_finish(Recognizer *r);
// 当前这句话的文本（UTF-8，调用方不释放；下次调用前有效）
const char *recognizer_text(Recognizer *r);
// 是否到了句尾（按尾部静音时长判断）
int recognizer_is_endpoint(const Recognizer *r);
// 开始新的一句：清空文本，保留编码器状态
void recognizer_reset_utterance(Recognizer *r);

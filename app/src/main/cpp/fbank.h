// 流式 80 维 log-mel fbank，参数与 kaldi-native-fbank / sherpa-onnx 默认一致：
// 16 kHz，帧长 25 ms，帧移 10 ms，povey 窗，预加重 0.97，去直流，dither 0，snip_edges=false，
// mel 20 Hz ~ 7600 Hz，样本按 [-1, 1] 浮点输入。
#pragma once

#include <stddef.h>

#define FBANK_DIM 80

typedef struct Fbank Fbank;

Fbank *fbank_create(void);
void fbank_destroy(Fbank *fb);
void fbank_reset(Fbank *fb);
// 送入样本；新算出的帧追加到内部帧缓冲
void fbank_accept(Fbank *fb, const float *samples, size_t n);
// 输入结束：补齐最后的帧（尾部反射）
void fbank_finish(Fbank *fb);
// 已算出的总帧数（从流开始计）
size_t fbank_num_frames(const Fbank *fb);
// 第 index 帧（index 必须仍在缓冲内）
const float *fbank_frame(const Fbank *fb, size_t index);
// 丢弃 index 之前的帧，释放缓冲
void fbank_discard_before(Fbank *fb, size_t index);

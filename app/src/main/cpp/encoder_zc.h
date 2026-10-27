// 流式 zipformer 编码器（RKNN），状态在 NPU 内存中零拷贝衔接。
// 状态的原生输出布局由随模型分发的 encoder_layout.txt 给出（设备上用 zcbench 校准生成）。
#pragma once

#include <stddef.h>

typedef struct EncoderZc EncoderZc;

// model：模型数据，调用方须保证在编码器销毁前一直有效；layout_text：encoder_layout.txt 的内容
EncoderZc *encoder_zc_create(const void *model, size_t model_len, const char *layout_text, char *err, size_t err_len);
void encoder_zc_destroy(EncoderZc *e);
// 状态清零（新的一段流）
void encoder_zc_reset(EncoderZc *e);
int encoder_zc_chunk_frames(const EncoderZc *e);  // 每块输入帧数 T
int encoder_zc_chunk_shift(const EncoderZc *e);   // 每块前进的帧数（模型元数据 decode_chunk_len）
int encoder_zc_out_frames(const EncoderZc *e);    // 每块输出帧数
int encoder_zc_out_dim(const EncoderZc *e);
// x: T*80 float32；out: out_frames*out_dim float32。成功返回 0
int encoder_zc_run(EncoderZc *e, const float *x, float *out);

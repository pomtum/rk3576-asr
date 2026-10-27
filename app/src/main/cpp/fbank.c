#include "fbank.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define SR 16000
#define FRAME_LEN 400
#define FRAME_SHIFT 160
#define NFFT 512
#define NBIN (NFFT / 2 + 1)

struct Fbank {
    // 样本缓冲：samples[0] 对应流中的第 sample_base 个样本
    float *samples;
    size_t n_samples, cap_samples, sample_base;
    int finished;
    // 帧缓冲：frames[0] 对应第 frame_base 帧
    float *frames;
    size_t n_frames, cap_frames, frame_base;
    // 预计算
    float window[FRAME_LEN];
    float mel_lo[FBANK_DIM], mel_center[FBANK_DIM], mel_hi[FBANK_DIM];
    int bin_lo[FBANK_DIM], bin_hi[FBANK_DIM];
    float *mel_w[FBANK_DIM];
    float cos_t[NFFT / 2], sin_t[NFFT / 2];
    int bitrev[NFFT];
};

static float mel_scale(float f) { return 1127.0f * logf(1.0f + f / 700.0f); }

static void fft(const Fbank *fb, float *re, float *im) {
    for (int i = 0; i < NFFT; i++) {
        int j = fb->bitrev[i];
        if (j > i) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= NFFT; len <<= 1) {
        int step = NFFT / len;
        for (int i = 0; i < NFFT; i += len) {
            for (int k = 0; k < len / 2; k++) {
                float wr = fb->cos_t[k * step], wi = -fb->sin_t[k * step];
                int a = i + k, b = i + k + len / 2;
                float xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
                re[b] = re[a] - xr; im[b] = im[a] - xi;
                re[a] += xr; im[a] += xi;
            }
        }
    }
}

Fbank *fbank_create(void) {
    Fbank *fb = calloc(1, sizeof(Fbank));
    for (int i = 0; i < FRAME_LEN; i++)
        fb->window[i] = powf(0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / (FRAME_LEN - 1)), 0.85f);
    for (int i = 0; i < NFFT / 2; i++) {
        fb->cos_t[i] = cosf(2.0f * (float)M_PI * i / NFFT);
        fb->sin_t[i] = sinf(2.0f * (float)M_PI * i / NFFT);
    }
    for (int i = 0, bits = 9; i < NFFT; i++) {
        int r = 0;
        for (int b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
        fb->bitrev[i] = r;
    }
    // 三角 mel 滤波器（Kaldi 方式：在 mel 域线性插值）
    float lo = mel_scale(20.0f), hi = mel_scale(SR / 2.0f - 400.0f);
    float delta = (hi - lo) / (FBANK_DIM + 1);
    for (int m = 0; m < FBANK_DIM; m++) {
        float l = lo + m * delta, c = l + delta, r = c + delta;
        fb->bin_lo[m] = -1;
        fb->mel_w[m] = calloc(NBIN, sizeof(float));
        for (int k = 0; k < NBIN; k++) {
            float mk = mel_scale((float)k * SR / NFFT);
            float w = 0.0f;
            if (mk > l && mk < r) w = mk <= c ? (mk - l) / (c - l) : (r - mk) / (r - c);
            if (w > 0.0f) {
                if (fb->bin_lo[m] < 0) fb->bin_lo[m] = k;
                fb->bin_hi[m] = k;
            }
            fb->mel_w[m][k] = w;
        }
    }
    return fb;
}

void fbank_destroy(Fbank *fb) {
    if (!fb) return;
    for (int m = 0; m < FBANK_DIM; m++) free(fb->mel_w[m]);
    free(fb->samples);
    free(fb->frames);
    free(fb);
}

void fbank_reset(Fbank *fb) {
    fb->n_samples = fb->sample_base = 0;
    fb->n_frames = fb->frame_base = 0;
    fb->finished = 0;
}

// 取流中第 idx 个样本（snip_edges=false：两端反射）
static float sample_at(const Fbank *fb, long idx, long total) {
    if (idx < 0) idx = -idx - 1;
    if (total > 0 && idx >= total) idx = 2 * total - idx - 1;
    long rel = idx - (long)fb->sample_base;
    if (rel < 0 || rel >= (long)fb->n_samples) return 0.0f;
    return fb->samples[rel];
}

static void push_frame(Fbank *fb, const float *v) {
    if (fb->n_frames == fb->cap_frames) {
        fb->cap_frames = fb->cap_frames ? fb->cap_frames * 2 : 256;
        fb->frames = realloc(fb->frames, fb->cap_frames * FBANK_DIM * sizeof(float));
    }
    memcpy(fb->frames + fb->n_frames * FBANK_DIM, v, FBANK_DIM * sizeof(float));
    fb->n_frames++;
}

static void compute_frame(Fbank *fb, size_t k, long total) {
    float re[NFFT], im[NFFT], out[FBANK_DIM];
    long start = (long)k * FRAME_SHIFT + FRAME_SHIFT / 2 - FRAME_LEN / 2;
    float mean = 0.0f;
    for (int i = 0; i < FRAME_LEN; i++) { re[i] = sample_at(fb, start + i, total); mean += re[i]; }
    mean /= FRAME_LEN;
    for (int i = 0; i < FRAME_LEN; i++) re[i] -= mean;
    for (int i = FRAME_LEN - 1; i > 0; i--) re[i] -= 0.97f * re[i - 1];
    re[0] -= 0.97f * re[0];
    for (int i = 0; i < FRAME_LEN; i++) re[i] *= fb->window[i];
    for (int i = FRAME_LEN; i < NFFT; i++) re[i] = 0.0f;
    memset(im, 0, sizeof(im));
    fft(fb, re, im);
    float power[NBIN];
    for (int k2 = 0; k2 < NBIN; k2++) power[k2] = re[k2] * re[k2] + im[k2] * im[k2];
    for (int m = 0; m < FBANK_DIM; m++) {
        float e = 0.0f;
        for (int k2 = fb->bin_lo[m]; k2 <= fb->bin_hi[m]; k2++) e += power[k2] * fb->mel_w[m][k2];
        out[m] = logf(e > 1.1920929e-07f ? e : 1.1920929e-07f);
    }
    push_frame(fb, out);
}

static void compute_ready(Fbank *fb) {
    size_t total_in = fb->sample_base + fb->n_samples;
    // 输入结束时按 Kaldi snip_edges=false 的帧数：(样本数 + 帧移/2) / 帧移
    size_t final_frames = (total_in + FRAME_SHIFT / 2) / FRAME_SHIFT;
    for (;;) {
        size_t k = fb->frame_base + fb->n_frames;
        long end = (long)k * FRAME_SHIFT + FRAME_SHIFT / 2 + FRAME_LEN / 2;
        if (!fb->finished && end > (long)total_in) break;
        if (fb->finished && k >= final_frames) break;
        compute_frame(fb, k, fb->finished ? (long)total_in : 0);
    }
    // 丢掉已不再需要的样本（保留最后一帧窗口之前的一段）
    size_t next = fb->frame_base + fb->n_frames;
    long keep_from = (long)next * FRAME_SHIFT + FRAME_SHIFT / 2 - FRAME_LEN / 2 - FRAME_LEN;
    if (keep_from > (long)fb->sample_base) {
        size_t drop = (size_t)keep_from - fb->sample_base;
        if (drop > fb->n_samples) drop = fb->n_samples;
        memmove(fb->samples, fb->samples + drop, (fb->n_samples - drop) * sizeof(float));
        fb->n_samples -= drop;
        fb->sample_base += drop;
    }
}

void fbank_accept(Fbank *fb, const float *samples, size_t n) {
    if (fb->n_samples + n > fb->cap_samples) {
        fb->cap_samples = (fb->n_samples + n) * 2;
        fb->samples = realloc(fb->samples, fb->cap_samples * sizeof(float));
    }
    memcpy(fb->samples + fb->n_samples, samples, n * sizeof(float));
    fb->n_samples += n;
    compute_ready(fb);
}

void fbank_finish(Fbank *fb) {
    fb->finished = 1;
    compute_ready(fb);
}

size_t fbank_num_frames(const Fbank *fb) { return fb->frame_base + fb->n_frames; }

const float *fbank_frame(const Fbank *fb, size_t index) { return fb->frames + (index - fb->frame_base) * FBANK_DIM; }

void fbank_discard_before(Fbank *fb, size_t index) {
    if (index <= fb->frame_base) return;
    size_t drop = index - fb->frame_base;
    if (drop > fb->n_frames) drop = fb->n_frames;
    memmove(fb->frames, fb->frames + drop * FBANK_DIM, (fb->n_frames - drop) * FBANK_DIM * sizeof(float));
    fb->n_frames -= drop;
    fb->frame_base += drop;
}

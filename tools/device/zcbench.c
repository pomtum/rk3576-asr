// Zero-copy state passing for the streaming encoder, and layout calibration.
// All inputs/outputs are bound to NPU memory with rknn_set_io_mem (native layout, fp16). After each chunk the
// states are copied from the native output layout to the native input layout with tight loops.
// The regular API (fp16 in/out) is used as the reference: for every state the program picks the layout formula
// that matches it element by element, writes encoder_layout.txt (shipped with the model), then compares the
// outputs and CPU time of both paths.
// Usage: zcbench <encoder.rknn> <features.f32> [chunks]
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "rknn_api.h"

#define MAXIO 128

static void *read_all(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    *len = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    void *buf = malloc(*len);
    if (fread(buf, 1, *len, f) != *len) { perror("fread"); exit(1); }
    fclose(f);
    return buf;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static double cpu_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static inline uint16_t f32_to_f16(float f) { _Float16 h = (_Float16)f; uint16_t u; memcpy(&u, &h, 2); return u; }
static inline float f16_to_f32(uint16_t u) { _Float16 h; memcpy(&h, &u, 2); return (float)h; }

typedef struct {
    rknn_context ctx;
    rknn_input_output_num io;
    rknn_tensor_attr in[MAXIO], out[MAXIO], nin[MAXIO], nout[MAXIO];
} Enc;

static void enc_init(Enc *e, void *model, size_t len) {
    if (rknn_init(&e->ctx, model, (uint32_t)len, 0, NULL) != RKNN_SUCC) { printf("init fail\n"); exit(1); }
    rknn_query(e->ctx, RKNN_QUERY_IN_OUT_NUM, &e->io, sizeof(e->io));
    for (uint32_t i = 0; i < e->io.n_input; i++) {
        e->in[i].index = e->nin[i].index = i;
        rknn_query(e->ctx, RKNN_QUERY_INPUT_ATTR, &e->in[i], sizeof(rknn_tensor_attr));
        rknn_query(e->ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &e->nin[i], sizeof(rknn_tensor_attr));
    }
    for (uint32_t i = 0; i < e->io.n_output; i++) {
        e->out[i].index = e->nout[i].index = i;
        rknn_query(e->ctx, RKNN_QUERY_OUTPUT_ATTR, &e->out[i], sizeof(rknn_tensor_attr));
        rknn_query(e->ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &e->nout[i], sizeof(rknn_tensor_attr));
    }
}

// ---- 布局映射：输出逻辑形状按 (N,C,H,W) 理解，候选原生布局公式 ----
typedef struct { int N, C, H, W; } Shape4;

static Shape4 out_shape(const rknn_tensor_attr *a, int variant) {
    Shape4 s = {1, 1, 1, 1};
    if (a->n_dims == 4) { s.N = a->dims[0]; s.C = a->dims[1]; s.H = a->dims[2]; s.W = a->dims[3]; }
    else if (a->n_dims == 3) {
        if (variant == 0) { s.N = a->dims[0]; s.C = a->dims[1]; s.W = a->dims[2]; }      // (N,C,W)
        else { s.C = a->dims[0]; s.H = a->dims[1]; s.W = a->dims[2]; }                     // (C,H,W)
    } else if (a->n_dims == 2) { s.N = a->dims[0]; s.W = a->dims[1]; }
    else { s.W = a->n_elems; }
    return s;
}

// 原生输出中逻辑元素 (n,c,h,w) 的位置；kind 0 = 普通行主序，1 = NC1HWC2（C2=8，ws = 每个通道组的元素数）
static inline size_t native_idx(int kind, Shape4 s, int ws, int n, int c, int h, int w) {
    if (kind == 0) return (((size_t)n * s.C + c) * s.H + h) * s.W + w;
    // NC1HWC2：每个通道组 (n, c1) 占 plane 个元素（可能带尾部对齐），组内按 (h, w, c2) 排列
    int c1n = (s.C + 7) / 8;
    return ((size_t)n * c1n + c / 8) * (size_t)ws + ((size_t)h * s.W + w) * 8 + c % 8;
}

typedef struct { int kind, variant, ws; Shape4 s; } Map;

// 原生输出 -> 普通 NCHW（输出的逻辑形状）
// kind 2：3 维 (d0,d1,d2) 以最后一维为通道的 NC1HWC2，s = {N=d0, C=d2, H=d1, W=1}，产出 (d0,d1,d2) 行主序
static void apply_map(const Map *m, const uint16_t *src, uint16_t *dst) {
    Shape4 s = m->s;
    if (m->kind == 0) { memcpy(dst, src, (size_t)s.N * s.C * s.H * s.W * 2); return; }
    if (m->kind == 2) {
        int c1n = (s.C + 7) / 8;
        for (int n = 0; n < s.N; n++)
            for (int h = 0; h < s.H; h++) {
                uint16_t *d = dst + ((size_t)n * s.H + h) * s.C;
                for (int c1 = 0; c1 < c1n; c1++) {
                    const uint16_t *p = src + (((size_t)n * c1n + c1) * s.H + h) * 8;
                    int cn = s.C - c1 * 8 < 8 ? s.C - c1 * 8 : 8;
                    memcpy(d + c1 * 8, p, (size_t)cn * 2);
                }
            }
        return;
    }
    for (int n = 0; n < s.N; n++)
        for (int c = 0; c < s.C; c++)
            for (int h = 0; h < s.H; h++) {
                uint16_t *d = dst + (((size_t)n * s.C + c) * s.H + h) * s.W;
                const uint16_t *p = src + native_idx(1, s, m->ws, n, c, h, 0);
                for (int w = 0; w < s.W; w++) d[w] = p[(size_t)w * 8];
            }
}

// 普通 NCHW -> 原生输入布局。nin 为原生输入属性；4 维输入的逻辑形状是 NHWC（in 属性 dims = N,H,W,C）
static void write_input(const rknn_tensor_attr *in, const rknn_tensor_attr *nin, Shape4 s, const uint16_t *src, uint16_t *dst) {
    if (nin->fmt == RKNN_TENSOR_NC1HWC2) {
        int ws = (int)(nin->size_with_stride / 2 / ((size_t)s.N * ((s.C + 7) / 8)));
        for (int n = 0; n < s.N; n++)
            for (int c = 0; c < s.C; c++)
                for (int h = 0; h < s.H; h++)
                    for (int w = 0; w < s.W; w++)
                        dst[native_idx(1, s, ws, n, c, h, w)] = src[(((size_t)n * s.C + c) * s.H + h) * s.W + w];
    } else if (in->n_dims == 4 && in->fmt == RKNN_TENSOR_NHWC && s.C > 1) {
        for (int n = 0; n < s.N; n++)
            for (int c = 0; c < s.C; c++)
                for (int h = 0; h < s.H; h++)
                    for (int w = 0; w < s.W; w++)
                        dst[(((size_t)n * s.H + h) * s.W + w) * s.C + c] = src[(((size_t)n * s.C + c) * s.H + h) * s.W + w];
    } else {
        memcpy(dst, src, (size_t)s.N * s.C * s.H * s.W * 2);
    }
}

// 普通 NCHW -> 参照路径的输入布局（4 维 NHWC），用于逐元素核对
static void to_ref_layout(const rknn_tensor_attr *in, Shape4 s, const uint16_t *src, uint16_t *dst) {
    rknn_tensor_attr plain = *in;
    plain.fmt = RKNN_TENSOR_UNDEFINED;
    write_input(in, &plain, s, src, dst);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    size_t mlen, flen;
    void *model = read_all(argv[1], &mlen);
    float *feats = read_all(argv[2], &flen);
    int max_chunks = argc > 3 ? atoi(argv[3]) : 25;

    static Enc ref, zc;
    enc_init(&ref, model, mlen);
    enc_init(&zc, model, mlen);
    uint32_t nin_n = ref.io.n_input, nout_n = ref.io.n_output;
    uint32_t T = ref.in[0].dims[1], F = ref.in[0].dims[2];
    int nchunk = (int)(flen / 4 / (T * F));
    if (nchunk > max_chunks) nchunk = max_chunks;
    uint32_t eo_n = ref.out[0].n_elems;

    // ---- 参照路径：常规接口，fp16 状态 ----
    void *rst[MAXIO], *rnx[MAXIO];
    for (uint32_t i = 1; i < nin_n; i++) { rst[i] = calloc(ref.in[i].n_elems, 8); rnx[i] = calloc(ref.in[i].n_elems, 8); }
    float *ref_eo = malloc((size_t)nchunk * eo_n * 4);

    // ---- 零拷贝路径：绑定 NPU 内存 ----
    rknn_tensor_mem *min[MAXIO], *mout[MAXIO];
    for (uint32_t i = 0; i < nin_n; i++) {
        min[i] = rknn_create_mem(zc.ctx, zc.nin[i].size_with_stride);
        memset(min[i]->virt_addr, 0, zc.nin[i].size_with_stride);
        rknn_tensor_attr a = zc.nin[i];
        if (a.fmt == RKNN_TENSOR_NC1HWC2) a.pass_through = 1;
        if (rknn_set_io_mem(zc.ctx, min[i], &a) != RKNN_SUCC) {
            printf("set_io_mem in %u (%s) fail: native fmt=%d type=%d n_dims=%u dims=(%u,%u,%u,%u,%u) size=%u w_stride=%u | plain fmt=%d dims=(%u,%u,%u,%u)\n",
                   i, a.name, a.fmt, a.type, a.n_dims, a.dims[0], a.dims[1], a.dims[2], a.dims[3], a.dims[4], a.size_with_stride,
                   a.w_stride, zc.in[i].fmt, zc.in[i].dims[0], zc.in[i].dims[1], zc.in[i].dims[2], zc.in[i].dims[3]);
            return 1;
        }
    }
    for (uint32_t i = 0; i < nout_n; i++) {
        mout[i] = rknn_create_mem(zc.ctx, zc.nout[i].size_with_stride);
        rknn_tensor_attr a = zc.nout[i];
        if (rknn_set_io_mem(zc.ctx, mout[i], &a) != RKNN_SUCC) { printf("set_io_mem out %u fail\n", i); return 1; }
    }
    printf("x native type=%d fmt=%d size=%u; encoder_out native type=%d size=%u; processed_lens in type=%d\n",
           zc.nin[0].type, zc.nin[0].fmt, zc.nin[0].size_with_stride, zc.nout[0].type, zc.nout[0].size_with_stride,
           zc.nin[nin_n - 1].type);

    static uint16_t scratch[1 << 20];
    Map map[MAXIO];
    memset(map, 0, sizeof(map));
    float *zc_eo = malloc((size_t)nchunk * eo_n * 4);
    double w_ref = 0, c_ref = 0, w_zc = 0, c_zc = 0, w_cp = 0, c_cp = 0, w_run = 0;

    for (int c = 0; c < nchunk; c++) {
        const float *x = feats + (size_t)c * T * F;
        // 参照
        rknn_input in[MAXIO];
        rknn_output out[MAXIO];
        memset(in, 0, sizeof(in));
        memset(out, 0, sizeof(out));
        for (uint32_t i = 0; i < nin_n; i++) {
            int is64 = ref.in[i].type == RKNN_TENSOR_INT64;
            in[i].index = i;
            in[i].type = i == 0 ? RKNN_TENSOR_FLOAT32 : is64 ? RKNN_TENSOR_INT64 : RKNN_TENSOR_FLOAT16;
            in[i].fmt = ref.in[i].fmt;
            in[i].buf = i == 0 ? (void *)x : rst[i];
            in[i].size = ref.in[i].n_elems * (i == 0 ? 4 : is64 ? 8 : 2);
        }
        for (uint32_t i = 0; i < nout_n; i++) {
            int is64 = ref.out[i].type == RKNN_TENSOR_INT64;
            out[i].index = i;
            out[i].want_float = i == 0;
            out[i].is_prealloc = 1;
            out[i].buf = i == 0 ? (void *)(ref_eo + (size_t)c * eo_n) : rnx[i];
            out[i].size = ref.out[i].n_elems * (i == 0 ? 4 : is64 ? 8 : 2);
        }
        double w0 = now_ms(), c0 = cpu_ms();
        rknn_inputs_set(ref.ctx, nin_n, in);
        rknn_run(ref.ctx, NULL);
        rknn_outputs_get(ref.ctx, nout_n, out, NULL);
        for (uint32_t i = 1; i < nin_n; i++) {
            rknn_tensor_attr *a = &ref.in[i];
            int is64 = a->type == RKNN_TENSOR_INT64;
            if (!is64 && a->n_dims == 4) {
                int n = a->dims[0], h = a->dims[1], w = a->dims[2], ch = a->dims[3];
                for (int b = 0; b < n; b++)
                    for (int y = 0; y < h; y++)
                        for (int xx = 0; xx < w; xx++)
                            for (int k = 0; k < ch; k++)
                                ((uint16_t *)rst[i])[b * h * w * ch + y * w * ch + xx * ch + k] =
                                    ((uint16_t *)rnx[i])[b * h * w * ch + k * h * w + y * w + xx];
            } else {
                memcpy(rst[i], rnx[i], a->n_elems * (is64 ? 8 : 2));
            }
        }
        double w1 = now_ms(), c1 = cpu_ms();

        // 零拷贝：特征 float32 -> fp16 写入原生输入内存
        double w2 = now_ms(), c2 = cpu_ms();
        uint16_t *xin = min[0]->virt_addr;
        for (uint32_t k = 0; k < T * F; k++) xin[k] = f32_to_f16(x[k]);
        rknn_mem_sync(zc.ctx, min[0], RKNN_MEMORY_SYNC_TO_DEVICE);
        double wr0 = now_ms();
        if (rknn_run(zc.ctx, NULL) != RKNN_SUCC) { printf("zc run fail\n"); return 1; }
        double wr1 = now_ms();
        for (uint32_t i = 0; i < nout_n; i++) rknn_mem_sync(zc.ctx, mout[i], RKNN_MEMORY_SYNC_FROM_DEVICE);
        double w3 = now_ms(), c3 = cpu_ms();
        // encoder_out fp16 -> float32
        uint16_t *eo16 = mout[0]->virt_addr;
        for (uint32_t k = 0; k < eo_n; k++) zc_eo[(size_t)c * eo_n + k] = f16_to_f32(eo16[k]);

        if (c == 0) {
            // 第一块：用参照结果为每个状态选布局公式
            int bad = 0;
            for (uint32_t i = 1; i < nin_n; i++) {
                rknn_tensor_attr *ia = &zc.in[i], *oa = &zc.out[i];
                if (ia->type == RKNN_TENSOR_INT64) { map[i].kind = 0; map[i].s = (Shape4){1, 1, 1, (int)ia->n_elems}; continue; }
                int found = 0;
                for (int kind = 0; kind < 3 && !found; kind++)
                    for (int variant = 0; variant < 2 && !found; variant++) {
                        if (kind == 2 && (oa->n_dims != 3 || variant)) continue;
                        Map m = {kind, variant, 0, out_shape(oa, variant)};
                        if (kind == 2) m.s = (Shape4){(int)oa->dims[0], (int)oa->dims[2], (int)oa->dims[1], 1};
                        size_t groups = (size_t)m.s.N * ((m.s.C + 7) / 8);
                        m.ws = kind == 1 ? (int)(zc.nout[i].size_with_stride / 2 / groups) : m.s.W;
                        if (kind == 1 && m.ws < m.s.H * m.s.W * 8) continue;
                        if (kind == 2 && zc.nout[i].size_with_stride < (uint32_t)(m.s.N * ((m.s.C + 7) / 8) * m.s.H * 16)) continue;
                        if (kind == 0 && zc.nout[i].size_with_stride < ia->n_elems * 2) continue;
                        uint16_t *tmp = malloc(ia->n_elems * 2), *tmp2 = malloc(ia->n_elems * 2);
                        apply_map(&m, mout[i]->virt_addr, tmp);
                        if (kind == 2) memcpy(tmp2, tmp, ia->n_elems * 2);
                        else to_ref_layout(ia, m.s, tmp, tmp2);
                        if (!memcmp(tmp2, rst[i], ia->n_elems * 2)) { map[i] = m; found = 1; }
                        free(tmp);
                        free(tmp2);
                    }
                if (!found) {
                    bad++;
                    if (bad < 6) printf("  no layout match for %s\n", ia->name);
                    if (bad <= 3) {  // 导出原生字节与参照值，供 PC 端反推布局
                        char p[256];
                        snprintf(p, sizeof(p), "layout_%u.native", i);
                        FILE *f = fopen(p, "wb"); fwrite(mout[i]->virt_addr, 1, zc.nout[i].size_with_stride, f); fclose(f);
                        snprintf(p, sizeof(p), "layout_%u.ref", i);
                        f = fopen(p, "wb"); fwrite(rst[i], 2, ia->n_elems, f); fclose(f);
                        printf("    dumped %s dims=(%u,%u,%u) native size=%u\n", p, oa->dims[0], oa->dims[1], oa->dims[2],
                               zc.nout[i].size_with_stride);
                    }
                }
            }
            int k0 = 0, k1 = 0;
            for (uint32_t i = 1; i < nin_n; i++) { if (map[i].kind == 0) k0++; else k1++; }
            printf("layout maps: plain=%d NC1HWC2=%d unmatched=%d\n", k0, k1, bad);
            if (bad) return 1;
            // 写出布局映射，随模型分发：每行 "输入序号 kind N C H W ws"
            FILE *lf = fopen("encoder_layout.txt", "w");
            for (uint32_t i = 1; i < nin_n; i++)
                fprintf(lf, "%u %d %d %d %d %d %d\n", i, map[i].kind, map[i].s.N, map[i].s.C, map[i].s.H, map[i].s.W, map[i].ws);
            fclose(lf);
            printf("wrote encoder_layout.txt\n");
        }
        // 状态：原生输出 -> 原生输入
        double w4 = now_ms(), c4 = cpu_ms();
        for (uint32_t i = 1; i < nin_n; i++) {
            if (zc.in[i].type == RKNN_TENSOR_INT64) memcpy(min[i]->virt_addr, mout[i]->virt_addr, zc.in[i].n_elems * 8);
            else if (map[i].kind == 0 && zc.nin[i].fmt != RKNN_TENSOR_NC1HWC2 && !(zc.in[i].n_dims == 4 && map[i].s.C > 1))
                memcpy(min[i]->virt_addr, mout[i]->virt_addr, zc.in[i].n_elems * 2);
            else if (map[i].kind == 2) apply_map(&map[i], mout[i]->virt_addr, min[i]->virt_addr);
            else {
                apply_map(&map[i], mout[i]->virt_addr, scratch);
                write_input(&zc.in[i], &zc.nin[i], map[i].s, scratch, min[i]->virt_addr);
            }
            rknn_mem_sync(zc.ctx, min[i], RKNN_MEMORY_SYNC_TO_DEVICE);
        }
        double w5 = now_ms(), c5 = cpu_ms();
        if (c > 0) {
            w_ref += w1 - w0; c_ref += c1 - c0;
            w_zc += (w3 - w2) + (w5 - w4); c_zc += (c3 - c2) + (c5 - c4);
            w_cp += w5 - w4; c_cp += c5 - c4; w_run += wr1 - wr0;
        }
    }
    int n = nchunk - 1;
    double num = 0, den = 0;
    for (size_t k = 0; k < (size_t)nchunk * eo_n; k++) {
        double d = zc_eo[k] - ref_eo[k];
        num += d * d;
        den += (double)ref_eo[k] * ref_eo[k];
    }
    printf("encoder_out zero-copy vs reference over %d chunks: rel_err=%.6f\n", nchunk, sqrt(num / den));
    printf("per chunk (640 ms audio):\n");
    printf("  reference (inputs_set+run+outputs_get+reorder, fp16 states)  wall %6.2f ms  cpu %6.2f ms\n", w_ref / n, c_ref / n);
    printf("  zero-copy total                                             wall %6.2f ms  cpu %6.2f ms\n", w_zc / n, c_zc / n);
    printf("    of which rknn_run wall %6.2f ms; state copy wall %6.2f ms cpu %6.2f ms\n", w_run / n, w_cp / n, c_cp / n);
    return 0;
}

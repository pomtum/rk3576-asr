#include "recognizer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "asr_profile.h"
#include "encoder_zc.h"
#include "fbank.h"
#include "rknn_api.h"

#define CONTEXT 2
#define MAX_TOKENS 4096
#define MAX_VOCAB 8192
#define FRAME_SEC 0.04f         // 编码器输出一帧对应 40 ms

// 断句规则，与 sherpa-onnx 默认一致
#define RULE1_SILENCE 2.4f      // 一个字都没出时，静音这么久算一句结束
#define RULE2_SILENCE 1.4f      // 出过字后，尾部静音这么久算一句结束
#define RULE3_MAX_LEN 20.0f     // 一句最长时长

typedef struct {
    rknn_context ctx;
    int in_elems, out_elems;
} Small;

struct Recognizer {
    Fbank *fb;
    EncoderZc *enc;
    Small dec, joi;
    char *vocab[MAX_VOCAB];
    int vocab_size;
    int unk_id;                  // 词表里的 <unk>，没有则为 -1
    size_t next_frame;           // 下一块从第几帧特征开始
    float *x, *enc_out, *dec_out, *logit;
    int64_t hyp[MAX_TOKENS + CONTEXT];
    int n_hyp;                   // 含开头的 CONTEXT 个占位
    int trailing_blanks, utt_frames;
    char *text;
    size_t text_cap;
};

static int small_load(Small *s, const void *model, size_t len) {
    if (rknn_init(&s->ctx, (void *)model, (uint32_t)len, 0, NULL) != RKNN_SUCC) return -1;
    rknn_tensor_attr a;
    memset(&a, 0, sizeof(a));
    rknn_query(s->ctx, RKNN_QUERY_INPUT_ATTR, &a, sizeof(a));
    s->in_elems = (int)a.n_elems;
    memset(&a, 0, sizeof(a));
    rknn_query(s->ctx, RKNN_QUERY_OUTPUT_ATTR, &a, sizeof(a));
    s->out_elems = (int)a.n_elems;
    return 0;
}

static void small_free(Small *s) {
    if (s->ctx) rknn_destroy(s->ctx);
}

static int run_decoder(Recognizer *r) {
    PROF_BEGIN();
    rknn_input in;
    memset(&in, 0, sizeof(in));
    in.index = 0;
    in.type = RKNN_TENSOR_INT64;
    in.fmt = RKNN_TENSOR_UNDEFINED;
    in.buf = r->hyp + r->n_hyp - CONTEXT;
    in.size = CONTEXT * sizeof(int64_t);
    rknn_output out;
    memset(&out, 0, sizeof(out));
    out.want_float = 1;
    out.is_prealloc = 1;
    out.buf = r->dec_out;
    out.size = (uint32_t)r->dec.out_elems * 4;
    if (rknn_inputs_set(r->dec.ctx, 1, &in) || rknn_run(r->dec.ctx, NULL) || rknn_outputs_get(r->dec.ctx, 1, &out, NULL))
        return -1;
    PROF_END(PROF_DECODER);
    return 0;
}

static int run_joiner(Recognizer *r, const float *enc_frame) {
    PROF_BEGIN();
    rknn_input in[2];
    memset(in, 0, sizeof(in));
    for (int i = 0; i < 2; i++) {
        in[i].index = i;
        in[i].type = RKNN_TENSOR_FLOAT32;
        in[i].fmt = RKNN_TENSOR_UNDEFINED;
        in[i].size = (uint32_t)r->dec.out_elems * 4;
    }
    in[0].buf = (void *)enc_frame;
    in[1].buf = r->dec_out;
    rknn_output out;
    memset(&out, 0, sizeof(out));
    out.want_float = 1;
    out.is_prealloc = 1;
    out.buf = r->logit;
    out.size = (uint32_t)r->joi.out_elems * 4;
    if (rknn_inputs_set(r->joi.ctx, 2, in) || rknn_run(r->joi.ctx, NULL) || rknn_outputs_get(r->joi.ctx, 1, &out, NULL))
        return -1;
    PROF_END(PROF_JOINER);
    return 0;
}

// tokens.txt：每行 "<符号> <id>"
static int load_tokens(Recognizer *r, const char *text) {
    while (*text) {
        const char *eol = strchr(text, '\n');
        size_t len = eol ? (size_t)(eol - text) : strlen(text);
        while (len > 0 && (text[len - 1] == '\r' || text[len - 1] == ' ')) len--;
        const char *sp = NULL;
        for (size_t i = len; i > 0; i--) if (text[i - 1] == ' ') { sp = text + i - 1; break; }
        if (sp) {
            int id = atoi(sp + 1);
            if (id >= 0 && id < MAX_VOCAB) {
                r->vocab[id] = strndup(text, (size_t)(sp - text));
                if (id + 1 > r->vocab_size) r->vocab_size = id + 1;
            }
        }
        if (!eol) break;
        text = eol + 1;
    }
    r->unk_id = -1;
    for (int i = 0; i < r->vocab_size; i++)
        if (r->vocab[i] && !strcmp(r->vocab[i], "<unk>")) r->unk_id = i;
    return r->vocab_size > 0 ? 0 : -1;
}

static void start_hyp(Recognizer *r) {
    for (int i = 0; i < CONTEXT; i++) r->hyp[i] = -1;
    r->hyp[CONTEXT - 1] = 0;  // blank
    r->n_hyp = CONTEXT;
    r->trailing_blanks = 0;
    r->utt_frames = 0;
}

void recognizer_destroy(Recognizer *r) {
    if (!r) return;
    fbank_destroy(r->fb);
    encoder_zc_destroy(r->enc);
    small_free(&r->dec);
    small_free(&r->joi);
    for (int i = 0; i < r->vocab_size; i++) free(r->vocab[i]);
    free(r->x);
    free(r->enc_out);
    free(r->dec_out);
    free(r->logit);
    free(r->text);
    free(r);
}

Recognizer *recognizer_create(const ModelData *m, char *err, size_t err_len) {
    Recognizer *r = calloc(1, sizeof(Recognizer));
    r->enc = encoder_zc_create(m->encoder, m->encoder_len, m->layout, err, err_len);
    if (!r->enc) goto fail;
    if (small_load(&r->dec, m->decoder, m->decoder_len)) { snprintf(err, err_len, "cannot load decoder"); goto fail; }
    if (small_load(&r->joi, m->joiner, m->joiner_len)) { snprintf(err, err_len, "cannot load joiner"); goto fail; }
    if (load_tokens(r, m->tokens)) { snprintf(err, err_len, "cannot parse tokens"); goto fail; }
    int T = encoder_zc_chunk_frames(r->enc);
    r->x = malloc((size_t)T * FBANK_DIM * sizeof(float));
    r->enc_out = malloc((size_t)encoder_zc_out_frames(r->enc) * encoder_zc_out_dim(r->enc) * sizeof(float));
    r->dec_out = malloc((size_t)r->dec.out_elems * sizeof(float));
    r->logit = malloc((size_t)r->joi.out_elems * sizeof(float));
    r->fb = fbank_create();
    start_hyp(r);
    if (run_decoder(r)) { snprintf(err, err_len, "decoder run failed"); goto fail; }
    return r;
fail:
    recognizer_destroy(r);
    return NULL;
}

static int decode_chunk(Recognizer *r) {
    int T = encoder_zc_chunk_frames(r->enc);
    for (int t = 0; t < T; t++) memcpy(r->x + (size_t)t * FBANK_DIM, fbank_frame(r->fb, r->next_frame + t), FBANK_DIM * sizeof(float));
    if (encoder_zc_run(r->enc, r->x, r->enc_out)) return -1;
    int frames = encoder_zc_out_frames(r->enc), dim = encoder_zc_out_dim(r->enc);
    for (int t = 0; t < frames; t++) {
        if (run_joiner(r, r->enc_out + (size_t)t * dim)) return -1;
        int y = 0;
        for (int k = 1; k < r->joi.out_elems; k++) if (r->logit[k] > r->logit[y]) y = k;
        // 空白（0）与 <unk> 都当作没出字
        if (y != 0 && y != r->unk_id && r->n_hyp < MAX_TOKENS) {
            r->hyp[r->n_hyp++] = y;
            r->trailing_blanks = 0;
            if (run_decoder(r)) return -1;
        } else {
            r->trailing_blanks++;
        }
        r->utt_frames++;
    }
    r->next_frame += (size_t)encoder_zc_chunk_shift(r->enc);
    fbank_discard_before(r->fb, r->next_frame);
    return 0;
}

static int decode_ready(Recognizer *r) {
    int T = encoder_zc_chunk_frames(r->enc);
    while (fbank_num_frames(r->fb) >= r->next_frame + (size_t)T)
        if (decode_chunk(r)) return -1;
    return 0;
}

int recognizer_accept(Recognizer *r, const float *samples, size_t n) {
    PROF_BEGIN();
    fbank_accept(r->fb, samples, n);
    PROF_END(PROF_FBANK);
    return decode_ready(r);
}

int recognizer_finish(Recognizer *r) {
    // 补 0.6 s 静音，让最后一段语音进入完整的块
    static const float silence[9600];
    fbank_accept(r->fb, silence, sizeof(silence) / sizeof(silence[0]));
    fbank_finish(r->fb);
    return decode_ready(r);
}

const char *recognizer_text(Recognizer *r) {
    size_t need = 1;
    for (int i = CONTEXT; i < r->n_hyp; i++) need += strlen(r->vocab[r->hyp[i]] ? r->vocab[r->hyp[i]] : "") + 1;
    if (need > r->text_cap) { r->text_cap = need * 2; r->text = realloc(r->text, r->text_cap); }
    size_t len = 0;
    for (int i = CONTEXT; i < r->n_hyp; i++) {
        const char *t = r->vocab[r->hyp[i]];
        if (!t) continue;
        // 字节回退单元 "<0xXX>"：还原成原始字节（词表外的汉字会拆成几个 UTF-8 字节输出）
        if (strlen(t) == 6 && !strncmp(t, "<0x", 3) && t[5] == '>') {
            r->text[len++] = (char)strtol(t + 3, NULL, 16);
            continue;
        }
        // BPE 词首标记 "▁"（UTF-8: E2 96 81）转成空格
        while (*t) {
            if ((unsigned char)t[0] == 0xE2 && (unsigned char)t[1] == 0x96 && (unsigned char)t[2] == 0x81) {
                if (len > 0) r->text[len++] = ' ';
                t += 3;
            } else {
                r->text[len++] = *t++;
            }
        }
    }
    // 两个非 ASCII 字符（汉字等）之间的空格去掉；英文单词之间的保留
    size_t out = 0;
    for (size_t k = 0; k < len; k++) {
        if (r->text[k] == ' ' && out > 0 && k + 1 < len &&
            ((unsigned char)r->text[out - 1] & 0x80) && ((unsigned char)r->text[k + 1] & 0x80))
            continue;
        r->text[out++] = r->text[k];
    }
    r->text[out] = 0;
    return r->text;
}

int recognizer_is_endpoint(const Recognizer *r) {
    float silence = r->trailing_blanks * FRAME_SEC, length = r->utt_frames * FRAME_SEC;
    int has_text = r->n_hyp > CONTEXT;
    if (!has_text && silence >= RULE1_SILENCE) return 1;
    if (has_text && silence >= RULE2_SILENCE) return 1;
    if (length >= RULE3_MAX_LEN) return 1;
    return 0;
}

void recognizer_reset_utterance(Recognizer *r) {
    // 保留最后 CONTEXT 个 token 作为解码器上下文，让下一句衔接自然
    int64_t last[CONTEXT];
    for (int i = 0; i < CONTEXT; i++) last[i] = r->hyp[r->n_hyp - CONTEXT + i];
    start_hyp(r);
    for (int i = 0; i < CONTEXT; i++) r->hyp[i] = last[i];
    run_decoder(r);
}

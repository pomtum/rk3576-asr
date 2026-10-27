// Run the app's recognizer (same sources as the APK) on the device: feeds wav files in 100 ms pieces and
// prints the text, RTF and CPU time. Build with -DASR_PROFILE for a per-stage breakdown.
// Usage: asrcli <model_dir> <wav>...   (model_dir: encoder/decoder/joiner.rknn, tokens.txt, encoder_layout.txt)
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "asr_profile.h"
#include "recognizer.h"

double g_prof_cpu[PROF_N], g_prof_wall[PROF_N];
static const char *PROF_NAMES[PROF_N] = {"fbank", "enc_in", "enc_run", "enc_sync", "enc_state", "joiner", "decoder"};

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static double cpu_ms(void) { struct timespec t; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

static float *read_wav(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *b = malloc((size_t)len);
    fread(b, 1, (size_t)len, f);
    fclose(f);
    long pos = 12;
    while (pos + 8 <= len) {
        uint32_t size;
        memcpy(&size, b + pos + 4, 4);
        if (!memcmp(b + pos, "data", 4)) {
            *n = size / 2;
            float *s = malloc(*n * sizeof(float));
            for (size_t i = 0; i < *n; i++) { int16_t v; memcpy(&v, b + pos + 8 + i * 2, 2); s[i] = v / 32768.0f; }
            free(b);
            return s;
        }
        pos += 8 + size;
    }
    fprintf(stderr, "no data chunk\n");
    exit(1);
}

static void *slurp(const char *dir, const char *name, size_t *len) {
    char p[1024];
    snprintf(p, sizeof(p), "%s/%s", dir, name);
    FILE *f = fopen(p, "rb");
    if (!f) { perror(p); exit(1); }
    fseek(f, 0, SEEK_END);
    *len = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = malloc(*len + 1);
    fread(b, 1, *len, f);
    b[*len] = 0;
    fclose(f);
    return b;
}

static ModelData g_model;

static Recognizer *create(const char *dir, char *err, size_t err_len) {
    if (!g_model.encoder) {
        size_t n;
        g_model.encoder = slurp(dir, "encoder.rknn", &g_model.encoder_len);
        g_model.decoder = slurp(dir, "decoder.rknn", &g_model.decoder_len);
        g_model.joiner = slurp(dir, "joiner.rknn", &g_model.joiner_len);
        g_model.tokens = slurp(dir, "tokens.txt", &n);
        g_model.layout = slurp(dir, "encoder_layout.txt", &n);
    }
    return recognizer_create(&g_model, err, err_len);
}

int main(int argc, char **argv) {
    char err[256];
    double w0 = now_ms(), c0 = cpu_ms();
    Recognizer *r = create(argv[1], err, sizeof(err));
    if (!r) { printf("create failed: %s\n", err); return 1; }
    printf("load: wall %.0f ms cpu %.0f ms\n", now_ms() - w0, cpu_ms() - c0);
    double total_audio = 0, total_wall = 0, total_cpu = 0;
    for (int a = 2; a < argc; a++) {
        size_t n;
        float *s = read_wav(argv[a], &n);
        memset(g_prof_cpu, 0, sizeof(g_prof_cpu));
        memset(g_prof_wall, 0, sizeof(g_prof_wall));
        double w1 = now_ms(), c1 = cpu_ms();
        for (size_t i = 0; i < n; i += 1600) recognizer_accept(r, s + i, n - i < 1600 ? n - i : 1600);
        recognizer_finish(r);
        double w2 = now_ms(), c2 = cpu_ms();
        double dur = n / 16000.0;
        printf("%s: audio %.2fs wall %.0f ms (rtf %.3f) cpu %.0f ms (%.1f%% of one core) endpoint=%d\n  %s\n", argv[a], dur,
               w2 - w1, (w2 - w1) / 1000 / dur, c2 - c1, (c2 - c1) / 10 / dur, recognizer_is_endpoint(r), recognizer_text(r));
        for (int k = 0; k < PROF_N; k++)
            printf("    %-10s cpu %7.1f ms  wall %7.1f ms  (cpu %.1f ms per s of audio)\n", PROF_NAMES[k], g_prof_cpu[k], g_prof_wall[k], g_prof_cpu[k] / dur);
        total_audio += dur; total_wall += w2 - w1; total_cpu += c2 - c1;
        free(s);
        // 下一段音频用新的识别器（真实场景里 finish 之后流就结束了）
        recognizer_destroy(r);
        r = create(argv[1], err, sizeof(err));
    }
    printf("TOTAL audio %.2fs rtf %.3f cpu %.1f%% of one core\n", total_audio, total_wall / 1000 / total_audio,
           total_cpu / 10 / total_audio);
    recognizer_destroy(r);
    return 0;
}

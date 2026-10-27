// NPU sanity check: classify pre-decoded 224x224 RGB images with mobilenet_v1 (from rknn-toolkit2 examples)
// and print the top-5. Usage: npucheck <model.rknn> <img.rgb> [img2.rgb ...]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "rknn_api.h"

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

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.rknn img.rgb...\n", argv[0]); return 2; }
    size_t mlen;
    void *model = read_all(argv[1], &mlen);
    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, model, (uint32_t)mlen, 0, NULL);
    if (ret != RKNN_SUCC) { printf("rknn_init fail %d\n", ret); return 1; }

    rknn_sdk_version ver;
    rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver));
    printf("api %s driver %s\n", ver.api_version, ver.drv_version);

    rknn_tensor_attr out_attr;
    memset(&out_attr, 0, sizeof(out_attr));
    out_attr.index = 0;
    rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr));

    for (int a = 2; a < argc; a++) {
        size_t ilen;
        void *img = read_all(argv[a], &ilen);
        rknn_input in;
        memset(&in, 0, sizeof(in));
        in.index = 0;
        in.type = RKNN_TENSOR_UINT8;
        in.fmt = RKNN_TENSOR_NHWC;
        in.size = (uint32_t)ilen;
        in.buf = img;
        rknn_inputs_set(ctx, 1, &in);
        double t0 = now_ms();
        ret = rknn_run(ctx, NULL);
        double t1 = now_ms();
        if (ret != RKNN_SUCC) { printf("rknn_run fail %d\n", ret); return 1; }
        rknn_output out;
        memset(&out, 0, sizeof(out));
        out.want_float = 1;
        rknn_outputs_get(ctx, 1, &out, NULL);
        float *p = (float *)out.buf;
        uint32_t n = out.size / sizeof(float);
        printf("%s run %.2f ms top5:", argv[a], t1 - t0);
        for (int k = 0; k < 5; k++) {
            uint32_t best = 0;
            for (uint32_t i = 1; i < n; i++) if (p[i] > p[best]) best = i;
            printf(" [%u] %.4f", best, p[best]);
            p[best] = -1e30f;
        }
        printf("\n");
        rknn_outputs_release(ctx, 1, &out);
        free(img);
    }
    rknn_destroy(ctx);
    return 0;
}

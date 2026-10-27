// Run one encoder chunk with RKNN per-op profiling enabled and print the per-op table.
// Usage: perfdetail <encoder.rknn> > perf.txt
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rknn_api.h"

int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb");
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *model = malloc(len);
    fread(model, 1, len, f);
    fclose(f);
    rknn_context ctx;
    if (rknn_init(&ctx, model, (uint32_t)len, RKNN_FLAG_COLLECT_PERF_MASK, NULL) != RKNN_SUCC) { printf("init fail\n"); return 1; }
    rknn_input_output_num io;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
    rknn_input *in = calloc(io.n_input, sizeof(rknn_input));
    for (uint32_t i = 0; i < io.n_input; i++) {
        rknn_tensor_attr a;
        memset(&a, 0, sizeof(a));
        a.index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &a, sizeof(a));
        int is64 = a.type == RKNN_TENSOR_INT64;
        in[i].index = i;
        in[i].type = is64 ? RKNN_TENSOR_INT64 : RKNN_TENSOR_FLOAT32;
        in[i].fmt = a.fmt;
        in[i].size = a.n_elems * (is64 ? 8 : 4);
        in[i].buf = calloc(a.n_elems, 8);
    }
    for (int k = 0; k < 3; k++) {
        rknn_inputs_set(ctx, io.n_input, in);
        rknn_run(ctx, NULL);
    }
    rknn_perf_detail pd;
    memset(&pd, 0, sizeof(pd));
    rknn_query(ctx, RKNN_QUERY_PERF_DETAIL, &pd, sizeof(pd));
    rknn_perf_run pr;
    rknn_query(ctx, RKNN_QUERY_PERF_RUN, &pr, sizeof(pr));
    printf("run_duration %lld us\n", (long long)pr.run_duration);
    // 原样输出明细表，交给 PC 端汇总
    fwrite(pd.perf_data, 1, pd.data_len, stdout);
    return 0;
}

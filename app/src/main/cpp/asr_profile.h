// 分段 CPU / 墙钟计时，仅在定义 ASR_PROFILE 时生效（命令行测试用）
#pragma once

#ifdef ASR_PROFILE
#include <time.h>

enum { PROF_FBANK, PROF_ENC_IN, PROF_ENC_RUN, PROF_ENC_SYNC, PROF_ENC_STATE, PROF_JOINER, PROF_DECODER, PROF_N };
extern double g_prof_cpu[PROF_N], g_prof_wall[PROF_N];

static inline double prof_cpu(void) { struct timespec t; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static inline double prof_wall(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
#define PROF_BEGIN() double _pc = prof_cpu(), _pw = prof_wall()
#define PROF_END(k) do { g_prof_cpu[k] += prof_cpu() - _pc; g_prof_wall[k] += prof_wall() - _pw; _pc = prof_cpu(); _pw = prof_wall(); } while (0)
#else
#define PROF_BEGIN() do {} while (0)
#define PROF_END(k) do {} while (0)
#endif

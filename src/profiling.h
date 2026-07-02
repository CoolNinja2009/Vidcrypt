#ifndef VIDCRYPT_PROFILING_H
#define VIDCRYPT_PROFILING_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef ENABLE_PROFILING

#if defined(_MSC_VER)
#include <intrin.h>
static inline int64_t prof_atomic_add(volatile int64_t *ptr, int64_t val) {
    return _InterlockedExchangeAdd64(ptr, val);
}
#else
static inline int64_t prof_atomic_add(volatile int64_t *ptr, int64_t val) {
    return __sync_fetch_and_add(ptr, val);
}
#endif

#if defined(_WIN32)
#include <windows.h>
static inline int64_t prof_clock_ns(void) {
    static LARGE_INTEGER freq = {0};
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (now.QuadPart * 1000000000LL) / freq.QuadPart;
}
#else
#include <time.h>
static inline int64_t prof_clock_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}
#endif

typedef enum {
    PROF_DECODE_VIDEO_OPEN    = 0,
    PROF_DECODE_FIRST_FRAME   = 1,
    PROF_DECODE_CALIBRATION   = 2,
    PROF_DECODE_HEADER_PARSE  = 3,
    PROF_DECODE_FRAME_ACQUIRE = 4,
    PROF_DECODE_BIT_EXTRACT   = 5,
    PROF_DECODE_RS_DECODE     = 6,
    PROF_DECODE_FILE_WRITE    = 7,
    PROF_DECODE_SHA256        = 8,
    PROF_ENCODE_FILE_READ     = 9,
    PROF_ENCODE_RS_ENCODE     = 10,
    PROF_ENCODE_FRAME_GENERATE= 11,
    PROF_ENCODE_VIDEO_WRITE   = 12,
    PROF_ENCODE_HEADER_BUILD  = 13,
    PROF_COUNT                = 14
} ProfStageId;

extern const char *PROF_STAGE_NAMES[PROF_COUNT];

typedef struct {
    int64_t total_ns;
    int64_t min_ns;
    int64_t max_ns;
    int64_t call_count;
    int64_t start_ns;
    int     depth;
} ProfAccum;

typedef struct {
    volatile int64_t total_ns;
    volatile int64_t call_count;
} ProfAtomicAccum;

typedef struct {
    ProfAccum      stages[PROF_COUNT];
    ProfAtomicAccum worker_stages[PROF_COUNT];
    int64_t        wall_start_ns;
    int            active;
    char           pipeline_name[64];
} ProfCtx;

static inline void prof_init(ProfCtx *ctx, const char *name) {
    memset(ctx, 0, sizeof(ProfCtx));
    ctx->wall_start_ns = prof_clock_ns();
    ctx->active = 1;
    if (name) {
        size_t len = strlen(name);
        if (len >= sizeof(ctx->pipeline_name)) len = sizeof(ctx->pipeline_name) - 1;
        memcpy(ctx->pipeline_name, name, len);
        ctx->pipeline_name[len] = '\0';
    }
}

static inline void prof_stage_begin(ProfCtx *ctx, ProfStageId id) {
    if (!ctx->active) return;
    ProfAccum *s = &ctx->stages[id];
    if (s->depth++ == 0) s->start_ns = prof_clock_ns();
}

static inline void prof_stage_end(ProfCtx *ctx, ProfStageId id) {
    if (!ctx->active) return;
    ProfAccum *s = &ctx->stages[id];
    if (--s->depth == 0) {
        int64_t elapsed = prof_clock_ns() - s->start_ns;
        s->total_ns += elapsed;
        s->call_count++;
        if (elapsed < s->min_ns) s->min_ns = elapsed;
        if (elapsed > s->max_ns) s->max_ns = elapsed;
    }
}

static inline void prof_worker_add(ProfCtx *ctx, ProfStageId id, int64_t elapsed_ns) {
    ProfAtomicAccum *w = &ctx->worker_stages[id];
    prof_atomic_add(&w->total_ns, elapsed_ns);
    prof_atomic_add(&w->call_count, 1);
}

static inline int64_t prof_worker_total(const ProfCtx *ctx, ProfStageId id) {
    return prof_atomic_add((volatile int64_t *)&ctx->worker_stages[id].total_ns, 0);
}

static inline int64_t prof_worker_calls(const ProfCtx *ctx, ProfStageId id) {
    return prof_atomic_add((volatile int64_t *)&ctx->worker_stages[id].call_count, 0);
}

typedef struct {
    ProfStageId id;
    double      total_ms;
    double      avg_ms;
    double      min_ms;
    double      max_ms;
    int64_t     call_count;
    double      pct_of_total;
} ProfReportLine;

typedef struct {
    char           pipeline_name[64];
    ProfReportLine lines[PROF_COUNT];
    int            line_count;
    double         wall_sec;
    double         accounted_ms;
} ProfReport;

void prof_report_build(ProfCtx *ctx, ProfReport *report);
void prof_report_print(const ProfReport *report);
int  prof_report_save_csv(const ProfReport *report, const char *path);

#else /* !ENABLE_PROFILING */

#define prof_init(ctx, name)            ((void)0)
#define prof_stage_begin(ctx, id)       ((void)0)
#define prof_stage_end(ctx, id)         ((void)0)
#define prof_worker_add(ctx, id, ns)    ((void)0)
#define prof_report_build(ctx, r)       ((void)0)
#define prof_report_print(r)            ((void)0)
#define prof_report_save_csv(r, p)      (-1)

typedef struct { int dummy; } ProfCtx;
typedef struct { int dummy; } ProfReport;

#endif /* ENABLE_PROFILING */

#ifdef __cplusplus
}
#endif

#endif /* VIDCRYPT_PROFILING_H */

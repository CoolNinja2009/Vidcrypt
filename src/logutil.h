#ifndef VIDCRYPT_LOGUTIL_H
#define VIDCRYPT_LOGUTIL_H

/* ─── Fine-grained logging utility ───────────────────────────────────
 * Provides structured logging to timestamped files in the log/ directory.
 *
 * Usage:
 *   Enable at compile time: -DENABLE_LOGGING=1   (or cmake -DENABLE_LOGGING=ON)
 *   When disabled, all LOG_* macros expand to nothing — zero runtime cost.
 *
 * Log file format:
 *   [HH:MM:SS.mmm] [LEVEL] [thread] [mem=XX.XMB] [file:line]   message
 *
 * Features:
 *   - Thread-safe (mutex-protected writes)
 *   - Auto-rotating filenames per run (logs/vidcrypt_YYYY-MM-DD_HH-MM-SS.log)
 *   - Indentation for nested operations (LOG_PUSH / LOG_POP)
 *   - Timers with automatic indentation (LOG_TIMER)
 *   - Metrics (LOG_METRIC)
 *   - Memory tracking (Windows: GetProcessMemoryInfo, Linux: /proc/self/status)
 *   - Caller file:line capture
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Compile-time guard ────────────────────────────────────────────
 * Define VIDCRYPT_LOG_ENABLE to activate. All log macros become no-ops
 * when this is not defined. */

#if defined(ENABLE_LOGGING) && ENABLE_LOGGING == 1
#  define VIDCRYPT_LOG_ENABLE 1
#endif


/* ═══════════════════════════════════════════════════════════════════ *
 *  ACTIVE LOGGING (VIDCRYPT_LOG_ENABLE defined)                      *
 * ═══════════════════════════════════════════════════════════════════ */

#ifdef VIDCRYPT_LOG_ENABLE

/* ─── Log levels ─────────────────────────────────────────────────── */
typedef enum {
    LOG_DEBUG = 0,
    LOG_INFO  = 1,
    LOG_WARN  = 2,
    LOG_ERROR = 3,
} LogLevel;

/* ─── Lifecycle ──────────────────────────────────────────────────── */
/* Initialize the logger. Creates log/ directory and opens a new
 * timestamped log file. Thread-safe — multiple calls are fine. */
void log_init(void);

/* Close the log file and flush all pending output. */
void log_close(void);

/* ─── Core write ─────────────────────────────────────────────────── */
void log_write(LogLevel level, const char *file, int line,
               const char *fmt, ...)
    #ifdef __GNUC__
    __attribute__((format(printf, 4, 5)))
    #endif
    ;

/* ─── Convenience macros ─────────────────────────────────────────── */
#define LOG_DEBUG(...)    log_write(LOG_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_INFO(...)     log_write(LOG_INFO,  __FILE__, __LINE__, __VA_ARGS__)
#define LOG_WARN(...)     log_write(LOG_WARN,  __FILE__, __LINE__, __VA_ARGS__)
#define LOG_ERROR(...)    log_write(LOG_ERROR, __FILE__, __LINE__, __VA_ARGS__)

/* ─── Exception helper ───────────────────────────────────────────── */
/* Logs a message plus the current errno string. */
void log_perror(const char *file, int line, const char *msg);

#define LOG_PERROR(msg)   log_perror(__FILE__, __LINE__, (msg))

/* ─── Indentation ────────────────────────────────────────────────── */
/* Push increases indent by 2 spaces. Pop decreases. */
void log_indent_push(void);
void log_indent_pop(void);

#define LOG_PUSH()        log_indent_push()
#define LOG_POP()         log_indent_pop()

/* ─── Separator ──────────────────────────────────────────────────── */
/* Writes a full-width separator line with an optional title. */
void log_separator(const char *title);

#define LOG_SEPARATOR(title) log_separator(title)

/* ─── Timers ─────────────────────────────────────────────────────── */
/* Lightweight RAII-style timer. Pushes indent on start, pops on end.
 * Logs elapsed wall-clock time with ms precision. */
typedef struct {
    const char *label;
    double      start_sec;
    bool        active;
} LogTimer;

/* Start a timer. Logs "BEGIN: <label>" and pushes indent. */
void log_timer_start(LogTimer *timer, const char *label);

/* Stop a timer. Pops indent, logs "END: <label> (X.XXXs)". */
void log_timer_stop(LogTimer *timer);

#define LOG_TIMER_START(t, label)  log_timer_start(&(t), (label))
#define LOG_TIMER_END(t)           log_timer_stop(&(t))

/* ─── Metrics ────────────────────────────────────────────────────── */
/* Log a named metric value: "[METRIC] name = value" */
void log_metric(const char *name, double value);

#define LOG_METRIC(name, value)   log_metric((name), (double)(value))


#else  /* !VIDCRYPT_LOG_ENABLE */


/* ═══════════════════════════════════════════════════════════════════ *
 *  NO-OP STUBS — all macros expand to nothing. Zero overhead.        *
 * ═══════════════════════════════════════════════════════════════════ */

#define LOG_DEBUG(...)            ((void)0)
#define LOG_INFO(...)             ((void)0)
#define LOG_WARN(...)             ((void)0)
#define LOG_ERROR(...)            ((void)0)
#define LOG_PERROR(msg)           ((void)0)
#define LOG_PUSH()                ((void)0)
#define LOG_POP()                 ((void)0)
#define LOG_SEPARATOR(title)      ((void)0)
#define LOG_TIMER_START(t, label) ((void)0)
#define LOG_TIMER_END(t)          ((void)0)
#define LOG_METRIC(name, value)   ((void)0)

/* Inline stubs so call sites still compile but are eliminated by the
 * compiler's dead-code removal. */
static inline void log_init(void) {}
static inline void log_close(void) {}

/* Timer struct definition still available for compilation. */
typedef struct {
    const char *label;
    double      start_sec;
    bool        active;
} LogTimer;

static inline void log_timer_start(LogTimer *t, const char *l) {
    (void)t; (void)l;
}
static inline void log_timer_stop(LogTimer *t) { (void)t; }

/* Allow unused-result to be silenced without warnings. */
static inline void log_write(LogLevel l, const char *f, int ln,
                              const char *fmt, ...) {
    (void)l; (void)f; (void)ln; (void)fmt;
}
static inline void log_perror(const char *f, int ln, const char *m) {
    (void)f; (void)ln; (void)m;
}
static inline void log_indent_push(void) {}
static inline void log_indent_pop(void) {}
static inline void log_separator(const char *t) { (void)t; }
static inline void log_metric(const char *n, double v) {
    (void)n; (void)v;
}

#endif /* VIDCRYPT_LOG_ENABLE */

#ifdef __cplusplus
}
#endif

#endif /* VIDCRYPT_LOGUTIL_H */

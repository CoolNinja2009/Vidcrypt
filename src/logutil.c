#include "logutil.h"

#if defined(VIDCRYPT_LOG_ENABLE)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <errno.h>

#ifdef _WIN32
#  include <windows.h>
#  include <psapi.h>
#  include <process.h>
#  define getpid _getpid
#  define strcasecmp _stricmp
#else
#  include <unistd.h>
#  include <sys/stat.h>
#  include <sys/time.h>
#  include <pthread.h>
#endif

/* ─── Internal state ─────────────────────────────────────────────── */
static FILE *g_log_file   = NULL;
static int   g_indent     = 0;
static int   g_init_count = 0;    /* allow multiple init/close pairs */

#ifdef _WIN32
static CRITICAL_SECTION g_lock;
static int              g_lock_init = 0;
#else
static pthread_mutex_t  g_lock = PTHREAD_MUTEX_INITIALIZER;
#endif

/* ─── Platform helpers ───────────────────────────────────────────── */

static double now_sec(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq;
    static int freq_queried = 0;
    LARGE_INTEGER counter;
    if (!freq_queried) {
        QueryPerformanceFrequency(&freq);
        freq_queried = 1;
    }
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

static void get_timestamp_ms(char *buf, int buf_size) {
#ifdef _WIN32
    SYSTEMTIME st;
    GetLocalTime(&st);
    _snprintf(buf, (size_t)buf_size, "%02d:%02d:%02d.%03d",
              st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
#else
    struct timeval tv;
    struct tm     tm;
    gettimeofday(&tv, NULL);
    localtime_r(&tv.tv_sec, &tm);
    strftime(buf, (size_t)buf_size, "%H:%M:%S", &tm);
    int len = (int)strlen(buf);
    _snprintf(buf + len, (size_t)(buf_size - len), ".%03d",
              (int)(tv.tv_usec / 1000));
#endif
}

static void get_filename_timestamp(char *buf, int buf_size) {
#ifdef _WIN32
    SYSTEMTIME st;
    GetLocalTime(&st);
    _snprintf(buf, (size_t)buf_size, "%04d-%02d-%02d_%02d-%02d-%02d",
              st.wYear, st.wMonth, st.wDay,
              st.wHour, st.wMinute, st.wSecond);
#else
    struct timeval tv;
    struct tm     tm;
    gettimeofday(&tv, NULL);
    localtime_r(&tv.tv_sec, &tm);
    strftime(buf, (size_t)buf_size, "%Y-%m-%d_%H-%M-%S", &tm);
#endif
}

static double memory_mb(void) {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    pmc.cb = sizeof(pmc);
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return (double)pmc.WorkingSetSize / (1024.0 * 1024.0);
    return -1.0;
#else
    static long page_size = 0;
    if (page_size == 0) page_size = sysconf(_SC_PAGESIZE);
    FILE *f = fopen("/proc/self/statm", "r");
    if (!f) return -1.0;
    long size;
    if (fscanf(f, "%ld", &size) != 1) { fclose(f); return -1.0; }
    fclose(f);
    return (double)size * (double)page_size / (1024.0 * 1024.0);
#endif
}

static const char *level_label(LogLevel level) {
    switch (level) {
        case LOG_DEBUG: return "DEBUG";
        case LOG_INFO:  return "INFO";
        case LOG_WARN:  return "WARN";
        case LOG_ERROR: return "ERROR";
        default:        return "?????";
    }
}

#ifdef _WIN32
static const char *thread_name(void) {
    DWORD tid = GetCurrentThreadId();
    static __declspec(thread) char buf[32];
    _snprintf(buf, sizeof(buf), "T-%04lx", (unsigned long)tid);
    return buf;
}
#else
static const char *thread_name(void) {
    static __thread char buf[32];
    snprintf(buf, sizeof(buf), "T-%04lx", (unsigned long)pthread_self());
    return buf;
}
#endif

/* ─── Lock helpers ───────────────────────────────────────────────── */

static void lock_init(void) {
#ifdef _WIN32
    if (!g_lock_init) {
        InitializeCriticalSection(&g_lock);
        g_lock_init = 1;
    }
#endif
}

static void lock_acquire(void) {
#ifdef _WIN32
    EnterCriticalSection(&g_lock);
#else
    pthread_mutex_lock(&g_lock);
#endif
}

static void lock_release(void) {
#ifdef _WIN32
    LeaveCriticalSection(&g_lock);
#else
    pthread_mutex_unlock(&g_lock);
#endif
}

/* ─── Directory creation ─────────────────────────────────────────── */

static bool ensure_log_dir(const char *path) {
#ifdef _WIN32
    if (CreateDirectoryA(path, NULL)) return true;
    return GetLastError() == ERROR_ALREADY_EXISTS;
#else
    if (mkdir(path, 0755) == 0) return true;
    return errno == EEXIST;
#endif
}

/* ─── Initialization ─────────────────────────────────────────────── */

void log_init(void) {
    lock_acquire();

    if (g_log_file) {
        g_init_count++;
        lock_release();
        return;
    }

    /* Create log/ directory */
    if (!ensure_log_dir("log")) {
        /* Can't create directory — fall back to stdout-only */
        fprintf(stderr, "[logutil] WARNING: cannot create log/ directory\n");
    }

    /* Generate filename: log/vidcrypt_YYYY-MM-DD_HH-MM-SS.log */
    char ts[64];
    get_filename_timestamp(ts, sizeof(ts));

    char path[512];
    int n = snprintf(path, sizeof(path), "log/vidcrypt_%s.log", ts);
    if (n < 0 || n >= (int)sizeof(path)) {
        /* Fallback */
        snprintf(path, sizeof(path), "log/vidcrypt.log");
    }

    g_log_file = fopen(path, "w");
    if (!g_log_file) {
        fprintf(stderr, "[logutil] ERROR: cannot open log file: %s (%s)\n",
                path, strerror(errno));
        lock_release();
        return;
    }

    setvbuf(g_log_file, NULL, _IONBF, 0);  /* unbuffered for crash safety */

    g_indent     = 0;
    g_init_count = 1;

    fprintf(g_log_file, "%s\n", "=");
    fprintf(g_log_file, " VIDCRYPT LOG SESSION\n");
    fprintf(g_log_file, " PID=%d\n", (int)getpid());
    fprintf(g_log_file, " File: %s\n", path);
    fprintf(g_log_file, "%s\n\n", "=");

    lock_release();

    LOG_INFO("Logger initialized → %s", path);
}

void log_close(void) {
    lock_acquire();

    if (!g_log_file) { lock_release(); return; }

    g_init_count--;
    if (g_init_count > 0) { lock_release(); return; }

    LOG_INFO("Logger shutting down");

    fprintf(g_log_file, "\n");
    fclose(g_log_file);
    g_log_file = NULL;
    g_indent   = 0;

    lock_release();
}

/* ─── Core write ─────────────────────────────────────────────────── */

void log_write(LogLevel level, const char *file, int line,
               const char *fmt, ...) {
    /* Lazily init if not yet called — allows LOG_* before log_init() */
    if (!g_log_file) {
        /* First call triggers auto-init */
        lock_acquire();
        if (!g_log_file) {
            /* Re-check under lock */
            lock_release();
            log_init();
            /* If init still failed, write to stderr */
            if (!g_log_file) {
                va_list args;
                va_start(args, fmt);
                fprintf(stderr, "[logutil] ");
                vfprintf(stderr, fmt, args);
                fprintf(stderr, "\n");
                va_end(args);
                return;
            }
        } else {
            lock_release();
        }
    }

    char timestamp[32];
    get_timestamp_ms(timestamp, sizeof(timestamp));

    double mem = memory_mb();
    const char *mem_str;
    char mem_buf[32];
    if (mem < 0) {
        mem_str = "mem=?";
    } else {
        snprintf(mem_buf, sizeof(mem_buf), "mem=%.1fMB", mem);
        mem_str = mem_buf;
    }

    const char *tname = thread_name();
    const char *lvl   = level_label(level);

    /* Strip long path from file (single pass with strrchr) */
    const char *short_file = file;
    const char *p = strrchr(file, '/');
#ifdef _WIN32
    {
        const char *p2 = strrchr(file, '\\');
        if (p2 > p) p = p2;
    }
#endif
    if (p) short_file = p + 1;

    /* Build indent */
    char indent[64];
    int ind = g_indent * 2;
    if (ind > 60) ind = 60;
    memset(indent, ' ', (size_t)ind);
    indent[ind] = '\0';

    lock_acquire();

    /* Print to stderr (important for real-time monitoring) */
    fprintf(stderr, "[%s] [%s] [%s] [%s] [%s:%d] %s",
            timestamp, lvl, tname, mem_str, short_file, line, indent);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");

    /* Write to log file */
    if (g_log_file) {
        fprintf(g_log_file, "[%s] [%s] [%s] [%s] [%s:%d] %s",
                timestamp, lvl, tname, mem_str, short_file, line, indent);
        va_start(args, fmt);
        vfprintf(g_log_file, fmt, args);
        va_end(args);
        fprintf(g_log_file, "\n");
        fflush(g_log_file);
    }

    lock_release();
}

/* ─── perror helper ──────────────────────────────────────────────── */

void log_perror(const char *file, int line, const char *msg) {
    log_write(LOG_ERROR, file, line, "%s: %s (errno=%d)",
              msg, strerror(errno), errno);
}

/* ─── Indentation ────────────────────────────────────────────────── */

void log_indent_push(void) {
    lock_acquire();
    g_indent++;
    lock_release();
}

void log_indent_pop(void) {
    lock_acquire();
    if (g_indent > 0) g_indent--;
    lock_release();
}

/* ─── Separator ──────────────────────────────────────────────────── */

void log_separator(const char *title) {
    lock_acquire();
    if (g_log_file) {
        fprintf(g_log_file, "\n");
        fprintf(g_log_file, "%s\n", "=");
        if (title && title[0]) {
            fprintf(g_log_file, " %s\n", title);
            fprintf(g_log_file, "%s\n", "=");
        }
        fflush(g_log_file);
    }
    fprintf(stderr, "\n");
    fprintf(stderr, "%s\n", "=");
    if (title && title[0]) {
        fprintf(stderr, " %s\n", title);
        fprintf(stderr, "%s\n", "=");
    }
    lock_release();
}

/* ─── Timers ─────────────────────────────────────────────────────── */

void log_timer_start(LogTimer *timer, const char *label) {
    if (!timer) return;
    timer->label     = label;
    timer->start_sec = now_sec();
    timer->active    = true;

    log_write(LOG_INFO, __FILE__, __LINE__, "BEGIN: %s", label);
    log_indent_push();
}

void log_timer_stop(LogTimer *timer) {
    if (!timer || !timer->active) return;
    timer->active = false;

    double elapsed = now_sec() - timer->start_sec;
    log_indent_pop();
    log_write(LOG_INFO, __FILE__, __LINE__, "END: %s (%.3fs)", timer->label, elapsed);
}

/* ─── Metrics ────────────────────────────────────────────────────── */

void log_metric(const char *name, double value) {
    log_write(LOG_INFO, __FILE__, __LINE__, "[METRIC] %s = %.6g", name, value);
}

#endif /* VIDCRYPT_LOG_ENABLE */

/* When logging is disabled, this TU would be empty — add a dummy to
 * suppress MSVC warning C4206 / GCC -Wpedantic "empty translation unit". */
#if !defined(VIDCRYPT_LOG_ENABLE)
typedef int vidcrypt_logutil_dummy_t;
#endif

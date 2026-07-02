#ifndef VIDCRYPT_WIN_PTHREAD_H
#define VIDCRYPT_WIN_PTHREAD_H

#if !defined(_WIN32)
#include_next <pthread.h>
#else

#include <windows.h>
#include <stdlib.h>

typedef HANDLE pthread_t;
typedef SRWLOCK pthread_mutex_t;
typedef CONDITION_VARIABLE pthread_cond_t;

#define PTHREAD_MUTEX_INITIALIZER SRWLOCK_INIT

static inline int pthread_mutex_init(pthread_mutex_t *m, void *attr) {
    (void)attr;
    InitializeSRWLock(m);
    return 0;
}

static inline int pthread_mutex_destroy(pthread_mutex_t *m) { (void)m; return 0; }
static inline int pthread_mutex_lock(pthread_mutex_t *m) { AcquireSRWLockExclusive(m); return 0; }
static inline int pthread_mutex_unlock(pthread_mutex_t *m) { ReleaseSRWLockExclusive(m); return 0; }

static inline int pthread_cond_init(pthread_cond_t *c, void *attr) {
    (void)attr;
    InitializeConditionVariable(c);
    return 0;
}

static inline int pthread_cond_destroy(pthread_cond_t *c) { (void)c; return 0; }
static inline int pthread_cond_wait(pthread_cond_t *c, pthread_mutex_t *m) {
    SleepConditionVariableSRW(c, m, INFINITE, 0);
    return 0;
}
static inline int pthread_cond_signal(pthread_cond_t *c) { WakeConditionVariable(c); return 0; }
static inline int pthread_cond_broadcast(pthread_cond_t *c) { WakeAllConditionVariable(c); return 0; }

typedef struct PthreadStart {
    void *(*fn)(void *);
    void *arg;
} PthreadStart;

static inline DWORD WINAPI pthread_start_thunk(LPVOID p) {
    PthreadStart *start = (PthreadStart *)p;
    void *(*fn)(void *) = start->fn;
    void *arg = start->arg;
    free(start);
    fn(arg);
    return 0;
}

static inline int pthread_create(pthread_t *thread, void *attr,
                                 void *(*start_routine)(void *),
                                 void *arg) {
    (void)attr;
    PthreadStart *start = (PthreadStart *)malloc(sizeof(*start));
    if (!start) return -1;
    start->fn = start_routine;
    start->arg = arg;
    *thread = CreateThread(NULL, 0, pthread_start_thunk, start, 0, NULL);
    if (!*thread) {
        free(start);
        return -1;
    }
    return 0;
}

static inline int pthread_join(pthread_t thread, void **retval) {
    (void)retval;
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return 0;
}

#endif

#endif

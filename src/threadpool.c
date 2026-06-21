#include "threadpool.h"
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#if defined(_MSC_VER)
#include <intrin.h>
#if defined(_M_ARM64)
#define MEMORY_BARRIER() __dmb(_ARM_BARRIER_SY)
#else
#define MEMORY_BARRIER() _ReadWriteBarrier()
#endif
#else
#define MEMORY_BARRIER() __sync_synchronize()
#endif

struct ThreadPool {
    pthread_t *threads;
    int num_threads;
    volatile int running;

    WorkItem *queue;
    int capacity;
    int head;
    int tail;
    int count;

    volatile int in_flight;

    pthread_mutex_t mutex;
    pthread_cond_t  cond_nonempty;
    pthread_cond_t  cond_nonfull;
    pthread_cond_t  cond_alldone;

    work_func_t work_func;
    void *pool_user_data;
};

static void *threadpool_worker(void *arg) {
    ThreadPool *pool = (ThreadPool *)arg;
    while (1) {
        pthread_mutex_lock(&pool->mutex);
        while (pool->count == 0 && pool->running)
            pthread_cond_wait(&pool->cond_nonempty, &pool->mutex);
        if (!pool->running) { pthread_mutex_unlock(&pool->mutex); break; }

        WorkItem item = pool->queue[pool->head];
        pool->head = (pool->head + 1) % pool->capacity;
        pool->count--;
        pthread_cond_signal(&pool->cond_nonfull);
        pthread_mutex_unlock(&pool->mutex);

        if (pool->work_func) pool->work_func(&item);
        if (item.data && item.data_size > 0) free(item.data);

        pthread_mutex_lock(&pool->mutex);
        pool->in_flight--;
        if (pool->in_flight == 0) pthread_cond_signal(&pool->cond_alldone);
        pthread_mutex_unlock(&pool->mutex);
    }
    return NULL;
}

ThreadPool *threadpool_create(int num_threads, work_func_t work_func) {
    ThreadPool *pool = (ThreadPool *)calloc(1, sizeof(ThreadPool));
    if (!pool) return NULL;
    pool->num_threads = num_threads;
    pool->work_func = work_func;
    pool->running = 1;
    pool->capacity = 4096;
    pool->head = 0;
    pool->tail = 0;
    pool->count = 0;
    pool->in_flight = 0;

    pool->queue = (WorkItem *)calloc((size_t)pool->capacity, sizeof(WorkItem));
    if (!pool->queue) { free(pool); return NULL; }

    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->cond_nonempty, NULL);
    pthread_cond_init(&pool->cond_nonfull, NULL);
    pthread_cond_init(&pool->cond_alldone, NULL);

    pool->threads = (pthread_t *)calloc((size_t)num_threads, sizeof(pthread_t));
    if (!pool->threads) { free(pool->queue); free(pool); return NULL; }

    for (int i = 0; i < num_threads; ++i)
        pthread_create(&pool->threads[i], NULL, threadpool_worker, pool);

    return pool;
}

void threadpool_destroy(ThreadPool *pool) {
    if (!pool) return;
    pthread_mutex_lock(&pool->mutex);
    pool->running = 0;
    pthread_cond_broadcast(&pool->cond_nonempty);
    pthread_cond_broadcast(&pool->cond_alldone);
    pthread_mutex_unlock(&pool->mutex);
    for (int i = 0; i < pool->num_threads; ++i)
        pthread_join(pool->threads[i], NULL);
    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->cond_nonempty);
    pthread_cond_destroy(&pool->cond_nonfull);
    pthread_cond_destroy(&pool->cond_alldone);
    free(pool->threads);
    free(pool->queue);
    free(pool);
}

int threadpool_submit(ThreadPool *pool, WorkItem *item) {
    pthread_mutex_lock(&pool->mutex);
    while (pool->count >= pool->capacity)
        pthread_cond_wait(&pool->cond_nonfull, &pool->mutex);
    pool->queue[pool->tail] = *item;
    pool->tail = (pool->tail + 1) % pool->capacity;
    pool->count++;
    pool->in_flight++;
    pthread_cond_signal(&pool->cond_nonempty);
    pthread_mutex_unlock(&pool->mutex);
    return 0;
}

int threadpool_pending(ThreadPool *pool) {
    pthread_mutex_lock(&pool->mutex);
    int c = pool->count;
    pthread_mutex_unlock(&pool->mutex);
    return c;
}

void threadpool_wait(ThreadPool *pool) {
    pthread_mutex_lock(&pool->mutex);
    while (pool->in_flight > 0)
        pthread_cond_wait(&pool->cond_alldone, &pool->mutex);
    pthread_mutex_unlock(&pool->mutex);
}

struct OrderedQueue {
    WorkItem *items;
    int capacity;
    int count;
    volatile int finished;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
};

OrderedQueue *ordered_queue_create(int max_items) {
    OrderedQueue *oq = (OrderedQueue *)calloc(1, sizeof(OrderedQueue));
    if (!oq) return NULL;
    oq->capacity = max_items;
    oq->count = 0;
    oq->finished = 0;
    oq->items = (WorkItem *)calloc((size_t)max_items, sizeof(WorkItem));
    if (!oq->items) { free(oq); return NULL; }
    pthread_mutex_init(&oq->mutex, NULL);
    pthread_cond_init(&oq->cond, NULL);
    return oq;
}

void ordered_queue_destroy(OrderedQueue *oq) {
    if (!oq) return;
    free(oq->items);
    pthread_mutex_destroy(&oq->mutex);
    pthread_cond_destroy(&oq->cond);
    free(oq);
}

int ordered_queue_push(OrderedQueue *oq, WorkItem *item) {
    pthread_mutex_lock(&oq->mutex);
    int insert_pos = 0;
    while (insert_pos < oq->count && oq->items[insert_pos].seq < item->seq)
        insert_pos++;
    if (insert_pos < oq->count)
        memmove(&oq->items[insert_pos + 1], &oq->items[insert_pos],
                (size_t)(oq->count - insert_pos) * sizeof(WorkItem));
    oq->items[insert_pos] = *item;
    oq->count++;
    pthread_cond_signal(&oq->cond);
    pthread_mutex_unlock(&oq->mutex);
    return 0;
}

WorkItem *ordered_queue_pop(OrderedQueue *oq, int64_t next_seq, bool *done) {
    pthread_mutex_lock(&oq->mutex);
    while (oq->count == 0 && !oq->finished)
        pthread_cond_wait(&oq->cond, &oq->mutex);
    if (oq->count == 0) { *done = true; pthread_mutex_unlock(&oq->mutex); return NULL; }
    if (oq->items[0].seq != next_seq) { *done = false; pthread_mutex_unlock(&oq->mutex); return NULL; }
    WorkItem *result = (WorkItem *)malloc(sizeof(WorkItem));
    if (result) *result = oq->items[0];
    oq->count--;
    memmove(&oq->items[0], &oq->items[1], (size_t)oq->count * sizeof(WorkItem));
    *done = false;
    pthread_mutex_unlock(&oq->mutex);
    return result;
}

void ordered_queue_finish(OrderedQueue *oq) {
    pthread_mutex_lock(&oq->mutex);
    oq->finished = 1;
    pthread_cond_broadcast(&oq->cond);
    pthread_mutex_unlock(&oq->mutex);
}

struct SPSCQueue {
    void **buffer;
    int capacity;
    int mask;
    volatile int head;
    volatile int tail;
};

SPSCQueue *spsc_queue_create(int capacity) {
    int cap = 1;
    while (cap < capacity) cap <<= 1;
    SPSCQueue *q = (SPSCQueue *)calloc(1, sizeof(SPSCQueue));
    if (!q) return NULL;
    q->buffer = (void **)calloc((size_t)cap, sizeof(void *));
    if (!q->buffer) { free(q); return NULL; }
    q->capacity = cap;
    q->mask = cap - 1;
    q->head = 0;
    q->tail = 0;
    return q;
}

void spsc_queue_destroy(SPSCQueue *q) {
    if (!q) return;
    free(q->buffer);
    free(q);
}

int spsc_queue_push(SPSCQueue *q, void *ptr) {
    int h = q->head;
    int next_h = (h + 1) & q->mask;
    if (next_h == q->tail) return -1;
    q->buffer[h] = ptr;
    MEMORY_BARRIER();
    q->head = next_h;
    return 0;
}

void *spsc_queue_pop(SPSCQueue *q) {
    int t = q->tail;
    if (t == q->head) return NULL;
    void *ptr = q->buffer[t];
    q->tail = (t + 1) & q->mask;
    return ptr;
}

int spsc_queue_count(SPSCQueue *q) {
    int h = q->head;
    int t = q->tail;
    if (h >= t) return h - t;
    return q->capacity - (t - h);
}

#ifndef VIDCRYPT_THREADPOOL_H
#define VIDCRYPT_THREADPOOL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct WorkItem {
    void   *data;
    void   *user_data;
    int64_t seq;
    int     data_size;
} WorkItem;

typedef void (*work_func_t)(WorkItem *item);

typedef struct ThreadPool ThreadPool;

ThreadPool *threadpool_create(int num_threads, work_func_t work_func);
void threadpool_destroy(ThreadPool *pool);
int threadpool_submit(ThreadPool *pool, WorkItem *item);
int threadpool_pending(ThreadPool *pool);
void threadpool_wait(ThreadPool *pool);

typedef struct OrderedQueue OrderedQueue;
OrderedQueue *ordered_queue_create(int max_items);
void ordered_queue_destroy(OrderedQueue *oq);
int ordered_queue_push(OrderedQueue *oq, WorkItem *item);
WorkItem *ordered_queue_pop(OrderedQueue *oq, int64_t next_seq, bool *done);
void ordered_queue_finish(OrderedQueue *oq);

typedef struct SPSCQueue SPSCQueue;
SPSCQueue *spsc_queue_create(int capacity);
void spsc_queue_destroy(SPSCQueue *q);
int spsc_queue_push(SPSCQueue *q, void *ptr);
void *spsc_queue_pop(SPSCQueue *q);
int spsc_queue_count(SPSCQueue *q);

#ifdef __cplusplus
}
#endif

#endif /* VIDCRYPT_THREADPOOL_H */

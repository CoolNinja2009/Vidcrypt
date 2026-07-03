package com.vidcrypt.concurrent;

import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Concurrency utilities matching the C threadpool.h/c interfaces.
 * Uses Java 21 virtual threads for the thread pool and ConcurrentLinkedQueue for SPSC.
 */
public final class VidcryptExecutors {
    private VidcryptExecutors() {}

    // ── Thread Pool ──

    public interface WorkFunction {
        void execute(Object data, Object userData, long seq, int dataSize);
    }

    public static class WorkItem {
        public Object data;
        public Object userData;
        public long seq;
        public int dataSize;

        public WorkItem() {}
        public WorkItem(Object data, Object userData, long seq, int dataSize) {
            this.data = data;
            this.userData = userData;
            this.seq = seq;
            this.dataSize = dataSize;
        }
    }

    /**
     * Concurrent work-stealing thread pool using virtual threads.
     * Matches the interface of threadpool.h.
     */
    public static class WorkPool implements AutoCloseable {
        private final ExecutorService executor;
        private final WorkFunction func;
        private final AtomicInteger pending = new AtomicInteger(0);
        private volatile CountDownLatch currentLatch;

        public WorkPool(int numThreads, WorkFunction func) {
            this.func = func;
            // Use virtual threads — lightweight, scales to millions
            this.executor = Executors.newVirtualThreadPerTaskExecutor();
            this.currentLatch = new CountDownLatch(0);
        }

        public int submit(WorkItem item) {
            pending.incrementAndGet();
            executor.submit(() -> {
                try {
                    func.execute(item.data, item.userData, item.seq, item.dataSize);
                } finally {
                    currentLatch.countDown();
                    pending.decrementAndGet();
                }
            });
            return 0;
        }

        public int submitBatch(WorkItem[] items, int count) {
            currentLatch = new CountDownLatch(count);
            for (int i = 0; i < count; i++) {
                submit(items[i]);
            }
            return 0;
        }

        public int pendingCount() {
            return pending.get();
        }

        public void waitAll() {
            try {
                currentLatch.await();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        @Override
        public void close() {
            executor.shutdown();
            try {
                executor.awaitTermination(5, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }

    // ── Ordered Queue (producer/consumer with sequence-number ordering) ──

    /**
     * Ordered queue that delivers items in sequence-number order.
     * Matches ordered_queue.h.
     */
    public static class OrderedQueue<T> {
        private final ConcurrentSkipListMap<Long, T> map = new ConcurrentSkipListMap<>();
        private volatile long nextSeq = 0;
        private volatile boolean finished = false;
        private final ReentrantLock lock = new ReentrantLock();
        private final Condition notEmpty = lock.newCondition();

        public OrderedQueue() {}

        public void push(long seq, T item) {
            map.put(seq, item);
            lock.lock();
            try {
                if (seq == nextSeq) notEmpty.signalAll();
            } finally {
                lock.unlock();
            }
        }

        public T pop() throws InterruptedException {
            lock.lock();
            try {
                while (!finished) {
                    T val = map.remove(nextSeq);
                    if (val != null) {
                        nextSeq++;
                        return val;
                    }
                    if (finished && map.isEmpty()) return null;
                    notEmpty.await();
                }
                T val = map.remove(nextSeq);
                if (val != null) nextSeq++;
                return val;
            } finally {
                lock.unlock();
            }
        }

        public void finish() {
            lock.lock();
            try {
                finished = true;
                notEmpty.signalAll();
            } finally {
                lock.unlock();
            }
        }

        public int size() { return map.size(); }
    }

    // ── SPSC (Single Producer Single Consumer) Queue ──

    /**
     * Lock-free SPSC queue using ConcurrentLinkedQueue.
     * Matches SPSCQueue from threadpool.h.
     */
    public static class SpscQueue<T> {
        private final ConcurrentLinkedQueue<T> queue = new ConcurrentLinkedQueue<>();
        private final int capacity;

        public SpscQueue(int capacity) {
            this.capacity = capacity;
        }

        public boolean push(T item) {
            if (queue.size() >= capacity) return false;
            queue.offer(item);
            return true;
        }

        public T pop() {
            return queue.poll();
        }

        public int count() {
            return queue.size();
        }

        public void clear() { queue.clear(); }
    }
}

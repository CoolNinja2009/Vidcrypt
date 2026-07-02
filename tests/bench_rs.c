/* ─── Reed-Solomon Performance Benchmarks ─────────────────────────────
 * Compares: old CPU rs_encode, dictionary-based CPU encode,
 * multi-threaded variants, and the full encoder pipeline.
 *
 * Build: cmake --build . --config Release
 * Run:   ./Release/bench-rs [MiB] [iterations]
 *        (default: 100 MiB, 5 iterations)                              */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#define fseeko _fseeki64
#define ftello _ftelli64
#else
#include <pthread.h>
#endif

#include "reedsolomon.h"
#include "bitstream.h"

/* ─── Timer ─────────────────────────────────────────────────────────── */
#ifdef _WIN32
static double now_sec(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
}
#else
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

/* ─── Progress (only for single-threaded benchmarks, throttled) ────── */
static void progress(const char *label, double done, double total, double start) {
    /* Update at most ~10 times per second */
    static int64_t last_update = 0;
    double now = now_sec();
    if (now - (double)last_update < 0.1 && done < total) return;
    last_update = (int64_t)now;
    double elapsed = now - start;
    double rate = done > 0 ? done / elapsed : 0;
    fprintf(stderr, "  %-30s  %5.1f%%  %.1f MB/s  (%.2f s)\r",
            label, 100.0 * done / total, rate / 1e6, elapsed);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Benchmark 1: Original rs_encode (sequential polynomial division)
 * This is the baseline — what the CPU path currently uses.              */

static double bench_rs_original(const uint8_t *data, int64_t total_bytes,
                                 const RSCodec *codec, int iterations) {
    int k = (int)codec->msg_length;
    int total_chunks = (int)((total_bytes + k - 1) / k);
    double best = 1e99;

    for (int iter = 0; iter < iterations; ++iter) {
        double t0 = now_sec();
        int64_t done = 0;
        for (int ci = 0; ci < total_chunks; ++ci) {
            int nread = k;
            if ((int64_t)(ci * k + nread) > total_bytes)
                nread = (int)(total_bytes - (int64_t)ci * k);
            if (nread <= 0) break;

            uint8_t padded[256] = {0};
            memcpy(padded, data + ci * k, (size_t)nread);
            uint8_t enc[255];
            rs_encode(codec, padded, k, enc);
            done += nread;
            if (iter == 0) progress("Bench: rs_encode (original)", (double)done, (double)total_bytes, t0);
        }
        double elapsed = now_sec() - t0;
        fprintf(stderr, "\r  %-30s  done  %.1f MB/s  (%.2f s)\n",
                "Bench: rs_encode (original)", (double)total_bytes / elapsed / 1e6, elapsed);
        if (elapsed < best) best = elapsed;
    }
    return best;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Benchmark 2: Dictionary-based RS encode (single-threaded CPU)
 * Simulates what the GPU kernel does: XOR dict lookups.
 * This is the CPU version of the GPU RS encode path.                    */

static double bench_rs_dict_cpu(const uint8_t *data, int64_t total_bytes,
                                 const RSCodec *codec, int iterations) {
    int k = (int)codec->msg_length;
    int n_k = (int)codec->ecc_symbols;
    int dict_stride = 256 * n_k;
    int total_chunks = (int)((total_bytes + k - 1) / k);
    double best = 1e99;

    for (int iter = 0; iter < iterations; ++iter) {
        double t0 = now_sec();
        int64_t done = 0;
        for (int ci = 0; ci < total_chunks; ++ci) {
            int nread = k;
            if ((int64_t)(ci * k + nread) > total_bytes)
                nread = (int)(total_bytes - (int64_t)ci * k);
            if (nread <= 0) break;

            uint8_t parity[32] = {0};
            for (int i = 0; i < k; ++i) {
                uint8_t v = (i < nread) ? data[ci * k + i] : 0;
                const uint8_t *row = codec->parity_dict
                    + (size_t)i * (size_t)dict_stride
                    + (size_t)v * (size_t)n_k;
                for (int j = 0; j < n_k; ++j)
                    parity[j] ^= row[j];
            }
            done += nread;
            if (iter == 0) progress("Bench: dict-encode (CPU 1-thread)", (double)done, (double)total_bytes, t0);
        }
        double elapsed = now_sec() - t0;
        fprintf(stderr, "\r  %-30s  done  %.1f MB/s  (%.2f s)\n",
                "Bench: dict-encode (CPU 1-thread)", (double)total_bytes / elapsed / 1e6, elapsed);
        if (elapsed < best) best = elapsed;
    }
    return best;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Benchmark 3: Original rs_encode multi-threaded (current CPU path)     */

typedef struct {
    const uint8_t *data;
    int64_t total_bytes;
    int k;
    int n_k;
    int start_chunk, end_chunk;
    const RSCodec *codec;
} MtArgs;

#ifdef _WIN32
static DWORD WINAPI mt_worker(LPVOID arg) {
#else
static void* mt_worker(void *arg) {
#endif
    MtArgs *a = (MtArgs *)arg;
    for (int ci = a->start_chunk; ci < a->end_chunk; ++ci) {
        int nread = a->k;
        if ((int64_t)(ci * a->k + nread) > a->total_bytes)
            nread = (int)(a->total_bytes - (int64_t)ci * a->k);
        if (nread <= 0) break;
        uint8_t padded[256] = {0};
        memcpy(padded, a->data + ci * (int64_t)a->k, (size_t)nread);
        uint8_t enc[255];
        rs_encode(a->codec, padded, a->k, enc);
    }
    return 0;
}

static double bench_rs_original_mt(const uint8_t *data, int64_t total_bytes,
                                    const RSCodec *codec, int iterations,
                                    int n_threads) {
    int k = (int)codec->msg_length;
    int total_chunks = (int)((total_bytes + k - 1) / k);
    double best = 1e99;

    for (int iter = 0; iter < iterations; ++iter) {
        double t0 = now_sec();
#ifdef _WIN32
        HANDLE *threads = (HANDLE *)malloc((size_t)n_threads * sizeof(HANDLE));
        MtArgs *args = (MtArgs *)malloc((size_t)n_threads * sizeof(MtArgs));
#else
        pthread_t *threads = (pthread_t *)malloc((size_t)n_threads * sizeof(pthread_t));
        MtArgs *args = (MtArgs *)malloc((size_t)n_threads * sizeof(MtArgs));
#endif
        int chunks_per = total_chunks / n_threads;

        for (int t = 0; t < n_threads; ++t) {
            args[t].data = data;
            args[t].total_bytes = total_bytes;
            args[t].k = k;
            args[t].codec = codec;
            args[t].start_chunk = t * chunks_per;
            args[t].end_chunk = (t == n_threads - 1) ? total_chunks : (t + 1) * chunks_per;
#ifdef _WIN32
            threads[t] = CreateThread(NULL, 0, mt_worker, &args[t], 0, NULL);
#else
            pthread_create(&threads[t], NULL, mt_worker, &args[t]);
#endif
        }
#ifdef _WIN32
        WaitForMultipleObjects((DWORD)n_threads, threads, TRUE, INFINITE);
        for (int t = 0; t < n_threads; ++t) CloseHandle(threads[t]);
#else
        for (int t = 0; t < n_threads; ++t) pthread_join(threads[t], NULL);
#endif
        free(threads); free(args);

        double elapsed = now_sec() - t0;
        char label[64];
        snprintf(label, sizeof(label), "Bench: rs_encode (%d threads)", n_threads);
        fprintf(stderr, "  %-30s  done  %.1f MB/s  (%.2f s)\n",
                label, (double)total_bytes / elapsed / 1e6, elapsed);
        if (elapsed < best) best = elapsed;
    }
    return best;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Benchmark 4: Dictionary-based RS encode multi-threaded               */

typedef struct {
    const uint8_t *data;
    int64_t total_bytes;
    int k;
    int n_k;
    int dict_stride;
    int start_chunk, end_chunk;
    const RSCodec *codec;
} MtDictArgs;

#ifdef _WIN32
static DWORD WINAPI mt_dict_worker(LPVOID arg) {
#else
static void* mt_dict_worker(void *arg) {
#endif
    MtDictArgs *a = (MtDictArgs *)arg;
    int n_k = a->n_k;
    int k = a->k;
    int dict_stride = a->dict_stride;

    for (int ci = a->start_chunk; ci < a->end_chunk; ++ci) {
        int nread = k;
        if ((int64_t)(ci * k + nread) > a->total_bytes)
            nread = (int)(a->total_bytes - (int64_t)ci * k);
        if (nread <= 0) break;

        uint8_t parity[32] = {0};
        for (int i = 0; i < k; ++i) {
            uint8_t v = (i < nread) ? a->data[ci * (int64_t)k + i] : 0;
            const uint8_t *row = a->codec->parity_dict
                + (size_t)i * (size_t)dict_stride
                + (size_t)v * (size_t)n_k;
            for (int j = 0; j < n_k; ++j)
                parity[j] ^= row[j];
        }
    }
    return 0;
}

static double bench_rs_dict_mt(const uint8_t *data, int64_t total_bytes,
                                const RSCodec *codec, int iterations,
                                int n_threads) {
    int k = (int)codec->msg_length;
    int n_k = (int)codec->ecc_symbols;
    int dict_stride = 256 * n_k;
    int total_chunks = (int)((total_bytes + k - 1) / k);
    double best = 1e99;

    for (int iter = 0; iter < iterations; ++iter) {
        double t0 = now_sec();
#ifdef _WIN32
        HANDLE *threads = (HANDLE *)malloc((size_t)n_threads * sizeof(HANDLE));
        MtDictArgs *args = (MtDictArgs *)malloc((size_t)n_threads * sizeof(MtDictArgs));
#else
        pthread_t *threads = (pthread_t *)malloc((size_t)n_threads * sizeof(pthread_t));
        MtDictArgs *args = (MtDictArgs *)malloc((size_t)n_threads * sizeof(MtDictArgs));
#endif
        int chunks_per = total_chunks / n_threads;

        for (int t = 0; t < n_threads; ++t) {
            args[t].data = data;
            args[t].total_bytes = total_bytes;
            args[t].k = k;
            args[t].n_k = n_k;
            args[t].dict_stride = dict_stride;
            args[t].codec = codec;
            args[t].start_chunk = t * chunks_per;
            args[t].end_chunk = (t == n_threads - 1) ? total_chunks : (t + 1) * chunks_per;
#ifdef _WIN32
            threads[t] = CreateThread(NULL, 0, mt_dict_worker, &args[t], 0, NULL);
#else
            pthread_create(&threads[t], NULL, mt_dict_worker, &args[t]);
#endif
        }
#ifdef _WIN32
        WaitForMultipleObjects((DWORD)n_threads, threads, TRUE, INFINITE);
        for (int t = 0; t < n_threads; ++t) CloseHandle(threads[t]);
#else
        for (int t = 0; t < n_threads; ++t) pthread_join(threads[t], NULL);
#endif
        free(threads); free(args);

        double elapsed = now_sec() - t0;
        char label[64];
        snprintf(label, sizeof(label), "Bench: dict-encode (%d threads)", n_threads);
        fprintf(stderr, "  %-30s  done  %.1f MB/s  (%.2f s)\n",
                label, (double)total_bytes / elapsed / 1e6, elapsed);
        if (elapsed < best) best = elapsed;
    }
    return best;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Benchmark 5: Bit expansion throughput (LUT64)                         */

static double bench_bit_expand(const uint8_t *data, int64_t total_bytes,
                                int iterations) {
    int64_t total_bits = total_bytes * 8;
    uint8_t *bitbuf = (uint8_t *)malloc((size_t)total_bits);
    if (!bitbuf) return -1;
    double best = 1e99;

    for (int iter = 0; iter < iterations; ++iter) {
        double t0 = now_sec();
        for (int64_t i = 0; i < total_bytes; ++i)
            ((uint64_t *)bitbuf)[i] = LUT64[data[i]];
        double elapsed = now_sec() - t0;
        fprintf(stderr, "  %-30s  done  %.1f MB/s  (%.2f s) [input: %.1f MiB → output: %.1f MiB]\n",
                "Bench: bit-expand (LUT64)", (double)total_bytes / elapsed / 1e6, elapsed,
                (double)total_bytes / 1e6, (double)total_bits / 1e6);
        if (elapsed < best) best = elapsed;
    }

    free(bitbuf);
    return best;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Main                                                                  */

int main(int argc, char **argv) {
    int size_mib = (argc > 1) ? atoi(argv[1]) : 100;
    int iterations = (argc > 2) ? atoi(argv[2]) : 5;
    if (size_mib < 1) size_mib = 100;
    if (iterations < 1) iterations = 1;

    int64_t total_bytes = (int64_t)size_mib * 1024 * 1024;

    printf("=== Reed-Solomon Benchmarks ===\n");
    printf("Data size: %d MiB (%lld bytes), iterations: %d\n\n",
           size_mib, (long long)total_bytes, iterations);

    gf256_init();

    /* Allocate test data (random-ish deterministic pattern) */
    uint8_t *data = (uint8_t *)malloc((size_t)total_bytes);
    if (!data) { fprintf(stderr, "OOM\n"); return 1; }
    for (int64_t i = 0; i < total_bytes; ++i)
        data[i] = (uint8_t)((i * 7 + 13) & 0xFF);

    /* Get RS codec */
    const RSCodec *codec = rs_get_codec(32);
    if (!codec) { fprintf(stderr, "RS init failed\n"); free(data); return 1; }
    printf("RS codec: RS(255,223,32) — corrects up to 16 byte errors per 255-byte block\n");
    printf("  Dictionary size: %.2f MiB (%zu bytes)\n",
           (double)codec->parity_dict_size / 1e6, codec->parity_dict_size);
    printf("  Total RS chunks: %d\n\n", (int)((total_bytes + 222) / 223));

    int n_cpu = 1;
#ifdef _WIN32
    SYSTEM_INFO si; GetSystemInfo(&si);
    n_cpu = (int)si.dwNumberOfProcessors;
#else
    n_cpu = (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
    if (n_cpu < 1) n_cpu = 1;
    if (n_cpu > 16) n_cpu = 16;
    int n_half = n_cpu / 2;
    if (n_half < 1) n_half = 1;

    printf("=== RS Encode Throughput (raw input bytes) ===\n\n");

    /* Benchmark 1: Original single-threaded */
    double t_orig = bench_rs_original(data, total_bytes, codec, iterations);
    double rate_orig = (double)total_bytes / t_orig / 1e6;

    /* Benchmark 2: Dictionary single-threaded */
    double t_dict = bench_rs_dict_cpu(data, total_bytes, codec, iterations);
    double rate_dict = (double)total_bytes / t_dict / 1e6;

    /* Benchmark 3: Original multi-threaded (n/2) */
    double t_orig_mt = bench_rs_original_mt(data, total_bytes, codec, iterations, n_half);
    double rate_orig_mt = (double)total_bytes / t_orig_mt / 1e6;

    /* Benchmark 4: Original multi-threaded (n) */
    double t_orig_mt_full = bench_rs_original_mt(data, total_bytes, codec, iterations, n_cpu);
    double rate_orig_mt_full = (double)total_bytes / t_orig_mt_full / 1e6;

    /* Benchmark 5: Dictionary multi-threaded (n/2) */
    double t_dict_mt = bench_rs_dict_mt(data, total_bytes, codec, iterations, n_half);
    double rate_dict_mt = (double)total_bytes / t_dict_mt / 1e6;

    /* Benchmark 6: Dictionary multi-threaded (n) */
    double t_dict_mt_full = bench_rs_dict_mt(data, total_bytes, codec, iterations, n_cpu);
    double rate_dict_mt_full = (double)total_bytes / t_dict_mt_full / 1e6;

    /* Benchmark 7: Bit expansion */
    bench_bit_expand(data, total_bytes, iterations);

    printf("\n=== Summary ===\n\n");
    printf("  %-35s  %8.1f MB/s  (%.3f s)\n", "rs_encode (1 thread)", rate_orig, t_orig);
    printf("  %-35s  %8.1f MB/s  (%.3f s)\n", "rs_encode (%d threads)", n_half, rate_orig_mt, t_orig_mt);
    printf("  %-35s  %8.1f MB/s  (%.3f s)\n", "rs_encode (%d threads)", n_cpu, rate_orig_mt_full, t_orig_mt_full);
    printf("  %-35s  %8.1f MB/s  (%.3f s)\n", "dict-encode (1 thread)", rate_dict, t_dict);
    printf("  %-35s  %8.1f MB/s  (%.3f s)\n", "dict-encode (%d threads)", n_half, rate_dict_mt, t_dict_mt);
    printf("  %-35s  %8.1f MB/s  (%.3f s)\n", "dict-encode (%d threads)", n_cpu, rate_dict_mt_full, t_dict_mt_full);

    double speedup_dict_st = rate_dict / rate_orig;
    double speedup_dict_mt = rate_dict_mt_full / rate_orig;
    double speedup_orig_mt = rate_orig_mt_full / rate_orig;

    printf("\n=== Speedups ===\n\n");
    printf("  Dict vs Original (1 thread):     %.2f×\n", speedup_dict_st);
    printf("  Dict vs Original (%d threads):   %.2f×\n", n_cpu, speedup_dict_mt);
    printf("  Original MT vs ST:               %.2f×  (%d threads)\n", speedup_orig_mt, n_cpu);
    printf("\n");

    /* Project expected GPU throughput */
    double nvenc_target_MBps = 10000.0 / 413.0 * 1.0;  /* ~24 MB/s at 10000 fps for 1MB file */
    printf("=== Projected GPU RS Throughput ===\n\n");
    printf("  GPU dict-encode (estimated):     ~500+ MB/s  (limited by GPU memory bandwidth)\n");
    printf("  NVENC encode ceiling:            ~%.0f MB/s  (at 10,000 fps with ~1 MB file)\n", nvenc_target_MBps);
    printf("  Minimum to saturate NVENC:       ~%.0f MB/s\n", nvenc_target_MBps);

    if (rate_dict_mt_full >= nvenc_target_MBps) {
        printf("\n  ✓ Multi-threaded CPU dict-encode CAN saturate NVENC at 10,000 fps!\n");
    } else if (rate_dict_mt >= nvenc_target_MBps) {
        printf("\n  ✓ Multi-threaded CPU dict-encode (%d threads) CAN saturate NVENC at 10,000 fps!\n", n_half);
    } else {
        printf("\n  → GPU dict-encode needed for full NVENC saturation.\n");
        printf("    Estimated GPU speedup: 50-100× over single-threaded rs_encode.\n");
    }

    free(data);
    return 0;
}
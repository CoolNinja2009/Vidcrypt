/* ═══════════════════════════════════════════════════════════════════════
 * bench_reedsolomon.c — RS encode/decode microbenchmark
 *
 * Measures throughput (MB/s) for encode and decode across all ECC levels.
 * Compares scalar vs SIMD paths when compiled with SIMD support.
 * ═══════════════════════════════════════════════════════════════════════ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "reedsolomon.h"
#include "reedsolomon_simd.h"

/* ── Timer ─────────────────────────────────────────────────────────── */
#ifdef _WIN32
#include <windows.h>
static double get_time_sec(void) {
    static LARGE_INTEGER freq = {0};
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart / (double)freq.QuadPart;
}
#else
static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

/* ── Helpers ───────────────────────────────────────────────────────── */
static void fill_random(uint8_t *buf, int len, unsigned seed) {
    srand(seed);
    for (int i = 0; i < len; i++)
        buf[i] = (uint8_t)(rand() & 0xFF);
}

#define ITERATIONS 200000
#define WARMUP       5000

typedef struct {
    double encode_mbps;
    double decode_clean_mbps;
    double decode_1err_mbps;
    double decode_bound_mbps;
} BenchResult;

static BenchResult bench_ecc(int ecc, const char *label) {
    const RSCodec *codec = rs_get_codec(ecc);
    int k = (int)codec->msg_length;
    int n = (int)codec->block_length;
    int max_errs = ecc / 2;

    uint8_t *msg     = (uint8_t *)malloc((size_t)k);
    uint8_t *encoded = (uint8_t *)malloc((size_t)n);
    uint8_t *decoded = (uint8_t *)malloc((size_t)k);

    BenchResult r = {0};

    /* ── Encode ────────────────────────────────────────────────── */
    fill_random(msg, k, 42);
    /* warmup */
    for (int i = 0; i < WARMUP; i++)
        rs_encode(codec, msg, k, encoded);

    double t0 = get_time_sec();
    for (int i = 0; i < ITERATIONS; i++)
        rs_encode(codec, msg, k, encoded);
    double t1 = get_time_sec();
    double encode_sec = t1 - t0;
    r.encode_mbps = (double)((int64_t)k * ITERATIONS) / (encode_sec * 1e6);

    /* ── Decode (clean) ────────────────────────────────────────── */
    {
        RSDecodeResult result;
        /* warmup */
        for (int i = 0; i < WARMUP; i++)
            rs_decode(codec, encoded, &result);

        t0 = get_time_sec();
        for (int i = 0; i < ITERATIONS; i++)
            rs_decode(codec, encoded, &result);
        t1 = get_time_sec();
        double sec = t1 - t0;
        r.decode_clean_mbps = (double)((int64_t)k * ITERATIONS) / (sec * 1e6);
    }

    /* ── Decode (1 error) ──────────────────────────────────────── */
    {
        RSDecodeResult result;
        uint8_t corrupted[255];
        memcpy(corrupted, encoded, 255);
        corrupted[0] ^= 0xFF; /* 1 error at position 0 */

        for (int i = 0; i < WARMUP; i++)
            rs_decode(codec, corrupted, &result);

        t0 = get_time_sec();
        for (int i = 0; i < ITERATIONS; i++)
            rs_decode(codec, corrupted, &result);
        t1 = get_time_sec();
        double sec = t1 - t0;
        r.decode_1err_mbps = (double)((int64_t)k * ITERATIONS) / (sec * 1e6);
    }

    /* ── Decode (bound errors) ─────────────────────────────────── */
    if (max_errs > 0) {
        RSDecodeResult result;
        uint8_t corrupted[255];
        memcpy(corrupted, encoded, 255);
        /* Inject max_errs errors at spread positions */
        for (int e = 0; e < max_errs; e++)
            corrupted[e * (255 / max_errs)] ^= (uint8_t)(0xFF - e);

        for (int i = 0; i < WARMUP; i++)
            rs_decode(codec, corrupted, &result);

        t0 = get_time_sec();
        for (int i = 0; i < ITERATIONS; i++)
            rs_decode(codec, corrupted, &result);
        t1 = get_time_sec();
        double sec = t1 - t0;
        r.decode_bound_mbps = (double)((int64_t)k * ITERATIONS) / (sec * 1e6);
    }

    printf("\n  %-6s   %7.1f      %7.1f         %7.1f         %7.1f\n",
           label, r.encode_mbps, r.decode_clean_mbps,
           r.decode_1err_mbps, r.decode_bound_mbps);

    free(msg);
    free(encoded);
    free(decoded);
    return r;
}

int main(void) {
    gf256_init();

    printf("=== Reed-Solomon Microbenchmark ===\n");
    printf("Path: %s\n", rs_path_name());
    printf("Iterations per test: %d\n\n", ITERATIONS);

    printf("  Level   Enc MB/s   Dec-clean MB/s  Dec-1err MB/s  Dec-bound MB/s\n");
    printf("  ------  ---------  --------------  --------------  ---------------\n");

    bench_ecc(32, "RS32");
    bench_ecc(16, "RS16");
    bench_ecc(8,  "RS8");

    /* No-RS path (encode_block / decode_block with NULL codec) */
    {
        uint8_t *msg = (uint8_t *)malloc(255);
        uint8_t *encoded = (uint8_t *)malloc(255);
        uint8_t *decoded = (uint8_t *)malloc(255);
        int status;
        fill_random(msg, 255, 99);

        /* encode warmup */
        for (int i = 0; i < WARMUP; i++)
            rs_encode_block(NULL, msg, 255, encoded);

        double t0 = get_time_sec();
        for (int i = 0; i < ITERATIONS; i++)
            rs_encode_block(NULL, msg, 255, encoded);
        double t1 = get_time_sec();
        double enc_mbps = (double)(255LL * ITERATIONS) / ((t1 - t0) * 1e6);

        /* decode warmup */
        for (int i = 0; i < WARMUP; i++)
            rs_decode_block(NULL, encoded, decoded, &status);

        t0 = get_time_sec();
        for (int i = 0; i < ITERATIONS; i++)
            rs_decode_block(NULL, encoded, decoded, &status);
        t1 = get_time_sec();
        double dec_mbps = (double)(255LL * ITERATIONS) / ((t1 - t0) * 1e6);

        printf("  %-6s   %7.1f      %7.1f         %7s         %7s\n",
               "NoRS", enc_mbps, dec_mbps, "-", "-");

        free(msg);
        free(encoded);
        free(decoded);
    }

    printf("\nDone.\n");
    return 0;
}

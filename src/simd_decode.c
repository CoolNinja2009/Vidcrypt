/* ═══════════════════════════════════════════════════════════════════════
 * simd_decode.c — Optimized tile decoding for Vidcrypt
 *
 * Optimization summary:
 * ───────────────────────────────────────────────────────────────────────
 * 1. Branchless counting:   (pixel >> 7) replaces conditional (pixel>=128)
 * 2. Early-majority exit:   Stop when count > half, or can't reach half
 *                            (only in general fallback — not in specialized
 *                            4-sample paths where unrolled is faster)
 * 3. Specialized paths:     Manually unrolled for block_size = 8 and 16
 * 4. Single-dispatch:       No intermediate function call layers on hot path
 * 5. SSE2 calibration:      _mm_sad_epu8 sums 16 px/op (contiguous region)
 * 6. AVX2 calibration:      _mm256_sad_epu8 sums 32 px/op (2x SSE2)
 * 7. Runtime dispatch:       Calibration reader selects best path at init
 *
 * Key insight for tile decoding:
 *   subsample = block_size / 2  ⇒  total_samples ALWAYS = 4
 *   Need count > 2 (3 or 4 white out of 4).  SIMD not beneficial for
 *   4 scattered pixels — overhead > gain.  Branchless unrolled is optimal.
 *
 * Key insight for calibration:
 *   tile_read_calibration() processes contiguous pixel rectangles.
 *   Each row is ~22 bytes — SSE2 sums 16 at once, AVX2 sums 32.
 *   This is the only path where SIMD pays off.
 * ═══════════════════════════════════════════════════════════════════════ */

#include "simd_decode.h"
#include "calibration.h"
#include <string.h>
#include <stdlib.h>

/* ──────────────────────────────────────────────────────────────────────
 * Portable compiler hints
 * ────────────────────────────────────────────────────────────────────── */
#if defined(_MSC_VER) && !defined(__clang__)
    #define ALWAYS_INLINE   __forceinline
    #define RESTRICT        __restrict
#elif defined(__GNUC__) || defined(__clang__)
    #define ALWAYS_INLINE   __attribute__((always_inline)) inline
    #define RESTRICT        __restrict__
#else
    #define ALWAYS_INLINE   inline
    #define RESTRICT
#endif

/* ──────────────────────────────────────────────────────────────────────
 * SIMD header inclusion
 *
 * SSE2 is guaranteed on x86-64.  AVX2 requires both compiler target
 * support (#define __AVX2__) AND runtime CPU detection.
 * ────────────────────────────────────────────────────────────────────── */
#include <emmintrin.h>                  /* SSE2 — always available on x86-64 */

#if defined(__AVX2__)
    #include <immintrin.h>              /* AVX2 — conditional on compiler flags */
    #define HAS_AVX2 1
#else
    #define HAS_AVX2 0
#endif

/* ──────────────────────────────────────────────────────────────────────
 * CPU feature detection (only compiled when AVX2 is enabled)
 * ────────────────────────────────────────────────────────────────────── */
#if HAS_AVX2
#if defined(_MSC_VER)
#   include <intrin.h>
static int _cpu_has_avx2(void) {
    int info[4];
    __cpuid(info, 0);
    if (info[0] < 7) return 0;
    __cpuid(info, 1);
    if (!(info[2] & (1U << 27))) return 0;   /* OSXSAVE */
    if (!(info[2] & (1U << 28))) return 0;   /* AVX */
    /* Check OS has enabled XMM & YMM state */
    uint64_t xcr = _xgetbv(0);
    if ((xcr & 6) != 6) return 0;
    __cpuidex(info, 7, 0);
    return (info[1] & (1U << 5)) != 0;       /* AVX2 */
}
#elif defined(__GNUC__) || defined(__clang__)
#   include <cpuid.h>
static uint64_t _xgetbv_cpuid(uint32_t idx) {
    uint32_t eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(idx));
    return ((uint64_t)edx << 32) | eax;
}
static int _cpu_has_avx2(void) {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_max(0, NULL) < 7) return 0;
    __cpuid_count(1, 0, eax, ebx, ecx, edx);
    if (!(ecx & (1U << 27)) || !(ecx & (1U << 28))) return 0;
    /* Check OS XSAVE for YMM registers */
    if ((_xgetbv_cpuid(0) & 6) != 6) return 0;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ebx & (1U << 5)) != 0;           /* AVX2 */
}
#else
static int _cpu_has_avx2(void) { return 0; }
#endif /* platform dispatch */
#endif /* HAS_AVX2 */

/* ═══════════════════════════════════════════════════════════════════════
 * Specialized count functions — ALWAYS_INLINE, fully unrolled, branchless
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Standard geometry:  subsample = block_size/2  ⇒  always 4 samples.
 * Branchless:   (pixel >> 7)  gives 1 for pixel >= 128, 0 otherwise.
 * No early-exit:  with only 4 samples, the unrolled path is fastest.
 * ═══════════════════════════════════════════════════════════════════════ */

/* ── Specialized: block_size = 8, subsample = 4 ─────────────────────
 *   off = 2, step = 4  ⇒  positions: (2,2) (2,6) (6,2) (6,6)
 * ─────────────────────────────────────────────────────────────────────── */
static ALWAYS_INLINE int count_white_block8(const uint8_t *RESTRICT src, int stride) {
    return (src[2 * stride + 2] >> 7)
         + (src[2 * stride + 6] >> 7)
         + (src[6 * stride + 2] >> 7)
         + (src[6 * stride + 6] >> 7);
}

/* ── Specialized: block_size = 16, subsample = 8 ────────────────────
 *   off = 4, step = 8  ⇒  positions: (4,4) (4,12) (12,4) (12,12)
 * ─────────────────────────────────────────────────────────────────────── */
static ALWAYS_INLINE int count_white_block16(const uint8_t *RESTRICT src, int stride) {
    return (src[4 * stride + 4] >> 7)
         + (src[4 * stride + 12] >> 7)
         + (src[12 * stride + 4] >> 7)
         + (src[12 * stride + 12] >> 7);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Public API — tile_count_white  (general fallback with early-exit)
 *
 * Used for non-standard geometry where block_size != 8 or 16.
 * Early-exit branches are beneficial for larger tile sizes.
 * ═══════════════════════════════════════════════════════════════════════ */
int tile_count_white(const uint8_t *src, int stride,
                     int tile_width, int tile_height,
                     int subsample)
{
    int off       = subsample / 2;
    int ncols     = (tile_width  - 1 - off) / subsample + 1;
    int nrows     = (tile_height - 1 - off) / subsample + 1;
    int total     = ncols * nrows;
    int half      = total / 2;
    int count     = 0;
    int remaining = total;

    for (int y = off; y < tile_height; y += subsample) {
        const uint8_t *RESTRICT row = src + (size_t)y * stride;
        for (int x = off; x < tile_width; x += subsample) {
            /* Branchless:  bit 7 set → >= 128 → 1 */
            count     += (row[x] >> 7);
            remaining--;

            /* Early-majority exit:  already won */
            if (count > half)                     return count;
            /* Can't-win exit:   even all white can't reach majority */
            if (count + remaining <= half)        return count;
        }
    }
    return count;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Public API — tile_validate_sync
 *
 * Single-dispatch:  checks block_size once, enters specialized or general
 * path.  No intermediate function call layers.
 * ═══════════════════════════════════════════════════════════════════════ */
bool tile_validate_sync(const uint8_t *gray, int stride,
                        int sync_y, int sync_x_start,
                        int block_size, int grid_cols,
                        int subsample)
{
    const uint8_t *RESTRICT base = gray + (size_t)sync_y * stride;

    if (subsample > 0 && subsample * 2 == (int)block_size) {
        /* ── block_size = 8 (most common) ── */
        if (block_size == 8) {
            for (int col = 0; col < grid_cols; ++col) {
                size_t x = (size_t)sync_x_start + (size_t)col * 8;
                int c = count_white_block8(base + x, stride);
                if ((c > 2) != (col & 1)) return false;
            }
            return true;
        }
        /* ── block_size = 16 ── */
        if (block_size == 16) {
            for (int col = 0; col < grid_cols; ++col) {
                size_t x = (size_t)sync_x_start + (size_t)col * 16;
                int c = count_white_block16(base + x, stride);
                if ((c > 2) != (col & 1)) return false;
            }
            return true;
        }
    }

    /* ── General fallback ── */
    for (int col = 0; col < grid_cols; ++col) {
        size_t x = (size_t)sync_x_start + (size_t)col * block_size;
        int c = tile_count_white(base + x, stride,
                                 block_size, block_size, subsample);
        if ((c > 2) != (col & 1)) return false;
    }
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Public API — tile_decode_grid
 *
 * Single-dispatch:  checks block_size once, enters specialized or general
 * path for both sync validation AND payload decode.
 * ═══════════════════════════════════════════════════════════════════════ */
int tile_decode_grid(const uint8_t *gray, int stride,
                     int grid_top_y, int grid_left_x,
                     int block_size,
                     int grid_cols, int grid_rows, int sync_rows,
                     int subsample,
                     uint8_t *bits_out, int max_bits,
                     bool *sync_ok)
{
    *sync_ok = tile_validate_sync(gray, stride, grid_top_y, grid_left_x,
                                   block_size, grid_cols, subsample);

    int bit_idx = 0;

    if (subsample > 0 && subsample * 2 == (int)block_size) {
        /* ── block_size = 8 (most common) ── */
        if (block_size == 8) {
            for (int row = sync_rows; row < grid_rows && bit_idx < max_bits; ++row) {
                size_t y = (size_t)grid_top_y + (size_t)row * 8;
                const uint8_t *RESTRICT row_base = gray + y * stride;
                for (int col = 0; col < grid_cols && bit_idx < max_bits; ++col) {
                    size_t x = (size_t)grid_left_x + (size_t)col * 8;
                    bits_out[bit_idx] = (uint8_t)((count_white_block8(row_base + x, stride) > 2) ? 1 : 0);
                    bit_idx++;
                }
            }
            return bit_idx;
        }
        /* ── block_size = 16 ── */
        if (block_size == 16) {
            for (int row = sync_rows; row < grid_rows && bit_idx < max_bits; ++row) {
                size_t y = (size_t)grid_top_y + (size_t)row * 16;
                const uint8_t *RESTRICT row_base = gray + y * stride;
                for (int col = 0; col < grid_cols && bit_idx < max_bits; ++col) {
                    size_t x = (size_t)grid_left_x + (size_t)col * 16;
                    bits_out[bit_idx] = (uint8_t)((count_white_block16(row_base + x, stride) > 2) ? 1 : 0);
                    bit_idx++;
                }
            }
            return bit_idx;
        }
    }

    /* ── General fallback ── */
    for (int row = sync_rows; row < grid_rows && bit_idx < max_bits; ++row) {
        size_t y = (size_t)grid_top_y + (size_t)row * block_size;
        const uint8_t *RESTRICT row_base = gray + y * stride;
        for (int col = 0; col < grid_cols && bit_idx < max_bits; ++col) {
            size_t x = (size_t)grid_left_x + (size_t)col * block_size;
            int c = tile_count_white(row_base + x, stride,
                                     block_size, block_size, subsample);
            bits_out[bit_idx] = (uint8_t)((c > 2) ? 1 : 0);
            bit_idx++;
        }
    }
    return bit_idx;
}

/* ═══════════════════════════════════════════════════════════════════════
 * PHASE 4:  tile_read_calibration — SIMD-accelerated calibration reader
 *
 * Unlike tile decoding, calibration reads CONTIGUOUS pixel rectangles
 * (~22×22 px per cell, 192 cells).  SSE2 sums 16 px/op, AVX2 sums 32.
 * This is the only path where SIMD provides real benefit.
 * ═══════════════════════════════════════════════════════════════════════ */

/* ── SSE2:  _mm_sad_epu8 sums 16 bytes → 2 × uint64 per iteration ── */
static int cal_read_sse2(const uint8_t *gray, int stride,
                          int width, int height,
                          uint8_t *bits, int max_bits)
{
    int cal_top    = (int)((float)height * CAL_TOP_FRAC);
    int cal_height = (int)((float)height * CAL_HEIGHT_FRAC);
    int cal_left   = (int)((float)width  * CAL_LEFT_FRAC);
    int cal_width  = (int)((float)width  * CAL_WIDTH_FRAC);

    if (cal_height < CAL_ROWS || cal_width < CAL_COLS) return 0;

    int cell_w    = cal_width  / CAL_COLS;
    int cell_h    = cal_height / CAL_ROWS;
    int sample_w  = (cell_w * 60) / 100;  if (sample_w < 2) sample_w = 2;
    int sample_h  = (cell_h * 60) / 100;  if (sample_h < 2) sample_h = 2;

    const __m128i vzero = _mm_setzero_si128();

    int count = 0;
    for (int row = 0; row < CAL_ROWS && count < max_bits; ++row) {
        for (int col = 0; col < CAL_COLS && count < max_bits; ++col) {
            int cx = cal_left + col * cell_w + cell_w / 2;
            int cy = cal_top  + row * cell_h + cell_h / 2;
            int sx = cx - sample_w / 2;
            int sy = cy - sample_h / 2;
            if (sx < 0) sx = 0;
            if (sy < 0) sy = 0;
            int x_end = sx + sample_w;  if (x_end > width)  x_end = width;
            int y_end = sy + sample_h;  if (y_end > height) y_end = height;

            int row_bytes = x_end - sx;
            if (row_bytes <= 0) return 0;

            int total_sum = 0;
            for (int yy = sy; yy < y_end; ++yy) {
                const uint8_t *RESTRICT row_ptr = gray + (size_t)yy * stride + sx;
                __m128i vsum = vzero;
                int xx = 0;

                /* Process 16 bytes per iteration */
                for (; xx + 16 <= row_bytes; xx += 16) {
                    __m128i v = _mm_loadu_si128((const __m128i *)(row_ptr + xx));
                    vsum = _mm_add_epi64(vsum, _mm_sad_epu8(v, vzero));
                }

                /* Horizontal reduce:  vsum = 2 × uint64 sums */
                int row_sum = (int)_mm_cvtsi128_si32(vsum)
                            + (int)_mm_cvtsi128_si32(_mm_srli_si128(vsum, 8));

                /* Scalar remainder */
                for (; xx < row_bytes; ++xx) row_sum += row_ptr[xx];
                total_sum += row_sum;
            }

            int samples = (y_end - sy) * row_bytes;
            if (samples == 0) return 0;
            bits[count++] = (uint8_t)((total_sum / samples) >= THRESHOLD ? 1 : 0);
        }
    }
    return count;
}

#if HAS_AVX2
/* ── AVX2:  _mm256_sad_epu8 sums 32 bytes → 4 × uint64 per iteration ── */
static int cal_read_avx2(const uint8_t *gray, int stride,
                          int width, int height,
                          uint8_t *bits, int max_bits)
{
    int cal_top    = (int)((float)height * CAL_TOP_FRAC);
    int cal_height = (int)((float)height * CAL_HEIGHT_FRAC);
    int cal_left   = (int)((float)width  * CAL_LEFT_FRAC);
    int cal_width  = (int)((float)width  * CAL_WIDTH_FRAC);

    if (cal_height < CAL_ROWS || cal_width < CAL_COLS) return 0;

    int cell_w    = cal_width  / CAL_COLS;
    int cell_h    = cal_height / CAL_ROWS;
    int sample_w  = (cell_w * 60) / 100;  if (sample_w < 2) sample_w = 2;
    int sample_h  = (cell_h * 60) / 100;  if (sample_h < 2) sample_h = 2;

    const __m256i vzero = _mm256_setzero_si256();

    int count = 0;
    for (int row = 0; row < CAL_ROWS && count < max_bits; ++row) {
        for (int col = 0; col < CAL_COLS && count < max_bits; ++col) {
            int cx = cal_left + col * cell_w + cell_w / 2;
            int cy = cal_top  + row * cell_h + cell_h / 2;
            int sx = cx - sample_w / 2;
            int sy = cy - sample_h / 2;
            if (sx < 0) sx = 0;
            if (sy < 0) sy = 0;
            int x_end = sx + sample_w;  if (x_end > width)  x_end = width;
            int y_end = sy + sample_h;  if (y_end > height) y_end = height;

            int row_bytes = x_end - sx;
            if (row_bytes <= 0) return 0;

            int total_sum = 0;
            for (int yy = sy; yy < y_end; ++yy) {
                const uint8_t *RESTRICT row_ptr = gray + (size_t)yy * stride + sx;
                __m256i vsum = vzero;
                int xx = 0;

                /* Process 32 bytes per iteration */
                for (; xx + 32 <= row_bytes; xx += 32) {
                    __m256i v = _mm256_loadu_si256((const __m256i *)(row_ptr + xx));
                    vsum = _mm256_add_epi64(vsum, _mm256_sad_epu8(v, vzero));
                }

                /* Horizontal reduce:  combine 128-bit lanes → 2 × uint64 */
                __m128i lo = _mm256_castsi256_si128(vsum);
                __m128i hi = _mm256_extracti128_si256(vsum, 1);
                __m128i comb = _mm_add_epi64(lo, hi);
                int row_sum = (int)_mm_cvtsi128_si32(comb)
                            + (int)_mm_cvtsi128_si32(_mm_srli_si128(comb, 8));

                /* Scalar remainder */
                for (; xx < row_bytes; ++xx) row_sum += row_ptr[xx];
                total_sum += row_sum;
            }

            int samples = (y_end - sy) * row_bytes;
            if (samples == 0) return 0;
            bits[count++] = (uint8_t)((total_sum / samples) >= THRESHOLD ? 1 : 0);
        }
    }
    return count;
}
#endif /* HAS_AVX2 */

/* ── Dispatch for tile_read_calibration ─────────────────────────────
 * SSE2 is always available on x86-64.  AVX2 is checked at init via CPUID
 * and selected if compiled in AND the CPU supports it.
 * ─────────────────────────────────────────────────────────────────────── */

typedef int (*cal_read_fn)(const uint8_t *, int, int, int, uint8_t *, int);

static cal_read_fn  cal_read_impl    = NULL;
static int          cal_read_ready   = 0;

static void cal_read_init_once(void) {
    if (cal_read_ready) return;
    cal_read_ready = 1;

#if HAS_AVX2
    if (_cpu_has_avx2()) {
        cal_read_impl = cal_read_avx2;
        return;
    }
#endif
    /* SSE2 is guaranteed on x86-64 */
    cal_read_impl = cal_read_sse2;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Public API — tile_read_calibration
 * ═══════════════════════════════════════════════════════════════════════ */
int tile_read_calibration(const uint8_t *gray, int stride,
                          int width, int height,
                          uint8_t *bits, int max_bits)
{
    cal_read_init_once();
    return cal_read_impl(gray, stride, width, height, bits, max_bits);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Public API — tile_expand_bits  (encoder path, not hot)
 * ═══════════════════════════════════════════════════════════════════════ */
void tile_expand_bits(const uint8_t *src_bits,
                      uint8_t *frame, int stride,
                      int block_y, int block_x,
                      int block_size,
                      int grid_cols, int grid_rows)
{
    for (int row = 0; row < grid_rows; ++row) {
        int y = block_y + row * block_size;
        for (int col = 0; col < grid_cols; ++col) {
            int bit_idx = row * grid_cols + col;
            uint8_t val = src_bits[bit_idx] ? 255 : 0;
            int x = block_x + col * block_size;
            for (int yy = y; yy < y + block_size; ++yy)
                memset(frame + yy * stride + x, val, (size_t)block_size);
        }
    }
}

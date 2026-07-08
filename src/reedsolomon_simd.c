/* ═══════════════════════════════════════════════════════════════════════
 * reedsolomon_simd.c — SIMD-accelerated Reed-Solomon for Vidcrypt
 *
 * Optimization summary:
 * ───────────────────────────────────────────────────────────────────────
 * 1. PSHUFB GF(256) multiply: 16 multiplies in ~4 cycles vs ~48+ scalar.
 *    Precomputed gf_lo[256][16] + gf_hi[256][16] tables (8KB, L1-hot).
 * 2. Vectorized syndrome:    Process 16 (SSSE3) or 32 (AVX2) syndromes
 *    in parallel per received byte, with precomputed power table.
 * 3. Matrix-multiply encode: Precompute (k × ecc) parity matrix at init;
 *    encode = data × matrix via SIMD GF mul, replacing LFSR.
 * 4. Vectorized Chien:       Evaluate error-locator at 16 positions
 *    simultaneously with PSHUFB constant-multiplier GF mul.
 * 5. Vectorized Forney:      SIMD omega polynomial computation + eval.
 * 6. Runtime CPU dispatch:   cpuid → AVX2 > SSSE3 > scalar (same
 *    pattern as simd_decode.c).
 *
 * Bit-identical to scalar implementation for all ECC levels.
 * ═══════════════════════════════════════════════════════════════════════ */

#include "reedsolomon_simd.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

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
 * SIMD header inclusion (mirrors simd_decode.c pattern)
 * ────────────────────────────────────────────────────────────────────── */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <emmintrin.h>      /* SSE2 */
#include <tmmintrin.h>      /* SSSE3 — required for PSHUFB */
#define HAS_X86_SIMD 1
#else
#define HAS_X86_SIMD 0
#define HAS_AVX2 0
#endif

#if HAS_X86_SIMD
#if defined(__AVX2__)
    #include <immintrin.h>  /* AVX2 */
    #define HAS_AVX2 1
#else
    #define HAS_AVX2 0
#endif
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * GF(256) SIMD multiply tables
 *
 * gf_lo[a][n] = a * n              for n in 0..15 (low nibble)
 * gf_hi[a][n] = a * (n << 4)       for n in 0..15 (high nibble)
 *
 * Then: a * b = pshufb(gf_lo[a], b & 0x0F) ^ pshufb(gf_hi[a], b >> 4)
 *
 * Total: 256 × 16 × 2 = 8192 bytes, L1-resident (L1D is 32KB on modern x86).
 * ═══════════════════════════════════════════════════════════════════════ */
uint8_t gf_lo_table[256][16];
uint8_t gf_hi_table[256][16];

static bool simd_tables_initialized = false;

static void gf_simd_tables_init(void) {
    if (simd_tables_initialized) return;
    simd_tables_initialized = true;

    gf256_init();  /* ensure gf_exp/gf_log/gf_mul_table are built */

    for (int a = 0; a < 256; a++) {
        for (int n = 0; n < 16; n++) {
            gf_lo_table[a][n] = gf_mul((uint8_t)a, (uint8_t)n);
            gf_hi_table[a][n] = gf_mul((uint8_t)a, (uint8_t)(n << 4));
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Syndrome power table
 *
 * syndrome_pow[pos][syn] = alpha^(syn * pos)
 *   pos = 0..254  (position from the right: 254-j)
 *   syn = 0..31   (syndrome index, up to max ecc=32)
 *
 * Stored as [256][32] — contiguous 32 bytes per position for SIMD load.
 * ═══════════════════════════════════════════════════════════════════════ */
void syndrome_pow_init(uint8_t table[256][32], int ecc) {
    gf256_init();
    for (int syn = 0; syn < ecc; syn++) {
        for (int pos = 0; pos < 255; pos++) {
            int power = (syn * pos) % 255;
            table[pos][syn] = gf_exp[power];
        }
        table[255][syn] = 0; /* padding */
    }
    /* zero-fill unused syndrome slots */
    for (int syn = ecc; syn < 32; syn++) {
        for (int pos = 0; pos < 256; pos++) {
            table[pos][syn] = 0;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Chien search step table
 *
 * The scalar Chien search evaluates sigma at alpha^(255-i) for i = 0..254,
 * which is the REVERSE order: alpha^0, alpha^254, alpha^253, ..., alpha^1.
 *
 * For SIMD, we evaluate at 16 consecutive REVERSE-order points:
 *   chien_rev_step[j][m] = alpha^(j * (255 - m))  for m = 0..15
 *   base_pow[j]          = alpha^(j * (255 - chunk))
 *
 * This ensures sigma(alpha^(255-(chunk+m))) for m=0..15 matches the scalar's
 * evaluation at alpha^(255-i) where i = chunk+m.
 * ═══════════════════════════════════════════════════════════════════════ */
static uint8_t chien_step[33][16];  /* chien_step[j][m] = alpha^(j * (255 - m)) */

static void chien_step_init(void) {
    static bool done = false;
    if (done) return;
    done = true;
    gf256_init();
    for (int j = 0; j <= 32; j++) {
        for (int m = 0; m < 16; m++) {
            /* alpha^(j * (255 - m)) = alpha^(255j - jm) = alpha^(-jm) = alpha^(255 - (jm mod 255)) */
            int power = (j * (255 - m)) % 255;
            chien_step[j][m] = (j == 0) ? 1 : gf_exp[power];
        }
    }
    /* j=0: all entries are alpha^0 = 1 */
}

/* ═══════════════════════════════════════════════════════════════════════
 * Encode parity matrix
 *
 * For systematic (255, k) code, parity = data × G where G is (k × ecc).
 * G[i][j] = coefficient of x^(ecc-1-j) in the remainder of
 *           x^(ecc + k-1-i) mod g(x).
 *
 * Stored as [k][ecc] — contiguous ecc bytes per row for SIMD load.
 *
 * Compute from LAST row upward:
 *   Row k-1: x^ecc mod g(x)  (lowest-degree data term)
 *   Row i:   x^(ecc+k-1-i) mod g(x) = x^(ecc+(k-1-i)) mod g(x)
 *
 * Start at x^ecc mod g(x) (row k-1), then multiply by x to get each
 * preceding row up to row 0: x^(ecc+k-1) mod g(x).
 * ═══════════════════════════════════════════════════════════════════════ */
void encode_matrix_init(const RSCodec *codec, uint8_t *matrix) {
    int k = (int)codec->msg_length;
    int ecc = (int)codec->ecc_symbols;

    /* Compute x^ecc mod g(x) — the contribution for the last data byte */
    uint8_t remainder[256];
    memset(remainder, 0, 256);
    remainder[0] = 1;  /* start with x^0 */
    for (int s = 0; s < ecc; s++) {
        uint8_t feedback = remainder[ecc - 1];
        for (int j = ecc - 1; j > 0; j--)
            remainder[j] = remainder[j - 1];
        remainder[0] = 0;
        if (feedback) {
            for (int j = 0; j < ecc; j++)
                remainder[j] ^= gf_mul(feedback, codec->generator[j]);
        }
    }

    /* Work from last row (i = k-1) back to first (i = 0),
     * multiplying by x each step to go up one degree */
    for (int i = k - 1; i >= 0; i--) {
        /* Store row i: remainder[ecc-1-j] reverses to MSB-first parity order */
        for (int j = 0; j < ecc; j++) {
            matrix[i * ecc + j] = remainder[ecc - 1 - j];
        }

        if (i > 0) {
            /* Multiply by x mod g(x) to get x^(ecc+k-1-(i-1)) = x^(ecc+k-i) */
            uint8_t feedback = remainder[ecc - 1];
            for (int j = ecc - 1; j > 0; j--)
                remainder[j] = remainder[j - 1];
            remainder[0] = 0;
            if (feedback) {
                for (int j = 0; j < ecc; j++)
                    remainder[j] ^= gf_mul(feedback, codec->generator[j]);
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * CPU feature detection (identical to simd_decode.c pattern)
 * ═══════════════════════════════════════════════════════════════════════ */
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
    if ((_xgetbv_cpuid(0) & 6) != 6) return 0;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ebx & (1U << 5)) != 0;           /* AVX2 */
}
#else
static int _cpu_has_avx2(void) { return 0; }
#endif
#endif /* HAS_AVX2 */

/* ═══════════════════════════════════════════════════════════════════════
 * SSSE3 helper: multiply scalar by 16-byte vector
 *
 * result[i] = scalar * vec[i]   for i = 0..15
 * Uses PSHUFB with gf_lo/gf_hi tables.
 * ═══════════════════════════════════════════════════════════════════════ */
#if HAS_X86_SIMD
static ALWAYS_INLINE __m128i gf_mul_scalar_vec_ssse3(uint8_t scalar, __m128i vec) {
    __m128i lo_tbl = _mm_loadu_si128((const __m128i *)gf_lo_table[scalar]);
    __m128i hi_tbl = _mm_loadu_si128((const __m128i *)gf_hi_table[scalar]);
    __m128i lo_nib = _mm_and_si128(vec, _mm_set1_epi8(0x0F));
    __m128i hi_nib = _mm_and_si128(_mm_srli_epi16(vec, 4), _mm_set1_epi8(0x0F));
    return _mm_xor_si128(
        _mm_shuffle_epi8(lo_tbl, lo_nib),
        _mm_shuffle_epi8(hi_tbl, hi_nib));
}
#endif

#if HAS_AVX2
static ALWAYS_INLINE __m256i gf_mul_scalar_vec_avx2(uint8_t scalar, __m256i vec) {
    __m128i lo128 = _mm_loadu_si128((const __m128i *)gf_lo_table[scalar]);
    __m128i hi128 = _mm_loadu_si128((const __m128i *)gf_hi_table[scalar]);
    __m256i lo_tbl = _mm256_set_m128i(lo128, lo128);
    __m256i hi_tbl = _mm256_set_m128i(hi128, hi128);
    __m256i lo_nib = _mm256_and_si256(vec, _mm256_set1_epi8(0x0F));
    __m256i hi_nib = _mm256_and_si256(_mm256_srli_epi16(vec, 4), _mm256_set1_epi8(0x0F));
    return _mm256_xor_si256(
        _mm256_shuffle_epi8(lo_tbl, lo_nib),
        _mm256_shuffle_epi8(hi_tbl, hi_nib));
}
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * SCALAR IMPLEMENTATIONS (portable, always available)
 *
 * These are functionally identical to the original reedsolomon.c code
 * but restructured to work with the dispatch system. They serve as the
 * non-x86 fallback and as a verification reference.
 * ═══════════════════════════════════════════════════════════════════════ */

void rs_scalar_compute_syndromes(const RSCodec *codec, const uint8_t *received,
                                 uint8_t *syndromes, int *all_zero) {
    int n = (int)codec->block_length;
    int ecc = (int)codec->ecc_symbols;
    int zero = 1;
    for (int i = 0; i < ecc; ++i) {
        uint8_t sum = 0;
        for (int j = 0; j < n; ++j) {
            if (received[j]) {
                int power = ((int)gf_log[received[j]] + i * (n - 1 - j)) % 255;
                if (power < 0) power += 255;
                sum ^= gf_exp[power];
            }
        }
        syndromes[i] = sum;
        if (sum != 0) zero = 0;
    }
    *all_zero = zero;
}

void rs_scalar_encode(const RSCodec *codec, const uint8_t *msg,
                      int msg_len, uint8_t *encoded) {
    int k = (int)codec->msg_length;
    int n = (int)codec->block_length;
    int ecc = (int)codec->ecc_symbols;

    memset(encoded, 0, (size_t)n);
    int copy_len = msg_len < k ? msg_len : k;
    memcpy(encoded, msg, (size_t)copy_len);

    uint8_t bb[256];
    memset(bb, 0, 256);

    for (int i = 0; i < k; ++i) {
        uint8_t feedback = (uint8_t)(encoded[i] ^ bb[ecc - 1]);
        if (feedback != 0) {
            for (int j = ecc - 1; j > 0; --j)
                bb[j] = (uint8_t)(bb[j - 1] ^ gf_mul(feedback, codec->generator[j]));
            bb[0] = gf_mul(feedback, codec->generator[0]);
        } else {
            for (int j = ecc - 1; j > 0; --j)
                bb[j] = bb[j - 1];
            bb[0] = 0;
        }
    }

    for (int i = 0; i < ecc; ++i)
        encoded[k + i] = bb[ecc - 1 - i];
}

int rs_scalar_chien_search(const uint8_t *sigma, int sigma_deg,
                           uint8_t *error_positions) {
    int count = 0;
    for (int i = 0; i < 255; ++i) {
        uint8_t sum = sigma[0];
        uint8_t pow_val = gf_exp[255 - i];  /* alpha^(-i) = alpha^(255-i) */
        uint8_t pow_accum = pow_val;
        for (int j = 1; j <= sigma_deg; ++j) {
            sum ^= gf_mul(sigma[j], pow_accum);
            pow_accum = gf_mul(pow_accum, pow_val);
        }
        if (sum == 0) error_positions[count++] = (uint8_t)(254 - i);
    }
    return count;
}

void rs_scalar_forney(const uint8_t *received, const uint8_t *sigma,
                      int sigma_deg, const uint8_t *error_positions,
                      int num_errors, const uint8_t *syndromes,
                      int ecc, uint8_t *decoded) {
    memcpy(decoded, received, 255);
    if (num_errors == 0) return;

    uint8_t omega[256] = {0};
    for (int i = 0; i < ecc; ++i) {
        if (syndromes[i] == 0) continue;
        for (int j = 0; j <= sigma_deg; ++j)
            if (sigma[j] != 0 && i + j < ecc)
                omega[i + j] ^= gf_mul(syndromes[i], sigma[j]);
    }

    for (int i = 0; i < num_errors; ++i) {
        int pos = (int)error_positions[i];
        uint8_t x_inv = gf_exp[255 - (254 - pos)];  /* = alpha^(-(254-pos)) = alpha^(pos+1)?  Let's verify:
           pos is in 0..254. We want X_i = alpha^(254-pos). x_inv = X_i^(-1) = alpha^(-(254-pos)).
           Since alpha^255 = 1, alpha^(-(254-pos)) = alpha^(255 - (254-pos)) = alpha^(1 + pos).
           The original code has: gf_exp[255 - (254 - pos)] = gf_exp[1 + pos] = alpha^(pos+1).
           Wait: original code has `gf_exp[255 - (254 - pos)]`.
           When pos=0: 255-254=1, gf_exp[1]=alpha^1 = alpha. X_i = alpha^(254-0) = alpha^254.
           X_i^(-1) = alpha^(-254) = alpha^(1). Yes, correct. */

        uint8_t denom = 0;
        for (int j = 1; j <= sigma_deg; j += 2)
            if (sigma[j] != 0) {
                int pow_idx = ((int)gf_log[x_inv] * (j - 1)) % 255;
                denom ^= gf_mul(sigma[j], gf_exp[pow_idx >= 0 ? pow_idx : pow_idx + 255]);
            }

        uint8_t num = 0, pow_val = 1;
        for (int j = 0; j < ecc; ++j) {
            if (omega[j] != 0) num ^= gf_mul(omega[j], pow_val);
            pow_val = gf_mul(pow_val, x_inv);
        }

        uint8_t error_val = gf_div(num, denom);
        uint8_t X_i = gf_exp[(254 - pos) % 255];
        error_val = gf_mul(X_i, error_val);
        decoded[pos] ^= error_val;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * SSSE3 IMPLEMENTATIONS
 * ═══════════════════════════════════════════════════════════════════════ */
#if HAS_X86_SIMD

static void ssse3_compute_syndromes(const RSCodec *codec, const uint8_t *received,
                                    uint8_t *syndromes, int *all_zero) {
    int n = (int)codec->block_length;
    int ecc = (int)codec->ecc_symbols;

    /* We precompute syndrome_pow during codec init and store it alongside the codec.
     * For simplicity, we compute it on first call and cache per ECC level.
     * The table is syndrome_pow[pos][syn] for pos = 254..0 (descending).
     * We store it as: position j maps to pow_idx = n-1-j = 254-j.
     * So we access syndrome_pow[254-j][syn]. */

    /* Static cache: one table per possible ECC value (only 32, 16, 8 used) */
    static uint8_t pow_cache[256][32] = {0};
    static int pow_cache_ecc = 0;
    if (pow_cache_ecc != ecc) {
        syndrome_pow_init(pow_cache, ecc);
        pow_cache_ecc = ecc;
    }

    /* Process 16 syndromes at a time with SSSE3 */
    __m128i acc[2];  /* up to 32 syndromes → 2×128-bit accumulators */
    int chunks = (ecc + 15) / 16;
    for (int c = 0; c < chunks; c++) acc[c] = _mm_setzero_si128();

    for (int j = 0; j < n; j++) {
        uint8_t r = received[j];
        if (r == 0) continue;

        int pow_idx = n - 1 - j;  /* 254..0 */
        const uint8_t *pow_row = pow_cache[pow_idx];

        __m128i lo_tbl = _mm_loadu_si128((const __m128i *)gf_lo_table[r]);
        __m128i hi_tbl = _mm_loadu_si128((const __m128i *)gf_hi_table[r]);
        __m128i lo_mask = _mm_set1_epi8(0x0F);

        for (int c = 0; c < chunks; c++) {
            __m128i pv = _mm_loadu_si128((const __m128i *)(pow_row + c * 16));
            __m128i lo_nib = _mm_and_si128(pv, lo_mask);
            __m128i hi_nib = _mm_and_si128(_mm_srli_epi16(pv, 4), lo_mask);
            __m128i prod = _mm_xor_si128(
                _mm_shuffle_epi8(lo_tbl, lo_nib),
                _mm_shuffle_epi8(hi_tbl, hi_nib));
            acc[c] = _mm_xor_si128(acc[c], prod);
        }
    }

    /* Extract results */
    for (int c = 0; c < chunks; c++) {
        _mm_storeu_si128((__m128i *)(syndromes + c * 16), acc[c]);
    }

    /* Check all-zero */
    int zero = 1;
    for (int i = 0; i < ecc; i++) {
        if (syndromes[i] != 0) { zero = 0; break; }
    }
    *all_zero = zero;
}

static void ssse3_rs_encode(const RSCodec *codec, const uint8_t *msg,
                            int msg_len, uint8_t *encoded) {
    int k = (int)codec->msg_length;
    int n = (int)codec->block_length;
    int ecc = (int)codec->ecc_symbols;

    memset(encoded, 0, (size_t)n);
    int copy_len = msg_len < k ? msg_len : k;
    memcpy(encoded, msg, (size_t)copy_len);

    /* Precompute parity matrix (cached per ECC level — NOT per pointer,
     * because stack-allocated RSCodec structs from different callers
     * can alias the same address after stack frame reuse). */
    static uint8_t encode_mat[256 * 32];  /* max 255×32 = 8160 bytes */
    static int encode_mat_ecc = 0;
    if (encode_mat_ecc != ecc) {
        encode_matrix_init(codec, encode_mat);
        encode_mat_ecc = ecc;
    }

    /* Matrix-vector multiply: parity = sum_i data[i] * matrix_row[i] */
    int chunks = (ecc + 15) / 16;
    __m128i acc[2];
    for (int c = 0; c < chunks; c++) acc[c] = _mm_setzero_si128();

    for (int i = 0; i < k; i++) {
        uint8_t d = encoded[i];  /* data byte */
        if (d == 0) continue;

        const uint8_t *row = encode_mat + (size_t)i * (size_t)ecc;

        __m128i lo_tbl = _mm_loadu_si128((const __m128i *)gf_lo_table[d]);
        __m128i hi_tbl = _mm_loadu_si128((const __m128i *)gf_hi_table[d]);
        __m128i lo_mask = _mm_set1_epi8(0x0F);

        for (int c = 0; c < chunks; c++) {
            int col_off = c * 16;
            __m128i rv = _mm_loadu_si128((const __m128i *)(row + col_off));
            __m128i lo_nib = _mm_and_si128(rv, lo_mask);
            __m128i hi_nib = _mm_and_si128(_mm_srli_epi16(rv, 4), lo_mask);
            __m128i prod = _mm_xor_si128(
                _mm_shuffle_epi8(lo_tbl, lo_nib),
                _mm_shuffle_epi8(hi_tbl, hi_nib));
            acc[c] = _mm_xor_si128(acc[c], prod);
        }
    }

    /* Extract parity */
    for (int c = 0; c < chunks; c++) {
        uint8_t tmp[16];
        _mm_storeu_si128((__m128i *)tmp, acc[c]);
        int col_off = c * 16;
        int col_len = (c == chunks - 1) ? ecc - col_off : 16;
        for (int j = 0; j < col_len; j++) {
            encoded[k + col_off + j] = tmp[j];
        }
    }
}

static int ssse3_chien_search(const uint8_t *sigma, int sigma_deg,
                              uint8_t *error_positions) {
    chien_step_init();
    int count = 0;

    /* Process 16 positions at a time */
    for (int chunk = 0; chunk < 255; chunk += 16) {
        __m128i eval = _mm_setzero_si128();

        /* sigma[0] contributes 1 to all evaluations */
        /* sigma[0] broadcast — XOR with set1 is correct since GF(2) is characteristic 2 */
        eval = _mm_xor_si128(eval, _mm_set1_epi8((char)sigma[0]));

        for (int j = 1; j <= sigma_deg; j++) {
            if (sigma[j] == 0) continue;

            /* base = alpha^(j * (255 - chunk)) — matches scalar eval at alpha^(255-chunk-m) */
            int pow_base = (j * (255 - chunk)) % 255;
            uint8_t base_pow = (j == 0) ? 1 : gf_exp[pow_base];
            uint8_t coeff = gf_mul(sigma[j], base_pow);

            if (coeff == 0) continue;

            /* chien_step[j][m] = alpha^(j * (255 - m)) — stepping BACKWARDS */
            __m128i lo_tbl = _mm_loadu_si128((const __m128i *)gf_lo_table[coeff]);
            __m128i hi_tbl = _mm_loadu_si128((const __m128i *)gf_hi_table[coeff]);
            __m128i step_vec = _mm_loadu_si128((const __m128i *)chien_step[j]);
            __m128i lo_nib = _mm_and_si128(step_vec, _mm_set1_epi8(0x0F));
            __m128i hi_nib = _mm_and_si128(_mm_srli_epi16(step_vec, 4), _mm_set1_epi8(0x0F));
            __m128i term = _mm_xor_si128(
                _mm_shuffle_epi8(lo_tbl, lo_nib),
                _mm_shuffle_epi8(hi_tbl, hi_nib));
            eval = _mm_xor_si128(eval, term);
        }

        /* Check which evaluations are zero */
        uint8_t eval_bytes[16];
        _mm_storeu_si128((__m128i *)eval_bytes, eval);
        for (int m = 0; m < 16 && (chunk + m) < 255; m++) {
            if (eval_bytes[m] == 0) {
                /* Position matches scalar: eval at alpha^(255-(chunk+m)) -> pos = 254-(chunk+m) */
                int pos = 254 - (chunk + m);
                error_positions[count++] = (uint8_t)pos;
            }
        }
    }

    return count;
}

static void ssse3_forney(const uint8_t *received, const uint8_t *sigma,
                         int sigma_deg, const uint8_t *error_positions,
                         int num_errors, const uint8_t *syndromes,
                         int ecc, uint8_t *decoded) {
    memcpy(decoded, received, 255);
    if (num_errors == 0) return;

    /* Omega computation: omega[k] = sum_{i+j=k} syndrome[i]*sigma[j]  mod x^ecc */
    uint8_t omega[256];
    memset(omega, 0, 256);

    for (int i = 0; i < ecc; i++) {
        if (syndromes[i] == 0) continue;
        uint8_t syn = syndromes[i];
        __m128i lo_tbl = _mm_loadu_si128((const __m128i *)gf_lo_table[syn]);
        __m128i hi_tbl = _mm_loadu_si128((const __m128i *)gf_hi_table[syn]);

        /* Process sigma[0..sigma_deg] in 16-byte chunks, adding to omega[i..i+sigma_deg] */
        int sigma_chunks = (sigma_deg + 16) / 16;
        for (int sc = 0; sc < sigma_chunks; sc++) {
            uint8_t omega_chunk[16];
            __m128i sig_vec = _mm_loadu_si128((const __m128i *)(sigma + sc * 16));
            __m128i lo_nib = _mm_and_si128(sig_vec, _mm_set1_epi8(0x0F));
            __m128i hi_nib = _mm_and_si128(_mm_srli_epi16(sig_vec, 4), _mm_set1_epi8(0x0F));
            __m128i prod = _mm_xor_si128(
                _mm_shuffle_epi8(lo_tbl, lo_nib),
                _mm_shuffle_epi8(hi_tbl, hi_nib));
            _mm_storeu_si128((__m128i *)omega_chunk, prod);
            int base = sc * 16;
            int max_j = (sigma_deg - base) > 15 ? 15 : sigma_deg - base;
            for (int j = 0; j <= max_j && i + j + base < ecc; j++) {
                omega[i + j + base] ^= omega_chunk[j];
            }
        }
    }

    /* Forney error evaluation (scalar per error position — few errors typical) */
    for (int ei = 0; ei < num_errors; ei++) {
        int pos = (int)error_positions[ei];
        uint8_t x_inv = gf_exp[255 - (254 - pos)];

        /* denominator: sigma_odd(x_inv) */
        uint8_t denom = 0;
        for (int j = 1; j <= sigma_deg; j += 2) {
            if (sigma[j] != 0) {
                int pow_idx = ((int)gf_log[x_inv] * (j - 1)) % 255;
                denom ^= gf_mul(sigma[j], gf_exp[pow_idx >= 0 ? pow_idx : pow_idx + 255]);
            }
        }

        /* numerator: omega(x_inv) */
        uint8_t num = 0, pow_val = 1;
        for (int j = 0; j < ecc; j++) {
            if (omega[j] != 0) num ^= gf_mul(omega[j], pow_val);
            pow_val = gf_mul(pow_val, x_inv);
        }

        uint8_t error_val = gf_div(num, denom);
        uint8_t X_i = gf_exp[(254 - pos) % 255];
        error_val = gf_mul(X_i, error_val);
        decoded[pos] ^= error_val;
    }
}

#endif /* HAS_X86_SIMD */

/* ═══════════════════════════════════════════════════════════════════════
 * AVX2 IMPLEMENTATIONS
 * ═══════════════════════════════════════════════════════════════════════ */
#if HAS_AVX2

static void avx2_compute_syndromes(const RSCodec *codec, const uint8_t *received,
                                   uint8_t *syndromes, int *all_zero) {
    int n = (int)codec->block_length;
    int ecc = (int)codec->ecc_symbols;

    static uint8_t pow_cache[256][32] = {0};
    static int pow_cache_ecc = 0;
    if (pow_cache_ecc != ecc) {
        syndrome_pow_init(pow_cache, ecc);
        pow_cache_ecc = ecc;
    }

    /* AVX2: process 32 syndromes in one pass */
    __m256i acc = _mm256_setzero_si256();

    for (int j = 0; j < n; j++) {
        uint8_t r = received[j];
        if (r == 0) continue;

        int pow_idx = n - 1 - j;
        const uint8_t *pow_row = pow_cache[pow_idx];

        __m256i pv = _mm256_loadu_si256((const __m256i *)pow_row);

        __m128i lo128 = _mm_loadu_si128((const __m128i *)gf_lo_table[r]);
        __m128i hi128 = _mm_loadu_si128((const __m128i *)gf_hi_table[r]);
        __m256i lo_tbl = _mm256_set_m128i(lo128, lo128);
        __m256i hi_tbl = _mm256_set_m128i(hi128, hi128);

        __m256i lo_nib = _mm256_and_si256(pv, _mm256_set1_epi8(0x0F));
        __m256i hi_nib = _mm256_and_si256(_mm256_srli_epi16(pv, 4), _mm256_set1_epi8(0x0F));

        __m256i prod = _mm256_xor_si256(
            _mm256_shuffle_epi8(lo_tbl, lo_nib),
            _mm256_shuffle_epi8(hi_tbl, hi_nib));
        acc = _mm256_xor_si256(acc, prod);
    }

    _mm256_storeu_si256((__m256i *)syndromes, acc);

    int zero = 1;
    for (int i = 0; i < ecc; i++) {
        if (syndromes[i] != 0) { zero = 0; break; }
    }
    *all_zero = zero;
}

static void avx2_rs_encode(const RSCodec *codec, const uint8_t *msg,
                           int msg_len, uint8_t *encoded) {
    int k = (int)codec->msg_length;
    int n = (int)codec->block_length;
    int ecc = (int)codec->ecc_symbols;

    memset(encoded, 0, (size_t)n);
    int copy_len = msg_len < k ? msg_len : k;
    memcpy(encoded, msg, (size_t)copy_len);

    /* Cache keyed by ECC level, NOT by pointer — stack-allocated RSCodec
     * structs from different callers can alias the same address. */
    static uint8_t encode_mat[256 * 32];
    static int encode_mat_ecc = 0;
    if (encode_mat_ecc != ecc) {
        encode_matrix_init(codec, encode_mat);
        encode_mat_ecc = ecc;
    }

    __m256i acc = _mm256_setzero_si256();

    for (int i = 0; i < k; i++) {
        uint8_t d = encoded[i];
        if (d == 0) continue;

        const uint8_t *row = encode_mat + (size_t)i * (size_t)ecc;
        __m256i rv = _mm256_loadu_si256((const __m256i *)row);

        __m128i lo128 = _mm_loadu_si128((const __m128i *)gf_lo_table[d]);
        __m128i hi128 = _mm_loadu_si128((const __m128i *)gf_hi_table[d]);
        __m256i lo_tbl = _mm256_set_m128i(lo128, lo128);
        __m256i hi_tbl = _mm256_set_m128i(hi128, hi128);

        __m256i lo_nib = _mm256_and_si256(rv, _mm256_set1_epi8(0x0F));
        __m256i hi_nib = _mm256_and_si256(_mm256_srli_epi16(rv, 4), _mm256_set1_epi8(0x0F));

        __m256i prod = _mm256_xor_si256(
            _mm256_shuffle_epi8(lo_tbl, lo_nib),
            _mm256_shuffle_epi8(hi_tbl, hi_nib));
        acc = _mm256_xor_si256(acc, prod);
    }

    uint8_t tmp[32];
    _mm256_storeu_si256((__m256i *)tmp, acc);
    for (int j = 0; j < ecc; j++) {
        encoded[k + j] = tmp[j];
    }
}

static int avx2_chien_search(const uint8_t *sigma, int sigma_deg,
                             uint8_t *error_positions) {
    chien_step_init();
    int count = 0;

    for (int chunk = 0; chunk < 255; chunk += 16) {
        __m128i eval = _mm_setzero_si128();
        eval = _mm_xor_si128(eval, _mm_set1_epi8((char)sigma[0]));

        for (int j = 1; j <= sigma_deg; j++) {
            if (sigma[j] == 0) continue;

            /* alpha^(j * (255 - chunk)) — matches scalar eval order */
            int pow_base = (j * (255 - chunk)) % 255;
            uint8_t base_pow = (j == 0) ? 1 : gf_exp[pow_base];
            uint8_t coeff = gf_mul(sigma[j], base_pow);

            if (coeff == 0) continue;

            __m128i lo_tbl = _mm_loadu_si128((const __m128i *)gf_lo_table[coeff]);
            __m128i hi_tbl = _mm_loadu_si128((const __m128i *)gf_hi_table[coeff]);
            __m128i step_vec = _mm_loadu_si128((const __m128i *)chien_step[j]);
            __m128i lo_nib = _mm_and_si128(step_vec, _mm_set1_epi8(0x0F));
            __m128i hi_nib = _mm_and_si128(_mm_srli_epi16(step_vec, 4), _mm_set1_epi8(0x0F));
            __m128i term = _mm_xor_si128(
                _mm_shuffle_epi8(lo_tbl, lo_nib),
                _mm_shuffle_epi8(hi_tbl, hi_nib));
            eval = _mm_xor_si128(eval, term);
        }

        uint8_t eval_bytes[16];
        _mm_storeu_si128((__m128i *)eval_bytes, eval);
        for (int m = 0; m < 16 && (chunk + m) < 255; m++) {
            if (eval_bytes[m] == 0) {
                error_positions[count++] = (uint8_t)(254 - (chunk + m));
            }
        }
    }

    return count;
}

static void avx2_forney(const uint8_t *received, const uint8_t *sigma,
                        int sigma_deg, const uint8_t *error_positions,
                        int num_errors, const uint8_t *syndromes,
                        int ecc, uint8_t *decoded) {
    /* Forney uses same omega computation as SSSE3 (max ~32 values, 128-bit fine).
     * Delegate to shared implementation. */
    ssse3_forney(received, sigma, sigma_deg, error_positions, num_errors,
                 syndromes, ecc, decoded);
}

#endif /* HAS_AVX2 */

/* ═══════════════════════════════════════════════════════════════════════
 * DISPATCH
 * ═══════════════════════════════════════════════════════════════════════ */

RSSimdFuncs rs_simd;
static RSPath forced_path = RS_PATH_AUTO;
static bool dispatch_ready = false;

void reedsolomon_simd_init(void) {
    if (dispatch_ready) return;
    dispatch_ready = true;

    gf_simd_tables_init();

    if (forced_path == RS_PATH_SCALAR) goto set_scalar;

#if HAS_AVX2
    if (forced_path == RS_PATH_AVX2 || _cpu_has_avx2()) {
        rs_simd.compute_syndromes  = avx2_compute_syndromes;
        rs_simd.rs_encode_simd     = avx2_rs_encode;
        rs_simd.chien_search       = avx2_chien_search;
        rs_simd.forney_algorithm   = avx2_forney;
        return;
    }
#endif

#if HAS_X86_SIMD
    if (forced_path == RS_PATH_SSSE3 || forced_path == RS_PATH_AUTO) {
        rs_simd.compute_syndromes  = ssse3_compute_syndromes;
        rs_simd.rs_encode_simd     = ssse3_rs_encode;
        rs_simd.chien_search       = ssse3_chien_search;
        rs_simd.forney_algorithm   = ssse3_forney;
        return;
    }
#endif

set_scalar:
    rs_simd.compute_syndromes  = rs_scalar_compute_syndromes;
    rs_simd.rs_encode_simd     = rs_scalar_encode;
    rs_simd.chien_search       = rs_scalar_chien_search;
    rs_simd.forney_algorithm   = rs_scalar_forney;
}

void rs_set_path(RSPath path) {
    forced_path = path;
    dispatch_ready = false;
    reedsolomon_simd_init();
}

const char *rs_path_name(void) {
    /* Check which function pointers are actually set */
    if (rs_simd.compute_syndromes == rs_scalar_compute_syndromes) return "scalar";
#if HAS_AVX2
    if (rs_simd.compute_syndromes == avx2_compute_syndromes) return "AVX2";
#endif
#if HAS_X86_SIMD
    if (rs_simd.compute_syndromes == ssse3_compute_syndromes) return "SSSE3";
#endif
    return "unknown";
}

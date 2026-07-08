#ifndef VIDCRYPT_REEDSOLOMON_SIMD_H
#define VIDCRYPT_REEDSOLOMON_SIMD_H

#include "reedsolomon.h"
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── SIMD GF(256) multiply tables ───────────────────────────────────
 * gf_lo[a][n] = a * n           for n in 0..15
 * gf_hi[a][n] = a * (n << 4)    for n in 0..15
 * Total: 8KB, L1-resident. Used by PSHUFB-based constant-multiplier
 * GF multiply: result[i] = pshufb(gf_lo[a], b[i]&0x0F) ^ pshufb(gf_hi[a], b[i]>>4)
 */
extern uint8_t gf_lo_table[256][16];
extern uint8_t gf_hi_table[256][16];

/* ── Syndrome power table ───────────────────────────────────────────
 * syndrome_pow[i][j] = alpha^(i*j) for i in 0..31, j in 0..254
 * Used to vectorize syndrome computation.
 * Allocated per-codec at init time.
 */
void syndrome_pow_init(uint8_t table[256][32], int ecc);

/* ── Encode parity matrix ──────────────────────────────────────────
 * Precomputed during codec init.  matrix[k][ecc] = generator coefficient
 * for data symbol i contributing to parity symbol j.
 * matrix[i][j] = coefficient of x^(ecc-1-j) in x^(ecc+k-1-i) mod g(x).
 */
void encode_matrix_init(const RSCodec *codec, uint8_t *matrix);

/* ── SIMD dispatch ───────────────────────────────────────────────── */
typedef struct {
    void (*compute_syndromes)(const RSCodec *codec, const uint8_t *received,
                              uint8_t *syndromes, int *all_zero);
    void (*rs_encode_simd)(const RSCodec *codec, const uint8_t *msg,
                           int msg_len, uint8_t *encoded);
    int  (*chien_search)(const uint8_t *sigma, int sigma_deg,
                         uint8_t *error_positions);
    void (*forney_algorithm)(const uint8_t *received, const uint8_t *sigma,
                             int sigma_deg, const uint8_t *error_positions,
                             int num_errors, const uint8_t *syndromes,
                             int ecc, uint8_t *decoded);
} RSSimdFuncs;

/* Called once at startup; detects CPU features, sets dispatch table. */
void reedsolomon_simd_init(void);

/* The active dispatch table. Set by reedsolomon_simd_init(). */
extern RSSimdFuncs rs_simd;

/* ── Direct scalar impl (for fallback and verification) ───────────── */
void rs_scalar_compute_syndromes(const RSCodec *codec, const uint8_t *received,
                                 uint8_t *syndromes, int *all_zero);
void rs_scalar_encode(const RSCodec *codec, const uint8_t *msg,
                      int msg_len, uint8_t *encoded);
int  rs_scalar_chien_search(const uint8_t *sigma, int sigma_deg,
                            uint8_t *error_positions);
void rs_scalar_forney(const uint8_t *received, const uint8_t *sigma,
                      int sigma_deg, const uint8_t *error_positions,
                      int num_errors, const uint8_t *syndromes,
                      int ecc, uint8_t *decoded);

/* ── Force a specific path (for benchmarking) ─────────────────────── */
typedef enum { RS_PATH_AUTO, RS_PATH_SCALAR, RS_PATH_SSSE3, RS_PATH_AVX2 } RSPath;
void rs_set_path(RSPath path);
const char *rs_path_name(void);

#ifdef __cplusplus
}
#endif

#endif /* VIDCRYPT_REEDSOLOMON_SIMD_H */

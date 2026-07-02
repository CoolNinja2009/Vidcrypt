#ifndef VIDCRYPT_REEDSOLOMON_H
#define VIDCRYPT_REEDSOLOMON_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* GF(256) primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1 (0x11D) */
extern uint8_t gf_log[256];
extern uint8_t gf_exp[512];

/* Full 256x256 precomputed multiplication table — replaces log/exp lookups
 * with a single memory load. 64 KB fits in L2 cache. ~2x faster for RS encode.
 * Initialized at startup by gf256_init(), so non-const. */
extern uint8_t gf_mul_table[256][256];

void gf256_init(void);

/* LUT-based fast multiplication (single lookup, no branches) */
static inline uint8_t gf_mul(uint8_t a, uint8_t b) {
    return gf_mul_table[a][b];
}

static inline uint8_t gf_div(uint8_t a, uint8_t b) {
    if (a == 0) return 0;
    if (b == 0) return 0;
    int diff = (int)gf_log[a] - (int)gf_log[b];
    return gf_exp[diff < 0 ? diff + 255 : diff];
}

static inline uint8_t gf_pow(uint8_t a, int power) {
    if (a == 0) return 0;
    if (power == 0) return 1;
    int log_a = (int)gf_log[a];
    int result = (log_a * power) % 255;
    if (result < 0) result += 255;
    return gf_exp[result];
}

static inline uint8_t gf_inv(uint8_t a) {
    if (a == 0) return 0;
    return gf_exp[255 - (int)gf_log[a]];
}

typedef struct {
    uint8_t  ecc_symbols;
    uint8_t  msg_length;
    uint8_t  block_length;
    uint8_t  generator[256];
    uint8_t  gen_degree;
    /* Precomputed parity dictionary for GPU/AVX2 acceleration.
     * Layout: dict[i * (256 * n_k) + v * n_k + j] = contribution of
     * byte value 'v' at message position 'i' to parity byte 'j'.
     * Size = k * 256 * n_k bytes (~1.8 MB for RS(255,223,32)).
     * NULL if not yet computed. Owned by the codec cache. */
    uint8_t *parity_dict;
    size_t   parity_dict_size;
} RSCodec;

bool rs_codec_init(RSCodec *codec, int ecc_symbols);
const RSCodec *rs_get_codec(int ecc_symbols);

int rs_encode(const RSCodec *codec, const uint8_t *msg, int msg_len, uint8_t *encoded);

typedef struct {
    int status;
    int corrected;
    uint8_t decoded[255];
} RSDecodeResult;

void rs_decode(const RSCodec *codec, const uint8_t *received, RSDecodeResult *result);

int rs_encode_block(const RSCodec *codec, const uint8_t *msg, int msg_len, uint8_t *encoded);
void rs_decode_block(const RSCodec *codec, const uint8_t *received,
                     uint8_t *decoded, int *status);

#ifdef __cplusplus
}
#endif

#endif /* VIDCRYPT_REEDSOLOMON_H */

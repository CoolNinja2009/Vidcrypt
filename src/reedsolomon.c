#include "reedsolomon.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define GF_PRIM_POLY 0x11Du

uint8_t gf_log[256];
uint8_t gf_exp[512];

/* Full 256x256 GF(256) multiplication LUT — 64 KB, fits in L2 cache.
 * Single lookup replaces log+add+exp+conditional. */
uint8_t gf_mul_table[256][256];

static bool tables_initialized = false;

void gf256_init(void) {
    if (tables_initialized) return;
    tables_initialized = true;

    uint16_t x = 1;
    for (int i = 0; i < 255; ++i) {
        gf_exp[i] = (uint8_t)x;
        gf_log[x] = (uint8_t)i;
        x = (uint16_t)(x << 1);
        if (x & 0x100) {
            x ^= GF_PRIM_POLY;
        }
    }
    gf_log[0] = 0;
    for (int i = 0; i < 255; ++i) {
        gf_exp[255 + i] = gf_exp[i];
    }

    /* Precompute full multiplication table */
    for (int a = 0; a < 256; ++a) {
        gf_mul_table[0][a] = 0;
        gf_mul_table[a][0] = 0;
    }
    for (int a = 1; a < 256; ++a) {
        int log_a = gf_log[a];
        for (int b = 1; b < 256; ++b) {
            int sum = log_a + gf_log[b];
            gf_mul_table[a][b] = gf_exp[sum >= 255 ? sum - 255 : sum];
        }
    }
}

bool rs_codec_init(RSCodec *codec, int ecc_symbols) {
    if (ecc_symbols <= 0 || ecc_symbols > 255) return false;
    gf256_init();
    memset(codec, 0, sizeof(RSCodec));
    codec->ecc_symbols  = (uint8_t)ecc_symbols;
    codec->msg_length   = (uint8_t)(255 - ecc_symbols);
    codec->block_length = 255;
    codec->gen_degree   = (uint8_t)ecc_symbols;

    memset(codec->generator, 0, 256);
    codec->generator[0] = 1;
    for (int i = 0; i < ecc_symbols; ++i) {
        uint8_t root = gf_exp[i];
        uint8_t tmp[256];
        memcpy(tmp, codec->generator, 256);
        for (int j = ecc_symbols; j > 0; --j)
            codec->generator[j] = codec->generator[j - 1];
        codec->generator[0] = 0;
        for (int j = 0; j <= ecc_symbols; ++j)
            codec->generator[j] ^= gf_mul(tmp[j], root);
    }

    /* ── Build parity dictionary for GPU/AVX2 acceleration ────────────
     * RS systematic encoding is linear:  parity = message × P
     * where P is a k×(n-k) matrix over GF(256).
     *
     * P[i][j] = coefficient of x^j in (x^(i+n-k) mod g(x))
     * Dict[i][v][j] = gf_mul(v, P[i][j])
     *
     * During encoding: parity[j] = XOR over i of Dict[i][msg[i]][j].
     * This replaces sequential polynomial division with pure XOR —
     * zero GF arithmetic on the hot path.                          */
    {
        int k   = (int)codec->msg_length;
        int n_k = (int)codec->ecc_symbols;
        int dict_stride = 256 * n_k;  /* bytes per message position */
        codec->parity_dict_size = (size_t)k * (size_t)dict_stride;
        codec->parity_dict = (uint8_t *)malloc(codec->parity_dict_size);
        if (!codec->parity_dict) { return false; }

        /* Step 1: compute P matrix iteratively.
         * P[0] = x^n_k mod g(x) = lower coefficients of g(x).
         * P[i+1] = x * P[i] mod g(x).                              */
        uint8_t *P = (uint8_t *)calloc((size_t)k * (size_t)n_k, 1);
        if (!P) { free(codec->parity_dict); codec->parity_dict = NULL; return false; }

        memcpy(P, codec->generator, (size_t)n_k);  /* P[0][*] = g[*] */

        for (int i = 1; i < k; ++i) {
            uint8_t *prev = P + (size_t)(i - 1) * (size_t)n_k;
            uint8_t *cur  = P + (size_t)i * (size_t)n_k;
            uint8_t high  = prev[n_k - 1];
            if (high != 0) {
                for (int j = n_k - 1; j > 0; --j)
                    cur[j] = prev[j - 1] ^ gf_mul(high, codec->generator[j]);
                cur[0] = gf_mul(high, codec->generator[0]);
            } else {
                for (int j = n_k - 1; j > 0; --j)
                    cur[j] = prev[j - 1];
                cur[0] = 0;
            }
        }

        /* Reverse P so P[i] = x^(n-1-i) mod g(x).
         * Message byte m_i (at position i) contributes x^(k-1-i+n_k) = x^(254-i).
         * Currently P[0]=x^32, P[1]=x^33, ..., P[222]=x^254 — need reversed. */
        for (int i = 0; i < k / 2; ++i) {
            int ri = k - 1 - i;
            for (int j = 0; j < n_k; ++j) {
                uint8_t tmp = P[i * n_k + j];
                P[i * n_k + j]      = P[ri * n_k + j];
                P[ri * n_k + j]     = tmp;
            }
        }

        /* Step 2: build dictionary from P (with parity byte order reversal).
         *
         * rs_encode stores parity as: encoded[k+i] = bb[ecc-1-i].
         * So encoded[k+0] = coefficient of x^(n_k-1), and
         *    encoded[k+n_k-1] = coefficient of x^0.
         *
         * The GPU kernel writes: enc[k + parity_j] where parity_j is the
         * thread index, and thread j computes the coefficient of x^j.
         *
         * To match rs_encode's output, Dict[i][v][parity_j] must give the
         * contribution to encoded[k + parity_j] = bb[n_k-1-parity_j]
         * = coefficient of x^(n_k-1-parity_j).
         *
         * So: Dict[i][v][j] = gf_mul(v, P[i][n_k-1-j])              */
        for (int i = 0; i < k; ++i) {
            uint8_t *dict_pos = codec->parity_dict + (size_t)i * (size_t)dict_stride;
            for (int v = 0; v < 256; ++v) {
                uint8_t *dict_v = dict_pos + (size_t)v * (size_t)n_k;
                for (int j = 0; j < n_k; ++j)
                    dict_v[j] = gf_mul_table[v][P[i * n_k + (n_k - 1 - j)]];
            }
        }
        free(P);
    }

    return true;
}

#include <pthread.h>
static pthread_mutex_t codec_cache_mutex = PTHREAD_MUTEX_INITIALIZER;
static RSCodec codec_cache[256];
static bool codec_cached[256] = {false};

const RSCodec *rs_get_codec(int ecc_symbols) {
    if (ecc_symbols <= 0 || ecc_symbols > 255) return NULL;
    pthread_mutex_lock(&codec_cache_mutex);
    if (!codec_cached[ecc_symbols]) {
        rs_codec_init(&codec_cache[ecc_symbols], ecc_symbols);
        codec_cached[ecc_symbols] = true;
    }
    pthread_mutex_unlock(&codec_cache_mutex);
    return &codec_cache[ecc_symbols];
}

int rs_encode(const RSCodec *codec, const uint8_t *msg, int msg_len, uint8_t *encoded) {
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

    return n;
}

int rs_encode_block(const RSCodec *codec, const uint8_t *msg, int msg_len, uint8_t *encoded) {
    if (codec == NULL) {
        memcpy(encoded, msg, (size_t)msg_len);
        return msg_len;
    }
    return rs_encode(codec, msg, msg_len, encoded);
}

static int compute_syndromes(const RSCodec *codec, const uint8_t *received, uint8_t *syndromes) {
    int n = (int)codec->block_length;
    int ecc = (int)codec->ecc_symbols;
    int all_zero = 1;
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
        if (sum != 0) all_zero = 0;
    }
    return all_zero;
}

static int berlekamp_massey(const uint8_t *syndromes, int ecc,
                            uint8_t *sigma, int *sigma_deg) {
    uint8_t C[256] = {0}, B[256] = {0};
    int L = 0, m = 1;
    uint8_t b = 1;
    B[0] = 1; C[0] = 1;

    for (int n = 0; n < ecc; ++n) {
        uint8_t d = syndromes[n];
        for (int i = 1; i <= L; ++i) d ^= gf_mul(C[i], syndromes[n - i]);

        if (d == 0) {
            ++m;
        } else {
            uint8_t T[256];
            memcpy(T, C, 256);
            uint8_t coeff = gf_div(d, b);
            for (int i = 0; i < 256 - m; ++i)
                if (B[i] != 0) C[i + m] ^= gf_mul(coeff, B[i]);
            if (2 * L <= n) {
                L = n + 1 - L;
                memcpy(B, T, 256);
                b = d;
                m = 1;
            } else {
                ++m;
            }
        }
    }
    memcpy(sigma, C, 256);
    *sigma_deg = L;
    return L;
}

static int chien_search(const uint8_t *sigma, int sigma_deg, uint8_t *error_positions) {
    int count = 0;
    for (int i = 0; i < 255; ++i) {
        uint8_t sum = sigma[0];
        uint8_t pow_val = gf_exp[255 - i];
        uint8_t pow_accum = pow_val;
        for (int j = 1; j <= sigma_deg; ++j) {
            sum ^= gf_mul(sigma[j], pow_accum);
            pow_accum = gf_mul(pow_accum, pow_val);
        }
        if (sum == 0) error_positions[count++] = (uint8_t)(254 - i);
    }
    return count;
}

static void forney_algorithm(const uint8_t *received, const uint8_t *sigma, int sigma_deg,
                              const uint8_t *error_positions, int num_errors,
                              const uint8_t *syndromes, int ecc, uint8_t *decoded) {
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
        uint8_t x_inv = gf_exp[255 - (254 - pos)];

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

void rs_decode(const RSCodec *codec, const uint8_t *received, RSDecodeResult *result) {
    int ecc = (int)codec->ecc_symbols;
    int k = (int)codec->msg_length;
    memset(result, 0, sizeof(RSDecodeResult));

    uint8_t syndromes[256];
    int no_errors = compute_syndromes(codec, received, syndromes);

    if (no_errors) {
        result->status = 0;
        result->corrected = 0;
        memcpy(result->decoded, received, (size_t)k);
        return;
    }

    uint8_t sigma[256];
    int sigma_deg;
    berlekamp_massey(syndromes, ecc, sigma, &sigma_deg);

    uint8_t error_positions[255];
    int num_errors = chien_search(sigma, sigma_deg, error_positions);

    if (num_errors == 0 || num_errors != sigma_deg) {
        result->status = -1;
        result->corrected = 0;
        memcpy(result->decoded, received, (size_t)k);
        return;
    }

    forney_algorithm(received, sigma, sigma_deg, error_positions, num_errors,
                     syndromes, ecc, result->decoded);
    result->status = (num_errors > 0) ? 1 : 0;
    result->corrected = num_errors;
}

void rs_decode_block(const RSCodec *codec, const uint8_t *received,
                     uint8_t *decoded, int *status) {
    if (codec == NULL) {
        memcpy(decoded, received, (size_t)255);
        if (status) *status = 0;
        return;
    }
    RSDecodeResult result;
    rs_decode(codec, received, &result);
    memcpy(decoded, result.decoded, (size_t)codec->msg_length);
    if (status) *status = result.status;
}

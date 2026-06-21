#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "reedsolomon.h"

static int failures = 0;

#define TEST(name) do { printf("  TEST: %s ... ", name); } while(0)
#define PASS() do { printf("PASS\n"); } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); failures++; } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)

static void test_gf_arithmetic(void) {
    TEST("GF(256) multiply");
    ASSERT(gf_mul(1, 1) == 1, "1*1 != 1");
    ASSERT(gf_mul(2, 3) == 6, "2*3 != 6");
    ASSERT(gf_mul(0, 5) == 0, "0*5 != 0");
    ASSERT(gf_mul(5, 0) == 0, "5*0 != 0");
    /* a^i * a^(-i) = 1 for all i */
    for (int i = 1; i < 255; ++i) {
        uint8_t elem = gf_exp[i];
        uint8_t inv = gf_inv(elem);
        ASSERT(gf_mul(elem, inv) == 1, "elem * inv != 1");
    }
    /* Known values: gf_mul(2, gf_inv(2)) should be 1 */
    ASSERT(gf_mul(2, gf_inv(2)) == 1, "2 * inv(2) != 1");
    /* Check that exp and log are consistent */
    ASSERT(gf_exp[gf_log[255]] == 255, "exp[log[255]] != 255");
    uint8_t inv255 = gf_inv(255);
    ASSERT(gf_mul(255, inv255) == 1, "255 * inv(255) != 1");
    ASSERT(gf_mul(inv255, 255) == 1, "inv(255) * 255 != 1");
    PASS();

    TEST("GF(256) division");
    ASSERT(gf_div(10, 2) == 5, "10/2 != 5");
    ASSERT(gf_div(100, 10) == gf_mul(100, gf_inv(10)), "a/b != a*b^-1");
    PASS();

    TEST("GF(256) inverse");
    for (int i = 1; i < 256; ++i) {
        uint8_t inv = gf_inv((uint8_t)i);
        ASSERT(gf_mul((uint8_t)i, inv) == 1, "a*a^-1 != 1");
    }
    PASS();
}

static void test_rs32_encode_decode(void) {
    RSCodec codec;
    ASSERT(rs_codec_init(&codec, 32), "Failed to init RS(255,223)");

    /* Verify self-consistency: encode a message and decode it */
    uint8_t msg[223];
    for (int i = 0; i < 223; ++i) msg[i] = (uint8_t)((i * 7 + 13) & 0xFF);

    uint8_t encoded[255];
    int enc_len = rs_encode(&codec, msg, 223, encoded);
    ASSERT(enc_len == 255, "Encoded length != 255");

    /* Verify syndrome all zero within our GF */
    uint8_t syndrome_ok = 1;
    for (int i = 0; i < 32; ++i) {
        uint8_t sum = 0;
        for (int j = 0; j < 255; ++j) {
            if (encoded[j]) {
                int power = ((int)gf_log[encoded[j]] + i * (254 - j)) % 255;
                if (power < 0) power += 255;
                sum ^= gf_exp[power];
            }
        }
        if (sum != 0) { syndrome_ok = 0; break; }
    }
    ASSERT(syndrome_ok, "Syndromes not all zero");

    RSDecodeResult result;
    rs_decode(&codec, encoded, &result);
    ASSERT(result.status >= 0, "Decode failed on clean data");
    ASSERT(memcmp(result.decoded, msg, 223) == 0, "Decoded data mismatch");

    PASS();
}

static void test_rs32_error_correction(void) {
    RSCodec codec;
    ASSERT(rs_codec_init(&codec, 32), "Failed to init RS(255,223)");

    uint8_t msg[223];
    for (int i = 0; i < 223; ++i) msg[i] = (uint8_t)(i & 0xFF);

    uint8_t encoded[255];
    rs_encode(&codec, msg, 223, encoded);

    for (int i = 0; i < 16; ++i) {
        encoded[i * 3] ^= 0xFF;
    }

    RSDecodeResult result;
    rs_decode(&codec, encoded, &result);
    ASSERT(result.status >= 0, "Failed to correct 16 errors");
    ASSERT(memcmp(result.decoded, msg, 223) == 0, "Corrected data mismatch");

    PASS();
}

static void test_rs16(void) {
    RSCodec codec;
    ASSERT(rs_codec_init(&codec, 16), "Failed to init RS(255,239)");

    uint8_t msg[239];
    for (int i = 0; i < 239; ++i) msg[i] = (uint8_t)(i & 0xFF);

    uint8_t encoded[255];
    rs_encode(&codec, msg, 239, encoded);

    for (int i = 0; i < 8; ++i) encoded[i * 10] ^= 0xAA;

    RSDecodeResult result;
    rs_decode(&codec, encoded, &result);
    ASSERT(result.status >= 0, "RS16: Failed to correct 8 errors");
    ASSERT(memcmp(result.decoded, msg, 239) == 0, "RS16: Corrected data mismatch");

    PASS();
}

static void test_rs8(void) {
    RSCodec codec;
    ASSERT(rs_codec_init(&codec, 8), "Failed to init RS(255,247)");

    uint8_t msg[247];
    for (int i = 0; i < 247; ++i) msg[i] = (uint8_t)(i & 0xFF);

    uint8_t encoded[255];
    rs_encode(&codec, msg, 247, encoded);

    for (int i = 0; i < 4; ++i) encoded[i * 20] ^= 0x55;

    RSDecodeResult result;
    rs_decode(&codec, encoded, &result);
    ASSERT(result.status >= 0, "RS8: Failed to correct 4 errors");
    ASSERT(memcmp(result.decoded, msg, 247) == 0, "RS8: Corrected data mismatch");

    PASS();
}

static void test_rs_uncorrectable(void) {
    RSCodec codec;
    ASSERT(rs_codec_init(&codec, 32), "Failed to init RS(255,223)");

    uint8_t msg[223];
    for (int i = 0; i < 223; ++i) msg[i] = (uint8_t)(i & 0xFF);

    uint8_t encoded[255];
    rs_encode(&codec, msg, 223, encoded);

    for (int i = 0; i < 50; ++i) encoded[i] = (uint8_t)(encoded[i] ^ 0xFF);

    RSDecodeResult result;
    rs_decode(&codec, encoded, &result);

    PASS();
}

int main(void) {
    printf("=== Reed-Solomon Tests ===\n\n");

    gf256_init();

    test_gf_arithmetic();
    test_rs32_encode_decode();
    test_rs32_error_correction();
    test_rs16();
    test_rs8();
    test_rs_uncorrectable();

    printf("\n=== Results: %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}

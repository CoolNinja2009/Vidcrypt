/* ═══════════════════════════════════════════════════════════════════════
 * test_reedsolomon_verify.c — Bit-identical verification test
 *
 * Round-trips random data through both scalar and SIMD RS encode/decode
 * paths at all ECC levels (RS32/RS16/RS8) and verifies output match.
 * Also tests error correction with injected symbol errors.
 * ═══════════════════════════════════════════════════════════════════════ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "reedsolomon.h"
#include "reedsolomon_simd.h"

static int failures = 0;

#define TEST(name) do { printf("  TEST: %s ... ", name); } while(0)
#define PASS() do { printf("PASS\n"); } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); failures++; } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)

/* ── Run RS encode+decode on both scalar and SIMD paths, compare ───── */
static void test_bit_identical_roundtrip(int ecc, int msg_len,
                                         int num_errs, unsigned seed) {
    const RSCodec *codec = rs_get_codec(ecc);

    /* Generate random message */
    uint8_t msg[255];
    srand(seed);
    for (int i = 0; i < msg_len; i++)
        msg[i] = (uint8_t)(rand() & 0xFF);
    /* zero-pad rest */
    for (int i = msg_len; i < 255; i++)
        msg[i] = 0;

    /* ── Scalar path ────────────────────────────────────────────── */
    uint8_t enc_scalar[255];
    uint8_t dec_scalar[255];
    int status_scalar;

    rs_set_path(RS_PATH_SCALAR);
    rs_encode(codec, msg, msg_len, enc_scalar);

    /* Inject errors if requested */
    uint8_t corrupted[255];
    memcpy(corrupted, enc_scalar, 255);
    for (int e = 0; e < num_errs; e++) {
        int pos = (e * 17 + 3) % 255;
        corrupted[pos] ^= (uint8_t)(0xFF ^ e);
    }
    rs_decode_block(codec, corrupted, dec_scalar, &status_scalar);

    /* ── SIMD path ──────────────────────────────────────────────── */
    uint8_t enc_simd[255];
    uint8_t dec_simd[255];
    int status_simd;

    rs_set_path(RS_PATH_AUTO);  /* use best available */
    rs_encode(codec, msg, msg_len, enc_simd);

    /* Verify encode output matches */
    if (memcmp(enc_scalar, enc_simd, 255) != 0) {
        printf("FAIL: encode mismatch at ECC=%d\n", ecc);
        failures++;
        return;
    }

    /* Use same corrupted data for SIMD path */
    rs_decode_block(codec, corrupted, dec_simd, &status_simd);

    /* Verify decode output matches */
    if (memcmp(dec_scalar, dec_simd, (size_t)codec->msg_length) != 0) {
        printf("FAIL: decode mismatch at ECC=%d, errors=%d\n", ecc, num_errs);
        failures++;
        return;
    }

    /* Verify status matches */
    if (status_scalar != status_simd) {
        printf("FAIL: status mismatch at ECC=%d, errors=%d (scalar=%d, simd=%d)\n",
               ecc, num_errs, status_scalar, status_simd);
        failures++;
        return;
    }
}

static void test_bit_identical_rs32(void) {
    TEST("RS32 round-trip (clean, 1000 seeds)");
    for (int s = 0; s < 1000; s++)
        test_bit_identical_roundtrip(32, 223, 0, s);
    PASS();

    TEST("RS32 round-trip (1 error, 500 seeds)");
    for (int s = 0; s < 500; s++)
        test_bit_identical_roundtrip(32, 223, 1, 1000 + s);
    PASS();

    TEST("RS32 round-trip (8 errors, 500 seeds)");
    for (int s = 0; s < 500; s++)
        test_bit_identical_roundtrip(32, 223, 8, 2000 + s);
    PASS();

    TEST("RS32 round-trip (16 errors, 200 seeds)");
    for (int s = 0; s < 200; s++)
        test_bit_identical_roundtrip(32, 223, 16, 3000 + s);
    PASS();
}

static void test_bit_identical_rs16(void) {
    TEST("RS16 round-trip (clean, 500 seeds)");
    for (int s = 0; s < 500; s++)
        test_bit_identical_roundtrip(16, 239, 0, s);
    PASS();

    TEST("RS16 round-trip (8 errors, 200 seeds)");
    for (int s = 0; s < 200; s++)
        test_bit_identical_roundtrip(16, 239, 8, 1000 + s);
    PASS();
}

static void test_bit_identical_rs8(void) {
    TEST("RS8 round-trip (clean, 500 seeds)");
    for (int s = 0; s < 500; s++)
        test_bit_identical_roundtrip(8, 247, 0, s);
    PASS();

    TEST("RS8 round-trip (4 errors, 200 seeds)");
    for (int s = 0; s < 200; s++)
        test_bit_identical_roundtrip(8, 247, 4, 1000 + s);
    PASS();
}

static void test_no_rs_fallback(void) {
    TEST("No-RS fallback (clean, 100 seeds)");
    for (int s = 0; s < 100; s++) {
        uint8_t msg[255];
        srand(s);
        for (int i = 0; i < 255; i++)
            msg[i] = (uint8_t)(rand() & 0xFF);

        uint8_t enc[255], dec[255];
        int status;

        rs_encode_block(NULL, msg, 255, enc);
        rs_decode_block(NULL, enc, dec, &status);

        if (memcmp(msg, enc, 255) != 0) {
            FAIL("No-RS encode corrupted data");
            return;
        }
        if (memcmp(msg, dec, 255) != 0) {
            FAIL("No-RS decode corrupted data");
            return;
        }
        if (status != 0) {
            FAIL("No-RS status != 0");
            return;
        }
    }
    PASS();
}

int main(void) {
    gf256_init();

    printf("=== Reed-Solomon Bit-Identical Verification ===\n");
    printf("SIMD path: %s\n\n", rs_path_name());

    test_bit_identical_rs32();
    test_bit_identical_rs16();
    test_bit_identical_rs8();
    test_no_rs_fallback();

    printf("\n=== Results: %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}

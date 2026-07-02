#include <stdio.h>
#include <string.h>
#include "crc16.h"

static int failures = 0;
#define TEST(name) do { printf("  TEST: %s ... ", name); } while(0)
#define PASS() do { printf("PASS\n"); } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); failures++; } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)

static void test_crc16_basic(void) {
    TEST("CRC-16-CCITT of empty data");
    ASSERT(crc16(NULL, 0) == 0xFFFF, "CRC of empty != 0xFFFF");
    PASS();

    TEST("CRC-16-CCITT of known values");
    /* "123456789" -> 0x29B1 (CRC-16-CCITT) */
    const uint8_t data[] = "123456789";
    uint16_t c = crc16(data, 9);
    ASSERT(c == 0x29B1, "CRC of '123456789' != 0x29B1");
    PASS();

    TEST("CRC-16-CCITT of all zeros");
    uint8_t zeros[16] = {0};
    ASSERT(crc16(zeros, 16) != 0, "CRC of zeros shouldn't be 0");
    PASS();
}

static void test_crc16_consistency(void) {
    TEST("CRC-16-CCITT consistency");
    uint8_t data[256];
    for (int i = 0; i < 256; ++i) data[i] = (uint8_t)i;
    uint16_t c1 = crc16(data, 256);
    uint16_t c2 = crc16(data, 256);
    ASSERT(c1 == c2, "CRC not deterministic");
    PASS();
}

int main(void) {
    printf("=== CRC-16 Tests ===\n\n");
    test_crc16_basic();
    test_crc16_consistency();
    printf("\n=== Results: %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}

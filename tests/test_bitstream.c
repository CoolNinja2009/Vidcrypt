#include <stdio.h>
#include <string.h>
#include "bitstream.h"

static int failures = 0;
#define TEST(name) do { printf("  TEST: %s ... ", name); } while(0)
#define PASS() do { printf("PASS\n"); } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); failures++; } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)

static void test_bits_to_bytes(void) {
    TEST("bits_to_bytes basic");
    uint8_t bits[] = {1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0};
    uint8_t bytes[4];
    size_t n = bits_to_bytes(bits, 16, bytes, 4);
    ASSERT(n == 2, "Expected 2 bytes");
    ASSERT(bytes[0] == 0xFF, "First byte != 0xFF");
    ASSERT(bytes[1] == 0x00, "Second byte != 0x00");
    PASS();
}

static void test_bytes_to_bits(void) {
    TEST("bytes_to_bits basic");
    uint8_t bytes[] = {0xFF, 0x00, 0xAA};
    uint8_t bits[24];
    bytes_to_bits(bytes, 3, bits, 24);
    ASSERT(bits[0] == 1, "Bit 0 != 1");
    ASSERT(bits[7] == 1, "Bit 7 != 1");
    ASSERT(bits[8] == 0, "Bit 8 != 0");
    ASSERT(bits[15] == 0, "Bit 15 != 0");
    /* 0xAA = 10101010 */
    ASSERT(bits[16] == 1, "Bit 16 != 1");
    ASSERT(bits[17] == 0, "Bit 17 != 0");
    PASS();
}

static void test_roundtrip(void) {
    TEST("bits <-> bytes roundtrip");
    uint8_t orig_bits[64];
    for (int i = 0; i < 64; ++i) orig_bits[i] = (uint8_t)(i % 2);
    uint8_t bytes[8];
    size_t nbytes = bits_to_bytes(orig_bits, 64, bytes, 8);
    ASSERT(nbytes == 8, "Expected 8 bytes");
    uint8_t decoded[64];
    bytes_to_bits(bytes, 8, decoded, 64);
    ASSERT(memcmp(orig_bits, decoded, 64) == 0, "Roundtrip mismatch");
    PASS();
}

static void test_get_set_bit(void) {
    TEST("get_bit / set_bit");
    uint64_t words[2] = {0};
    set_bit(words, 0, 1);
    set_bit(words, 63, 1);
    set_bit(words, 64, 1);
    set_bit(words, 127, 1);
    ASSERT(get_bit(words, 0) == 1, "Bit 0 != 1");
    ASSERT(get_bit(words, 63) == 1, "Bit 63 != 1");
    ASSERT(get_bit(words, 64) == 1, "Bit 64 != 1");
    ASSERT(get_bit(words, 127) == 1, "Bit 127 != 1");
    ASSERT(get_bit(words, 1) == 0, "Bit 1 != 0");
    ASSERT(get_bit(words, 65) == 0, "Bit 65 != 0");
    PASS();
}

int main(void) {
    printf("=== Bitstream Tests ===\n\n");
    test_bits_to_bytes();
    test_bytes_to_bits();
    test_roundtrip();
    test_get_set_bit();
    printf("\n=== Results: %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}

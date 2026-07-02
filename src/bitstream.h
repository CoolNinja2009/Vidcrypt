#ifndef VIDCRYPT_BITSTREAM_H
#define VIDCRYPT_BITSTREAM_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bits are stored in uint64_t words, MSB-first within each word.
 * Word 0 bits: [63=first bit, 62, ..., 0=64th bit]
 * Word 1 bits: [63=65th bit, ...] */

/* Store as many whole bytes as possible from bits[] into bytes[].
 * Returns number of bytes written = nbits / 8. MSB-first packing. */
size_t bits_to_bytes(const uint8_t *bits, size_t nbits, uint8_t *bytes, size_t max_bytes);

/* Convert bytes[] to bit array. MSB-first unpacking. */
void bytes_to_bits(const uint8_t *bytes, size_t nbytes, uint8_t *bits, size_t max_bits);

/* LUT-accelerated byte→bit unpacking. ~10x faster than the general loop.
 * 'bits_out' receives nbytes*8 bytes — each 0 or 1, MSB first per byte.
 * Bits pointer MUST be pre-allocated (nbytes * 8 bytes). */
void bytes_to_bits_fast(const uint8_t *bytes, size_t nbytes, uint8_t *bits_out);

/* LUT-accelerated byte→bit unpacking with 8-byte stride output.
 * Each source byte produces 8 bits written to bits_out at offsets
 * (i*8+0)...(i*8+7). This is the core hot-path for encoder bit packing. */
void bytes_to_bits_strided(const uint8_t *bytes, size_t nbytes,
                            uint8_t *bits_out, size_t stride);

/* Get direct pointer to the LUT row for a byte value.
 * Returns pointer to 8 uint8_t values (bits 7..0 of the byte). */
static inline const uint8_t* byte_to_bits_lut_row(uint8_t byte_val);

/* Pack bits into uint64_t words. Returns number of words written. */
size_t bits_to_words(const uint8_t *bits, size_t nbits, uint64_t *words, size_t max_words);

/* Unpack uint64_t words to bits. */
void words_to_bits(const uint64_t *words, size_t nwords, uint8_t *bits, size_t max_bits);

/* Count set bits in uint64_t packed array. */
#if defined(_MSC_VER)
#include <intrin.h>
static inline size_t count_bits_set(const uint64_t *words, size_t nwords) {
    size_t count = 0;
    for (size_t i = 0; i < nwords; ++i) {
#if defined(_M_X64) || defined(_M_ARM64)
        count += (size_t)__popcnt64(words[i]);
#else
        uint32_t lo = (uint32_t)(words[i] & 0xFFFFFFFFULL);
        uint32_t hi = (uint32_t)(words[i] >> 32);
        count += (size_t)(__popcnt(lo) + __popcnt(hi));
#endif
    }
    return count;
}
#else
static inline size_t count_bits_set(const uint64_t *words, size_t nwords) {
    size_t count = 0;
    for (size_t i = 0; i < nwords; ++i) {
        count += (size_t)__builtin_popcountll(words[i]);
    }
    return count;
}
#endif

/* Helper: get a single bit from packed array at given index. */
static inline int get_bit(const uint64_t *words, size_t index) {
    return (int)((words[index / 64] >> (63 - (index % 64))) & 1);
}

/* Helper: set a single bit in packed array at given index. */
static inline void set_bit(uint64_t *words, size_t index, int value) {
    uint64_t mask = (uint64_t)1 << (63 - (index % 64));
    if (value) {
        words[index / 64] |= mask;
    } else {
        words[index / 64] &= ~mask;
    }
}

/* Copy a range of bits from packed src to packed dst. */
void bit_copy_range(const uint64_t *src, size_t bit_offset, size_t nbits, uint64_t *dst);

/* Copy with offset on both sides. */
void bit_copy_offset(const uint64_t *src, size_t src_offset,
                     uint64_t *dst, size_t dst_offset, size_t nbits);

/* ─── LUT: byte → 8 bits (MSB-first) ──────────────────────────────────
 * Each entry maps one byte to its 8-bit expansion as separate uint8 values.
 * Index: byte value (0-255). Output: 8 bytes, each 0 or 1.
 * Usage: memcpy(dst, BYTE_TO_BITS_LUT[byte_val], 8);
 * This replaces the per-bit division loop and is ~10x faster. */
extern const uint8_t BYTE_TO_BITS_LUT[256][8];

/* 64-bit LUT alias: cast the existing 8-byte-per-entry table to uint64_t*.
 * Allows single-instruction expansion: dst64[i] = LUT64[b];
 * Eliminates 12M memcpy() calls in the hot encode path.
 * Safe on little-endian (x86) because LUT[b][0] is at the lowest address. */
#define LUT64 ((const uint64_t *)(const void *)BYTE_TO_BITS_LUT)

/* Inline lookup: returns pointer to 8 pre-computed bit bytes for 'byte_val' */
static inline const uint8_t* byte_to_bits_lut_row(uint8_t byte_val) {
    return BYTE_TO_BITS_LUT[byte_val];
}

#ifdef __cplusplus
}
#endif

#endif /* VIDCRYPT_BITSTREAM_H */

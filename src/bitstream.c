#include "bitstream.h"
#include <string.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

size_t bits_to_bytes(const uint8_t *bits, size_t nbits, uint8_t *bytes, size_t max_bytes) {
    size_t nbytes = nbits / 8;
    if (nbytes > max_bytes) nbytes = max_bytes;
    memset(bytes, 0, nbytes);
    for (size_t i = 0; i < nbytes * 8; ++i) {
        if (bits[i]) {
            bytes[i / 8] |= (uint8_t)(1 << (7 - (i % 8)));
        }
    }
    return nbytes;
}

void bytes_to_bits(const uint8_t *bytes, size_t nbytes, uint8_t *bits, size_t max_bits) {
    size_t nbits = nbytes * 8;
    if (nbits > max_bits) nbits = max_bits;
    for (size_t i = 0; i < nbits; ++i) {
        bits[i] = (uint8_t)((bytes[i / 8] >> (7 - (i % 8))) & 1);
    }
}

#ifdef __AVX2__
/* Inverse of _mm256_movemask_epi8: expand a 32-bit mask to 32 bytes
 * where byte k = (mask >> k) & 1. Uses AVX2 lane-based expansion. */
static inline __m256i inv_movemask_epi8(uint32_t mask) {
    /* Split mask into 8 4-bit groups, one per 32-bit lane */
    __m256i v = _mm256_setr_epi32(
        (int)((mask >>  0) & 0xF), (int)((mask >>  4) & 0xF),
        (int)((mask >>  8) & 0xF), (int)((mask >> 12) & 0xF),
        (int)((mask >> 16) & 0xF), (int)((mask >> 20) & 0xF),
        (int)((mask >> 24) & 0xF), (int)((mask >> 28) & 0xF)
    );
    /* Map 4 bits → 4 bytes with sign bit set per byte:
     * bit 0 → byte 0 MSB, bit 1 → byte 1 MSB, bit 2 → byte 2 MSB, bit 3 → byte 3 MSB */
    __m256i b0 = _mm256_and_si256(_mm256_slli_epi32(v, 7),  _mm256_set1_epi32(0x80808080u));
    __m256i b1 = _mm256_and_si256(_mm256_slli_epi32(v, 14), _mm256_set1_epi32(0x80808080u));
    __m256i b2 = _mm256_and_si256(_mm256_slli_epi32(v, 21), _mm256_set1_epi32(0x80808080u));
    __m256i b3 = _mm256_and_si256(_mm256_slli_epi32(v, 28), _mm256_set1_epi32(0x80808080u));
    __m256i bytes = _mm256_or_si256(_mm256_or_si256(b0, b1), _mm256_or_si256(b2, b3));
    /* Convert 0x80→0x01, 0x00→0x00 (shift right by 7 within 16-bit words) */
    return _mm256_srli_epi16(bytes, 7);
}
#endif

size_t bits_to_words(const uint8_t *bits, size_t nbits, uint64_t *words, size_t max_words) {
    size_t nwords = (nbits + 63) / 64;
    if (nwords > max_words) nwords = max_words;
    memset(words, 0, nwords * sizeof(uint64_t));

    size_t i = 0;
#ifdef __AVX2__
    /* AVX2: process 64 bits at a time using movemask */
    for (; i + 63 < nbits; i += 64) {
        __m256i v0 = _mm256_loadu_si256((const __m256i*)(bits + i));
        v0 = _mm256_slli_epi16(v0, 7);
        unsigned mask0 = _mm256_movemask_epi8(v0);
        __m256i v1 = _mm256_loadu_si256((const __m256i*)(bits + i + 32));
        v1 = _mm256_slli_epi16(v1, 7);
        unsigned mask1 = _mm256_movemask_epi8(v1);
        /* Bit-reverse for MSB-first packing */
        mask0 = ((mask0 & 0x55555555) << 1) | ((mask0 & 0xAAAAAAAA) >> 1);
        mask0 = ((mask0 & 0x33333333) << 2) | ((mask0 & 0xCCCCCCCC) >> 2);
        mask0 = ((mask0 & 0x0F0F0F0F) << 4) | ((mask0 & 0xF0F0F0F0) >> 4);
        mask0 = ((mask0 & 0x00FF00FF) << 8) | ((mask0 & 0xFF00FF00) >> 8);
        mask0 = (mask0 << 16) | (mask0 >> 16);
        mask1 = ((mask1 & 0x55555555) << 1) | ((mask1 & 0xAAAAAAAA) >> 1);
        mask1 = ((mask1 & 0x33333333) << 2) | ((mask1 & 0xCCCCCCCC) >> 2);
        mask1 = ((mask1 & 0x0F0F0F0F) << 4) | ((mask1 & 0xF0F0F0F0) >> 4);
        mask1 = ((mask1 & 0x00FF00FF) << 8) | ((mask1 & 0xFF00FF00) >> 8);
        mask1 = (mask1 << 16) | (mask1 >> 16);
        words[i / 64] = ((uint64_t)mask0 << 32) | (uint64_t)mask1;
    }
#endif
    /* Scalar tail (full path when no AVX2): process remaining bits */
    for (; i < nwords * 64 && i < nbits; ++i) {
        if (bits[i]) {
            words[i / 64] |= (uint64_t)1 << (63 - (i % 64));
        }
    }
    return nwords;
}

void words_to_bits(const uint64_t *words, size_t nwords, uint8_t *bits, size_t max_bits) {
    size_t total_bits = nwords * 64;
    if (total_bits > max_bits) total_bits = max_bits;

    size_t i = 0;
#ifdef __AVX2__
    /* AVX2: unpack 64 bits at a time using inverse movemask */
    for (; i + 63 < total_bits; i += 64) {
        uint64_t w = words[i / 64];
        /* Full 64-bit bit-reversal to undo bits_to_words reversal */
        w = ((w & 0x5555555555555555ULL) << 1) | ((w & 0xAAAAAAAAAAAAAAAAULL) >> 1);
        w = ((w & 0x3333333333333333ULL) << 2) | ((w & 0xCCCCCCCCCCCCCCCCULL) >> 2);
        w = ((w & 0x0F0F0F0F0F0F0F0FULL) << 4) | ((w & 0xF0F0F0F0F0F0F0F0ULL) >> 4);
        w = ((w & 0x00FF00FF00FF00FFULL) << 8) | ((w & 0xFF00FF00FF00FF00ULL) >> 8);
        w = ((w & 0x0000FFFF0000FFFFULL) << 16) | ((w & 0xFFFF0000FFFF0000ULL) >> 16);
        w = (w << 32) | (w >> 32);
        /* Split into two 32-bit masks; after reversal, each maps directly to output bytes */
        uint32_t mask0 = (uint32_t)(w >> 32);
        uint32_t mask1 = (uint32_t)w;
        /* Inverse movemask: expand each 32-bit mask to 32 bytes */
        __m256i bytes0 = inv_movemask_epi8(mask0);
        __m256i bytes1 = inv_movemask_epi8(mask1);
        _mm256_storeu_si256((__m256i*)(bits + i), bytes0);
        _mm256_storeu_si256((__m256i*)(bits + i + 32), bytes1);
    }
#endif
    /* Scalar tail (full path when no AVX2): process remaining bits */
    for (; i < total_bits; ++i) {
        bits[i] = (uint8_t)((words[i / 64] >> (63 - (i % 64))) & 1);
    }
}

void bit_copy_range(const uint64_t *src, size_t bit_offset, size_t nbits, uint64_t *dst) {
    memset(dst, 0, ((nbits + 63) / 64) * sizeof(uint64_t));
    for (size_t i = 0; i < nbits; ++i) {
        if (get_bit(src, bit_offset + i)) {
            set_bit(dst, i, 1);
        }
    }
}

void bit_copy_offset(const uint64_t *src, size_t src_offset,
                     uint64_t *dst, size_t dst_offset, size_t nbits) {
    for (size_t i = 0; i < nbits; ++i) {
        int bit = get_bit(src, src_offset + i);
        set_bit(dst, dst_offset + i, bit);
    }
}

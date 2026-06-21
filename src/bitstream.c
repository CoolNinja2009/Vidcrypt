#include "bitstream.h"
#include <string.h>

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

size_t bits_to_words(const uint8_t *bits, size_t nbits, uint64_t *words, size_t max_words) {
    size_t nwords = (nbits + 63) / 64;
    if (nwords > max_words) nwords = max_words;
    memset(words, 0, nwords * sizeof(uint64_t));
    for (size_t i = 0; i < nwords * 64 && i < nbits; ++i) {
        if (bits[i]) {
            words[i / 64] |= (uint64_t)1 << (63 - (i % 64));
        }
    }
    return nwords;
}

void words_to_bits(const uint64_t *words, size_t nwords, uint8_t *bits, size_t max_bits) {
    size_t total_bits = nwords * 64;
    if (total_bits > max_bits) total_bits = max_bits;
    for (size_t i = 0; i < total_bits; ++i) {
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

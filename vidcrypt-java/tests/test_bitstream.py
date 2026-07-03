"""Tests for bitstream operations."""

import pytest
import random


def bits_to_bytes(bits, nbits, max_bytes):
    """Match BitStream.bitsToBytes exactly."""
    nbytes = nbits // 8
    if nbytes > max_bytes:
        nbytes = max_bytes
    result = bytearray(nbytes)
    for i in range(nbytes * 8):
        if bits[i]:
            result[i // 8] |= 1 << (7 - (i % 8))
    return bytes(result), nbytes


def bytes_to_bits(bts, nbytes, max_bits):
    """Match BitStream.bytesToBits exactly."""
    nbits = nbytes * 8
    if nbits > max_bits:
        nbits = max_bits
    result = bytearray(nbits)
    for i in range(nbits):
        result[i] = (bts[i // 8] >> (7 - (i % 8))) & 1
    return bytes(result)


def bits_to_words(bits, nbits, max_words):
    nwords = (nbits + 63) // 64
    if nwords > max_words:
        nwords = max_words
    words = [0] * nwords
    for i in range(min(nwords * 64, nbits)):
        if bits[i]:
            words[i // 64] |= 1 << (63 - (i % 64))
    return words, nwords


def words_to_bits(words, nwords, max_bits):
    total_bits = nwords * 64
    if total_bits > max_bits:
        total_bits = max_bits
    result = bytearray(total_bits)
    for i in range(total_bits):
        result[i] = (words[i // 64] >> (63 - (i % 64))) & 1
    return bytes(result)


class TestBitsToBytes:
    def test_empty(self):
        b, n = bits_to_bytes([], 0, 100)
        assert n == 0
        assert b == b''

    def test_single_byte(self):
        bits = [1, 0, 0, 0, 0, 0, 0, 0]  # 0x80
        b, n = bits_to_bytes(bits, 8, 10)
        assert n == 1
        assert b == b'\x80'

    def test_two_bytes(self):
        # 0xAA 0x55
        bits = [1,0,1,0,1,0,1,0, 0,1,0,1,0,1,0,1]
        b, n = bits_to_bytes(bits, 16, 10)
        assert n == 2
        assert b == b'\xaa\x55'

    def test_roundtrip(self):
        original = bytes(random.randint(0, 255) for _ in range(100))
        bits = bytes_to_bits(original, len(original), len(original) * 8)
        b, n = bits_to_bytes(bits, len(original) * 8, len(original))
        assert n == len(original)
        assert b == original

    def test_max_bytes_limit(self):
        bits = [1] * 80  # 10 bytes
        b, n = bits_to_bytes(bits, 80, 5)
        assert n == 5


class TestBitsToWords:
    def test_single_bit(self):
        bits = [1] + [0] * 63
        words, n = bits_to_words(bits, 1, 1)
        assert n == 1
        assert words[0] == (1 << 63)

    def test_64_bits(self):
        bits = [1] * 64
        words, n = bits_to_words(bits, 64, 1)
        assert n == 1
        assert words[0] == 0xFFFFFFFFFFFFFFFF

    def test_roundtrip(self):
        for nbits in [8, 64, 128, 256]:
            bits = [random.randint(0, 1) for _ in range(nbits)]
            words, nw = bits_to_words(bits, nbits, (nbits + 63) // 64)
            bits2 = words_to_bits(words, nw, nbits)
            assert bytes(bits) == bits2


class TestCountBitsSet:
    def count_bits_set(words, nwords):
        return sum(bin(w).count('1') for w in words[:nwords])

    def test_all_zeros(self):
        words = [0] * 10
        assert TestCountBitsSet.count_bits_set(words, 10) == 0

    def test_full(self):
        words = [0xFFFFFFFFFFFFFFFF] * 5
        assert TestCountBitsSet.count_bits_set(words, 5) == 5 * 64

"""
Bitstream tests — pure Python implementation.

Tests bits<->bytes conversion and get/set bit operations matching
the C API in bitstream.h (MSB-first packing within uint64_t words).
"""

from typing import List


# ── Pure Python reimplementations ───────────────────────────────────

def bits_to_bytes(bits: List[int], nbits: int, max_bytes: int) -> int:
    """Convert bits (list of 0/1) to bytes, MSB-first. Returns bytes written."""
    nbytes = min(nbits // 8, max_bytes)
    for i in range(nbytes):
        byte = 0
        for j in range(8):
            if bits[i * 8 + j]:
                byte |= 1 << (7 - j)
        yield byte
    return nbytes


def bits_to_bytes_list(bits: List[int], nbits: int, max_bytes: int) -> List[int]:
    """Non-generator version that returns a list."""
    nbytes = min(nbits // 8, max_bytes)
    result = []
    for i in range(nbytes):
        byte = 0
        for j in range(8):
            if bits[i * 8 + j]:
                byte |= 1 << (7 - j)
        result.append(byte)
    return result


def bytes_to_bits(bytes_: List[int], nbytes: int, max_bits: int) -> List[int]:
    """Convert bytes to bits (list of 0/1), MSB-first."""
    nbits = min(nbytes * 8, max_bits)
    result = [0] * nbits
    for i in range(nbits):
        byte_idx = i // 8
        bit_idx = 7 - (i % 8)
        result[i] = 1 if (bytes_[byte_idx] >> bit_idx) & 1 else 0
    return result


def get_bit(words: List[int], index: int) -> int:
    """Get a single bit from a uint64_t-packed array at given index."""
    return 1 if (words[index // 64] >> (63 - (index % 64))) & 1 else 0


def set_bit(words: List[int], index: int, value: int) -> None:
    """Set a single bit in a uint64_t-packed array at given index."""
    mask = 1 << (63 - (index % 64))
    if value:
        words[index // 64] |= mask
    else:
        words[index // 64] &= ~mask


# ── Test helpers ────────────────────────────────────────────────────

_failures = 0


def _test(name: str):
    print(f"  TEST: {name} ... ", end="")


def _pass():
    print("PASS")


def _fail(msg: str):
    global _failures
    print(f"FAIL: {msg}")
    _failures += 1


def _assert(cond: bool, msg: str):
    if not cond:
        _fail(msg)


# ── Tests ───────────────────────────────────────────────────────────

def test_bits_to_bytes():
    _test("bits_to_bytes basic")
    bits = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    bytes_ = bits_to_bytes_list(bits, 16, 4)
    _assert(len(bytes_) == 2, f"Expected 2 bytes, got {len(bytes_)}")
    _assert(bytes_[0] == 0xFF, f"First byte != 0xFF (got 0x{bytes_[0]:02X})")
    _assert(bytes_[1] == 0x00, f"Second byte != 0x00 (got 0x{bytes_[1]:02X})")
    _pass()


def test_bytes_to_bits():
    _test("bytes_to_bits basic")
    bytes_ = [0xFF, 0x00, 0xAA]
    bits = bytes_to_bits(bytes_, 3, 24)
    _assert(bits[0] == 1, "Bit 0 != 1")
    _assert(bits[7] == 1, "Bit 7 != 1")
    _assert(bits[8] == 0, "Bit 8 != 0")
    _assert(bits[15] == 0, "Bit 15 != 0")
    # 0xAA = 10101010
    _assert(bits[16] == 1, "Bit 16 != 1")
    _assert(bits[17] == 0, "Bit 17 != 0")
    _pass()


def test_roundtrip():
    _test("bits <-> bytes roundtrip")
    orig_bits = [i % 2 for i in range(64)]
    bytes_ = bits_to_bytes_list(orig_bits, 64, 8)
    _assert(len(bytes_) == 8, f"Expected 8 bytes, got {len(bytes_)}")
    decoded = bytes_to_bits(bytes_, 8, 64)
    _assert(decoded == orig_bits, "Roundtrip mismatch")
    _pass()


def test_get_set_bit():
    _test("get_bit / set_bit")
    words = [0, 0]  # two uint64_t words
    set_bit(words, 0, 1)
    set_bit(words, 63, 1)
    set_bit(words, 64, 1)
    set_bit(words, 127, 1)
    _assert(get_bit(words, 0) == 1, "Bit 0 != 1")
    _assert(get_bit(words, 63) == 1, "Bit 63 != 1")
    _assert(get_bit(words, 64) == 1, "Bit 64 != 1")
    _assert(get_bit(words, 127) == 1, "Bit 127 != 1")
    _assert(get_bit(words, 1) == 0, "Bit 1 != 0")
    _assert(get_bit(words, 65) == 0, "Bit 65 != 0")
    _pass()


# ── Main ────────────────────────────────────────────────────────────

def main():
    global _failures
    _failures = 0
    print("=== Bitstream Tests ===\n")
    test_bits_to_bytes()
    test_bytes_to_bits()
    test_roundtrip()
    test_get_set_bit()
    print(f"\n=== Results: {_failures} failures ===")
    return 1 if _failures > 0 else 0


if __name__ == "__main__":
    exit(main())

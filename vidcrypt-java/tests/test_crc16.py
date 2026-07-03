"""Tests for CRC-16-CCITT."""

import pytest


def crc16(data, offset=0, length=None):
    """CRC-16-CCITT (poly 0x1021, init 0xFFFF) matching C/Java."""
    if length is None:
        length = len(data)
    crc = 0xFFFF
    for i in range(offset, offset + length):
        crc ^= (data[i] & 0xFF) << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


class TestCrc16:
    def test_empty(self):
        assert crc16(b'') == 0xFFFF

    def test_single_byte(self):
        assert crc16(b'\x00') == 0xE1F0
        assert crc16(b'\xFF') == 0xFF00

    def test_known_vector(self):
        # "123456789" → 0x29B1 for CRC-16-CCITT
        result = crc16(b'123456789')
        assert result == 0x29B1

    def test_reproducibility(self):
        data = bytes(range(256))
        c1 = crc16(data)
        c2 = crc16(data)
        assert c1 == c2

    def test_offset(self):
        data = bytes([0, 1, 2, 3, 4, 5])
        full = crc16(data)
        partial = crc16(data, 2, 4)
        # partial should not equal full
        assert full != partial

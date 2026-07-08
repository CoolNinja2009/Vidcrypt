"""
CRC-16-CCITT tests — pure Python implementation.

Polynomial: 0x1021 (x^16 + x^12 + x^5 + 1)
Initial value: 0xFFFF
No final XOR (matches C code: crc16.h)
"""


def crc16_ccitt(data: bytes) -> int:
    """Compute CRC-16-CCITT (poly=0x1021, init=0xFFFF)."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc


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

def test_crc16_basic():
    _test("CRC-16-CCITT of empty data")
    _assert(crc16_ccitt(b"") == 0xFFFF, "CRC of empty != 0xFFFF")
    _pass()

    _test("CRC-16-CCITT of known values")
    # "123456789" -> 0x29B1 (CRC-16-CCITT)
    c = crc16_ccitt(b"123456789")
    _assert(c == 0x29B1, f"CRC of '123456789' != 0x29B1 (got 0x{c:04X})")
    _pass()

    _test("CRC-16-CCITT of all zeros")
    c = crc16_ccitt(bytes(16))
    _assert(c != 0, "CRC of zeros shouldn't be 0")
    _pass()


def test_crc16_consistency():
    _test("CRC-16-CCITT consistency")
    data = bytes(range(256))
    c1 = crc16_ccitt(data)
    c2 = crc16_ccitt(data)
    _assert(c1 == c2, "CRC not deterministic")
    _pass()


# ── Main ────────────────────────────────────────────────────────────

def main():
    global _failures
    _failures = 0
    print("=== CRC-16 Tests ===\n")
    test_crc16_basic()
    test_crc16_consistency()
    print(f"\n=== Results: {_failures} failures ===")
    return 1 if _failures > 0 else 0


if __name__ == "__main__":
    exit(main())

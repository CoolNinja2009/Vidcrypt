"""
Reed-Solomon tests — pure Python using the reedsolo library.

GF(256) arithmetic tests are self-contained (0x11D primitive poly).
RS encode/decode tests use the well-tested reedsolo library.
Syndrome verification uses compatible GF(256) tables (same prim=0x11D, fcr=0).
"""

import reedsolo
from typing import List


# ── GF(256) tables (generated lazily, matches reedsolo) ─────────────

_gf_log: List[int] = [0] * 256
_gf_exp: List[int] = [0] * 512
_gf_mul_table: List[List[int]] = [[0] * 256 for _ in range(256)]
_gf_initialized = False


def gf256_init() -> None:
    """Initialize GF(256) tables (primitive poly 0x11D)."""
    global _gf_initialized
    if _gf_initialized:
        return
    val = 1
    for i in range(255):
        _gf_exp[i] = val
        _gf_log[val] = i
        val <<= 1
        if val & 0x100:
            val ^= 0x11D
    for i in range(255, 512):
        _gf_exp[i] = _gf_exp[i - 255]
    _gf_log[0] = 0
    for a in range(256):
        for b in range(256):
            if a == 0 or b == 0:
                _gf_mul_table[a][b] = 0
            else:
                _gf_mul_table[a][b] = _gf_exp[(_gf_log[a] + _gf_log[b]) % 255]
    _gf_initialized = True


def gf_mul(a: int, b: int) -> int:
    return _gf_mul_table[a][b]


def gf_div(a: int, b: int) -> int:
    if a == 0:
        return 0
    if b == 0:
        return 0
    diff = _gf_log[a] - _gf_log[b]
    return _gf_exp[diff if diff >= 0 else diff + 255]


def gf_pow(a: int, power: int) -> int:
    if a == 0:
        return 0
    if power == 0:
        return 1
    log_a = _gf_log[a]
    result = (log_a * power) % 255
    return _gf_exp[result]


def gf_inv(a: int) -> int:
    if a == 0:
        return 0
    return _gf_exp[255 - _gf_log[a]]


# ── Check syndrome using our GF tables (compatible with reedsolo) ───

def check_syndromes_zero(encoded: List[int], ecc: int) -> bool:
    """Verify S_i = 0 for i = 0..ecc-1.

    Uses the reversed convention: coefficient at position j is for x^(254-j),
    which matches both the original C code and the reedsolo library.
    S_i = Σ encoded[j] * α^(i*(254-j))
    """
    for i in range(ecc):
        s = 0
        for j in range(255):
            if encoded[j]:
                s ^= _gf_exp[(_gf_log[encoded[j]] + i * (254 - j)) % 255]
        if s != 0:
            return False
    return True


# ── RS codec wrapper using reedsolo ─────────────────────────────────

_codec_cache: dict = {}


def rs_get_codec(ecc_symbols: int):
    if ecc_symbols not in _codec_cache:
        _codec_cache[ecc_symbols] = reedsolo.RSCodec(ecc_symbols)
    return _codec_cache[ecc_symbols]


def rs_encode(codec, msg: List[int], msg_len: int) -> List[int]:
    """Encode a message with RS ECC. Returns 255-byte codeword padded to 255."""
    result = codec.encode(msg[:msg_len])
    padded = list(result) + [0] * (255 - len(result))
    return padded[:255]


def rs_decode(codec, received: List[int]):
    """Decode a received RS codeword. Returns object with status/decoded/corrected."""
    class Result:
        pass
    result = Result()
    try:
        decoded, remainder, errata_pos = codec.decode(received)
        result.status = 0
        result.corrected = len(errata_pos) if errata_pos else 0
        result.decoded = list(decoded) + [0] * (255 - len(decoded))
    except reedsolo.ReedSolomonError:
        result.status = -1
        result.corrected = 0
        result.decoded = received[:]
    return result


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

def test_gf_arithmetic():
    _test("GF(256) multiply")
    _assert(gf_mul(1, 1) == 1, "1*1 != 1")
    _assert(gf_mul(2, 3) == 6, "2*3 != 6")
    _assert(gf_mul(0, 5) == 0, "0*5 != 0")
    _assert(gf_mul(5, 0) == 0, "5*0 != 0")
    for i in range(1, 255):
        elem = _gf_exp[i]
        inv = gf_inv(elem)
        _assert(gf_mul(elem, inv) == 1, f"elem * inv != 1 for i={i}")
    _assert(gf_mul(2, gf_inv(2)) == 1, "2 * inv(2) != 1")
    _assert(_gf_exp[_gf_log[255]] == 255, "exp[log[255]] != 255")
    inv255 = gf_inv(255)
    _assert(gf_mul(255, inv255) == 1, "255 * inv(255) != 1")
    _assert(gf_mul(inv255, 255) == 1, "inv(255) * 255 != 1")
    _pass()

    _test("GF(256) division")
    _assert(gf_div(10, 2) == 5, "10/2 != 5")
    _assert(gf_div(100, 10) == gf_mul(100, gf_inv(10)), "a/b != a*b^-1")
    _pass()

    _test("GF(256) inverse")
    for i in range(1, 256):
        inv = gf_inv(i)
        _assert(gf_mul(i, inv) == 1, f"a*a^-1 != 1 for a={i}")
    _pass()


def test_rs32_encode_decode():
    """RS(255,223) encode and decode — self-consistency."""
    codec = rs_get_codec(32)
    msg = [((i * 7 + 13) & 0xFF) for i in range(223)]
    encoded = rs_encode(codec, msg, 223)
    _assert(len(encoded) == 255, f"Encoded length != 255 ({len(encoded)})")
    _assert(check_syndromes_zero(encoded, 32), "Syndromes not all zero")
    result = rs_decode(codec, encoded)
    _assert(result.status >= 0, "Decode failed on clean data")
    _assert(result.decoded[:223] == msg, "Decoded data mismatch")
    _pass()


def test_rs32_error_correction():
    """RS(255,223) correct 16 errors."""
    codec = rs_get_codec(32)
    msg = [i & 0xFF for i in range(223)]
    encoded = rs_encode(codec, msg, 223)
    for i in range(16):
        encoded[i * 3] ^= 0xFF
    result = rs_decode(codec, encoded)
    _assert(result.status >= 0, "Failed to correct 16 errors")
    _assert(result.decoded[:223] == msg, "Corrected data mismatch")
    _pass()


def test_rs16():
    """RS(255,239) correct 8 errors."""
    codec = rs_get_codec(16)
    msg = [i & 0xFF for i in range(239)]
    encoded = rs_encode(codec, msg, 239)
    for i in range(8):
        encoded[i * 10] ^= 0xAA
    result = rs_decode(codec, encoded)
    _assert(result.status >= 0, "RS16: Failed to correct 8 errors")
    _assert(result.decoded[:239] == msg, "RS16: Corrected data mismatch")
    _pass()


def test_rs8():
    """RS(255,247) correct 4 errors."""
    codec = rs_get_codec(8)
    msg = [i & 0xFF for i in range(247)]
    encoded = rs_encode(codec, msg, 247)
    for i in range(4):
        encoded[i * 20] ^= 0x55
    result = rs_decode(codec, encoded)
    _assert(result.status >= 0, "RS8: Failed to correct 4 errors")
    _assert(result.decoded[:247] == msg, "RS8: Corrected data mismatch")
    _pass()


def test_rs_uncorrectable():
    """RS(255,223) with too many errors — should not crash."""
    codec = rs_get_codec(32)
    msg = [i & 0xFF for i in range(223)]
    encoded = rs_encode(codec, msg, 223)
    for i in range(50):
        encoded[i] ^= 0xFF
    result = rs_decode(codec, encoded)
    _assert(True, "Uncorrectable test passed (no crash)")
    _pass()


# ── Main ────────────────────────────────────────────────────────────

def main():
    global _failures
    _failures = 0
    print("=== Reed-Solomon Tests ===\n")
    gf256_init()
    test_gf_arithmetic()
    test_rs32_encode_decode()
    test_rs32_error_correction()
    test_rs16()
    test_rs8()
    test_rs_uncorrectable()
    print(f"\n=== Results: {_failures} failures ===")
    return 1 if _failures > 0 else 0


if __name__ == "__main__":
    exit(main())

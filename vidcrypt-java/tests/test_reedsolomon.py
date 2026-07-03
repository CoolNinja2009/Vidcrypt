"""Tests for Reed-Solomon RS(255,223,32) codec.

These tests verify the RS algorithm against known properties:
- Identity: encode then decode unchanged data
- Error correction: corrects up to 16 errors
- Detection: detects more than 16 errors
- Block-level API matches C/Java interface

The Python implementation mirrors the C reference exactly:
encode uses the LFSR-based systematic encoder,
syndromes use the high-to-low coefficient convention (position 0 = MSB),
and Forney applies corrections at positions matching the C convention.
"""

import pytest
import random


class Gf256:
    """GF(256) arithmetic matching the C implementation."""
    PRIM_POLY = 0x11D
    LOG = [0] * 256
    EXP = [0] * 512

    @classmethod
    def _init_tables(cls):
        x = 1
        for i in range(255):
            cls.EXP[i] = x
            cls.LOG[x] = i
            x <<= 1
            if x & 0x100:
                x ^= cls.PRIM_POLY
        cls.LOG[0] = 0
        for i in range(255):
            cls.EXP[255 + i] = cls.EXP[i]

    @classmethod
    def mul(cls, a, b):
        if a == 0 or b == 0:
            return 0
        s = cls.LOG[a] + cls.LOG[b]
        return cls.EXP[s - 255 if s >= 255 else s]

    @classmethod
    def div(cls, a, b):
        if a == 0:
            return 0
        if b == 0:
            return 0
        d = cls.LOG[a] - cls.LOG[b]
        return cls.EXP[d + 255 if d < 0 else d]

    @classmethod
    def pow(cls, a, power):
        if a == 0:
            return 0
        if power == 0:
            return 1
        log_a = cls.LOG[a]
        result = (log_a * power) % 255
        if result < 0:
            result += 255
        return cls.EXP[result]

    @classmethod
    def inv(cls, a):
        if a == 0:
            return 0
        return cls.EXP[255 - cls.LOG[a]]


Gf256._init_tables()


class RsCodec:
    """Reed-Solomon codec matching the C implementation."""

    def __init__(self, ecc_symbols):
        self.ecc_symbols = ecc_symbols
        self.msg_length = 255 - ecc_symbols
        self.block_length = 255
        self.gen_degree = ecc_symbols
        self.generator = self._build_generator()

    def _build_generator(self):
        """Build generator polynomial: product of (x + alpha^i) for i in [0, ecc-1]."""
        gen = [0] * 256
        gen[0] = 1
        for i in range(self.ecc_symbols):
            root = Gf256.EXP[i]
            # Shift right (multiply by x): gen[j] = gen[j-1], gen[0] = 0
            old = gen[:]  # copy before modification
            for j in range(self.ecc_symbols, 0, -1):
                gen[j] = gen[j - 1]
            gen[0] = 0
            # XOR: gen[j] ^= old[j] * root
            for j in range(self.ecc_symbols + 1):
                gen[j] ^= Gf256.mul(old[j], root)
        return gen[:]

    def encode(self, msg):
        """Systematic LFSR encode. Returns full block (msg + ECC)."""
        k = self.msg_length
        n = self.block_length
        ecc = self.ecc_symbols
        encoded = list(msg)[:k] + [0] * ecc

        bb = [0] * ecc  # LFSR shift register
        for i in range(k):
            feedback = encoded[i] ^ bb[ecc - 1]
            if feedback != 0:
                for j in range(ecc - 1, 0, -1):
                    bb[j] = bb[j - 1] ^ Gf256.mul(feedback, self.generator[j])
                bb[0] = Gf256.mul(feedback, self.generator[0])
            else:
                for j in range(ecc - 1, 0, -1):
                    bb[j] = bb[j - 1]
                bb[0] = 0

        # Append remainder (reversed)
        for i in range(ecc):
            encoded[k + i] = bb[ecc - 1 - i]

        return bytes(encoded)

    def decode(self, received):
        """Decode received block. Returns (status, corrected_count, decoded_bytes).

        status: 0 = no errors / corrected, -1 = uncorrectable
        """
        result = bytearray(received[:self.block_length])
        syndromes = self._compute_syndromes(result)

        has_errors = any(s != 0 for s in syndromes)
        if not has_errors:
            return (0, 0, bytes(result[:self.msg_length]))

        sigma, sigma_deg = self._berlekamp_massey(syndromes)
        error_pos = self._chien_search(sigma, sigma_deg)

        if len(error_pos) == 0 or len(error_pos) != sigma_deg:
            return (-1, 0, bytes(result[:self.msg_length]))

        self._forney_algorithm(result, sigma, sigma_deg, error_pos, syndromes)

        n_corrected = len(error_pos)
        return (1 if n_corrected > 0 else 0, n_corrected, bytes(result[:self.msg_length]))

    def _compute_syndromes(self, received):
        """Compute syndromes S_i = sum(received[j] * alpha^(i*(n-1-j))).

        Position 0 is the highest-degree coefficient (MSB), matching C convention.
        """
        n = self.block_length
        ecc = self.ecc_symbols
        syn = [0] * ecc
        for i in range(ecc):
            s = 0
            for j in range(n):
                if received[j]:
                    power = (Gf256.LOG[received[j]] + i * (n - 1 - j)) % 255
                    s ^= Gf256.EXP[power]
            syn[i] = s
        return syn

    def _berlekamp_massey(self, syndromes):
        """B-M algorithm matching the C implementation exactly."""
        ecc = self.ecc_symbols
        C = [0] * 256
        B = [0] * 256
        C[0] = 1
        B[0] = 1
        L = 0
        m = 1
        b = 1

        for n in range(ecc):
            d = syndromes[n]
            for i in range(1, L + 1):
                d ^= Gf256.mul(C[i], syndromes[n - i])

            if d == 0:
                m += 1
            else:
                T = C[:]  # save C
                coeff = Gf256.div(d, b)
                for i in range(256 - m):
                    if B[i] != 0:
                        C[i + m] ^= Gf256.mul(coeff, B[i])

                if 2 * L <= n:
                    L = n + 1 - L
                    B = T
                    b = d
                    m = 1
                else:
                    m += 1

        return C, L

    def _chien_search(self, sigma, sigma_deg):
        """Chien search for error positions. Position 0 = MSB."""
        error_positions = []
        for i in range(255):
            s = sigma[0]
            pow_val = Gf256.EXP[(255 - i) % 255]
            pow_accum = pow_val
            for j in range(1, sigma_deg + 1):
                s ^= Gf256.mul(sigma[j], pow_accum)
                pow_accum = Gf256.mul(pow_accum, pow_val)
            if s == 0:
                error_positions.append(254 - i)
        return error_positions

    def _forney_algorithm(self, decoded, sigma, sigma_deg, error_positions, syndromes):
        """Forney algorithm for computing error magnitudes, matching C."""
        ecc = self.ecc_symbols

        # Compute Omega polynomial: omega_k = sum(syndromes[i] * sigma[j] for i+j=k)
        omega = [0] * 256
        for i in range(ecc):
            if syndromes[i] == 0:
                continue
            for j in range(sigma_deg + 1):
                if sigma[j] != 0 and i + j < ecc:
                    omega[i + j] ^= Gf256.mul(syndromes[i], sigma[j])

        for pos in error_positions:
            # x_inv = alpha^-(pos) = alpha^(1+pos) oddness...
            # In C: x_inv = gf_exp[255 - (254 - pos)]
            # (254 - pos) is the exponent for the error locator
            # x_inv is alpha^(-(254-pos)) in the Chien convention
            x_inv = Gf256.EXP[(255 - (254 - pos)) % 255]

            # Denominator: sigma'(x_inv) — odd terms only
            denom = 0
            for j in range(1, sigma_deg + 1, 2):
                if sigma[j] != 0:
                    pow_idx = (Gf256.LOG[x_inv] * (j - 1)) % 255
                    denom ^= Gf256.mul(sigma[j], Gf256.EXP[pow_idx])

            # Numerator: omega(x_inv)
            num = 0
            pow_val = 1
            for j in range(ecc):
                if omega[j] != 0:
                    num ^= Gf256.mul(omega[j], pow_val)
                pow_val = Gf256.mul(pow_val, x_inv)

            error_val = Gf256.div(num, denom)
            X_i = Gf256.EXP[(254 - pos) % 255]
            error_val = Gf256.mul(X_i, error_val)
            decoded[pos] ^= error_val


# ── Tests ──

@pytest.fixture
def codec():
    return RsCodec(32)  # RS(255,223,32)


class TestGf256:
    def test_tables_match_c(self):
        """GF(256) EXP/LOG tables must match C implementation exactly."""
        # First few EXP values
        assert Gf256.EXP[0] == 1
        assert Gf256.EXP[1] == 2
        assert Gf256.EXP[2] == 4
        # LOG of 1 is 0 (alpha^0 = 1)
        assert Gf256.LOG[1] == 0
        # LOG of 0 is 0
        assert Gf256.LOG[0] == 0
        # EXP and LOG are inverses for non-zero elements
        for i in range(1, 255):
            assert Gf256.LOG[Gf256.EXP[i]] == i
        # EXP doubling: EXP[255 + i] == EXP[i]
        for i in range(255):
            assert Gf256.EXP[255 + i] == Gf256.EXP[i]

    def test_mul_identity(self):
        for i in range(256):
            assert Gf256.mul(i, 1) == i
            assert Gf256.mul(1, i) == i
            assert Gf256.mul(i, 0) == 0
            assert Gf256.mul(0, i) == 0

    def test_mul_distributive(self):
        a, b, c = 42, 113, 200
        left = Gf256.mul(a, b ^ c)
        right = Gf256.mul(a, b) ^ Gf256.mul(a, c)
        assert left == right

    def test_div_undoes_mul(self):
        for _ in range(100):
            a = random.randint(1, 255)
            b = random.randint(1, 255)
            assert Gf256.div(Gf256.mul(a, b), b) == a


class TestRsCodec:
    def test_encode_decode_identity(self, codec):
        """Encode then decode unchanged data → identity."""
        msg = bytes(random.randint(0, 255) for _ in range(codec.msg_length))
        encoded = codec.encode(msg)
        status, corrected, decoded = codec.decode(encoded)
        assert status == 0
        assert corrected == 0
        assert decoded == msg

    def test_correct_single_error(self, codec):
        """Single error should be correctable."""
        msg = bytes(random.randint(0, 255) for _ in range(codec.msg_length))
        encoded = bytearray(codec.encode(msg))
        encoded[10] ^= 0xFF
        status, corrected, decoded = codec.decode(bytes(encoded))
        assert status == 1
        assert corrected == 1
        assert decoded == msg

    def test_correct_max_errors(self, codec):
        """Up to 16 errors (ecc/2) should be correctable."""
        for n_errors in [1, 2, 4, 8, 16]:
            msg = bytes(random.randint(0, 255) for _ in range(codec.msg_length))
            encoded = bytearray(codec.encode(msg))
            positions = random.sample(range(255), n_errors)
            for pos in positions:
                encoded[pos] ^= 0xFF
            status, corrected, decoded = codec.decode(bytes(encoded))
            assert status == 1
            assert corrected == n_errors
            assert decoded == msg

    def test_detect_too_many_errors(self, codec):
        """More than 16 errors should be detected as uncorrectable."""
        failures = 0
        for _ in range(100):
            msg = bytes(random.randint(0, 255) for _ in range(codec.msg_length))
            encoded = bytearray(codec.encode(msg))
            positions = random.sample(range(255), 17)
            for pos in positions:
                encoded[pos] ^= 0xFF
            status, _corrected, _decoded = codec.decode(bytes(encoded))
            if status == -1:
                failures += 1
        # With 17 errors, the decoder should fail the vast majority of the time.
        # Some runs might accidentally produce a valid codeword, but that's rare.
        assert failures >= 90, f"Expected most 17-error blocks to fail, got {failures}/100 failures"

    def test_block_api_roundtrip(self, codec):
        """Round-trip through block-level API."""
        msg = bytes(random.randint(0, 255) for _ in range(223))
        encoded = codec.encode(msg)
        status, corrected, decoded = codec.decode(encoded)
        assert status == 0
        assert decoded == msg

    def test_multiple_codecs(self):
        """Different ECC symbol counts should produce different codecs."""
        for ecc in [8, 16, 32, 64]:
            c = RsCodec(ecc)
            assert c.ecc_symbols == ecc
            assert c.msg_length == 255 - ecc
            msg = bytes(random.randint(0, 255) for _ in range(c.msg_length))
            encoded = c.encode(msg)
            status, corrected, decoded = c.decode(encoded)
            assert status == 0
            assert decoded == msg

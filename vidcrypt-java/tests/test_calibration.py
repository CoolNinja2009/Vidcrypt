"""Tests for calibration parameter encoding/decoding."""

import pytest


CAL_TOP_FRAC = 0.02
CAL_HEIGHT_FRAC = 0.04
CAL_LEFT_FRAC = 0.04
CAL_WIDTH_FRAC = 0.92
CAL_COLS = 48
CAL_ROWS = 4
CAL_BITS_TOTAL = 192
CAL_MAGIC_V1 = 0xCB01
CAL_MAGIC_V2 = 0xCB02
THRESHOLD = 128


def crc16(data, offset=0, length=None):
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


class CalParams:
    """Mimics the Java CalParams."""
    def __init__(self):
        self.frame_width = 1920
        self.frame_height = 1080
        self.margin_x = 96
        self.margin_y = 72
        self.block_size_x = 8
        self.block_size_y = 8
        self.grid_cols = 216
        self.grid_rows = 121
        self.rs_ecc_symbols = 32
        self.rs_data_bytes = 223
        self.header_version = 3
        self.sync_rows = 1
        self.calibration_rows = 4

    @property
    def cal_top(self):
        return int(self.frame_height * CAL_TOP_FRAC)

    @property
    def cal_bottom(self):
        return int(self.frame_height * (CAL_TOP_FRAC + CAL_HEIGHT_FRAC))

    @property
    def grid_top_y(self):
        return self.cal_bottom + self.margin_y

    @property
    def payload_rows(self):
        return self.grid_rows - self.sync_rows

    @property
    def payload_bits_per_frame(self):
        return self.payload_rows * self.grid_cols

    def build_calibration_bytes(self):
        out = bytearray(24)
        out[0] = (CAL_MAGIC_V2 >> 8) & 0xFF
        out[1] = CAL_MAGIC_V2 & 0xFF
        out[2] = (self.frame_width >> 8) & 0xFF
        out[3] = self.frame_width & 0xFF
        out[4] = (self.frame_height >> 8) & 0xFF
        out[5] = self.frame_height & 0xFF
        out[6] = (self.margin_x >> 8) & 0xFF
        out[7] = self.margin_x & 0xFF
        out[8] = (self.margin_y >> 8) & 0xFF
        out[9] = self.margin_y & 0xFF
        out[10] = (self.block_size_x >> 8) & 0xFF
        out[11] = self.block_size_x & 0xFF
        out[12] = (self.block_size_y >> 8) & 0xFF
        out[13] = self.block_size_y & 0xFF
        out[14] = (self.grid_cols >> 8) & 0xFF
        out[15] = self.grid_cols & 0xFF
        out[16] = (self.grid_rows >> 8) & 0xFF
        out[17] = self.grid_rows & 0xFF
        out[18] = self.rs_ecc_symbols & 0xFF
        out[19] = self.header_version & 0xFF
        out[20] = self.sync_rows & 0xFF
        out[21] = 0
        c = crc16(bytes(out), 0, 22)
        out[22] = (c >> 8) & 0xFF
        out[23] = c & 0xFF
        return bytes(out)

    @classmethod
    def parse_calibration_bytes(cls, data):
        magic = ((data[0] << 8) | data[1]) & 0xFFFF
        if magic not in (CAL_MAGIC_V1, CAL_MAGIC_V2):
            return None
        p = CalParams()
        p.frame_width = ((data[2] << 8) | data[3]) & 0xFFFF
        p.frame_height = ((data[4] << 8) | data[5]) & 0xFFFF
        p.margin_x = ((data[6] << 8) | data[7]) & 0xFFFF
        p.margin_y = ((data[8] << 8) | data[9]) & 0xFFFF
        p.block_size_x = ((data[10] << 8) | data[11]) & 0xFFFF
        p.block_size_y = ((data[12] << 8) | data[13]) & 0xFFFF
        p.grid_cols = ((data[14] << 8) | data[15]) & 0xFFFF
        p.grid_rows = ((data[16] << 8) | data[17]) & 0xFFFF
        p.rs_ecc_symbols = data[18] & 0xFF
        p.rs_data_bytes = 255 - p.rs_ecc_symbols
        p.header_version = data[19] & 0xFF
        p.sync_rows = data[20] & 0xFF
        # Verify CRC
        expected_crc = ((data[22] << 8) | data[23]) & 0xFFFF
        computed_crc = crc16(data, 0, 22)
        if expected_crc != computed_crc:
            return None
        return p


class TestCalParams:
    def test_build_parse_roundtrip(self):
        p = CalParams()
        data = p.build_calibration_bytes()
        assert len(data) == 24

        p2 = CalParams.parse_calibration_bytes(data)
        assert p2 is not None
        assert p2.frame_width == p.frame_width
        assert p2.frame_height == p.frame_height
        assert p2.margin_x == p.margin_x
        assert p2.margin_y == p.margin_y
        assert p2.block_size_x == p.block_size_x
        assert p2.block_size_y == p.block_size_y
        assert p2.grid_cols == p.grid_cols
        assert p2.grid_rows == p.grid_rows
        assert p2.rs_ecc_symbols == p.rs_ecc_symbols
        assert p2.header_version == p.header_version

    def test_magic_bytes(self):
        p = CalParams()
        data = p.build_calibration_bytes()
        magic = ((data[0] << 8) | data[1]) & 0xFFFF
        assert magic == CAL_MAGIC_V2

    def test_crc_invalid(self):
        p = CalParams()
        data = bytearray(p.build_calibration_bytes())
        data[0] ^= 0xFF  # Corrupt
        result = CalParams.parse_calibration_bytes(bytes(data))
        assert result is None

    def test_derived_properties(self):
        p = CalParams()
        p.frame_width = 1920
        p.frame_height = 1080
        p.margin_y = 72
        p.block_size_y = 8
        p.grid_rows = 121
        p.sync_rows = 1

        assert p.cal_top == 21     # 1080 * 0.02
        assert p.cal_bottom == 64  # 1080 * 0.06
        assert p.grid_top_y == 136  # 64 + 72
        assert p.payload_rows == 120  # 121 - 1
        assert p.payload_bits_per_frame == 120 * p.grid_cols


class TestCalibrationDots:
    """Test the write/read calibration dots cycle."""

    def write_calibration_dots(self, frame, width, height, params):
        cal_top = params.cal_top
        cal_bottom = params.cal_bottom
        cal_height = cal_bottom - cal_top
        cal_left = int(width * CAL_LEFT_FRAC)
        cal_width = int(width * CAL_WIDTH_FRAC)

        if cal_height < CAL_ROWS or cal_width < CAL_COLS:
            return

        cell_w = cal_width // CAL_COLS
        cell_h = cal_height // CAL_ROWS

        cal_bytes = params.build_calibration_bytes()
        cal_bits = bytearray(CAL_BITS_TOTAL)
        for i in range(CAL_BITS_TOTAL):
            cal_bits[i] = (cal_bytes[i // 8] >> (7 - (i % 8))) & 1

        for r in range(CAL_ROWS):
            for c in range(CAL_COLS):
                bit = cal_bits[r * CAL_COLS + c]
                val = 255 if bit else 0
                y0 = cal_top + r * cell_h
                x0 = cal_left + c * cell_w
                for y in range(y0, min(y0 + cell_h, height)):
                    row_off = y * width
                    for x in range(x0, min(x0 + cell_w, width)):
                        frame[row_off + x] = val

    def read_calibration_dots(self, gray, width, height):
        cal_top = int(height * CAL_TOP_FRAC)
        cal_height = int(height * CAL_HEIGHT_FRAC)
        cal_left = int(width * CAL_LEFT_FRAC)
        cal_width = int(width * CAL_WIDTH_FRAC)

        if cal_height < CAL_ROWS or cal_width < CAL_COLS:
            return None

        cell_w = cal_width // CAL_COLS
        cell_h = cal_height // CAL_ROWS

        bits = bytearray(CAL_BITS_TOTAL)
        for r in range(CAL_ROWS):
            for c in range(CAL_COLS):
                y0 = cal_top + r * cell_h
                x0 = cal_left + c * cell_w
                white = 0
                total = 0
                for y in range(y0, min(y0 + cell_h, height)):
                    row_off = y * width
                    for x in range(x0, min(x0 + cell_w, width)):
                        if gray[row_off + x] >= THRESHOLD:
                            white += 1
                        total += 1
                bits[r * CAL_COLS + c] = 1 if white > total // 2 else 0
        return bytes(bits)

    def test_write_read_roundtrip(self):
        p = CalParams()
        width, height = 1920, 1080
        frame = bytearray(width * height)
        self.write_calibration_dots(frame, width, height, p)

        bits = self.read_calibration_dots(frame, width, height)
        assert bits is not None
        assert len(bits) == CAL_BITS_TOTAL
        assert sum(bits) > 0  # Should have some ones

        # Convert bits back to params
        data = bytearray(24)
        for i in range(CAL_BITS_TOTAL):
            if bits[i]:
                data[i // 8] |= 1 << (7 - (i % 8))
        p2 = CalParams.parse_calibration_bytes(bytes(data))
        assert p2 is not None
        assert p2.frame_width == p.frame_width
        assert p2.frame_height == p.frame_height

    def test_noise_resilience(self):
        """Calibration dots should survive minor noise."""
        import random
        p = CalParams()
        width, height = 1920, 1080
        frame = bytearray(width * height)
        self.write_calibration_dots(frame, width, height, p)

        # Add 10% noise
        for i in range(len(frame)):
            if random.random() < 0.1:
                frame[i] ^= random.randint(0, 30)

        bits = self.read_calibration_dots(frame, width, height)
        assert bits is not None
        data = bytearray(24)
        for i in range(CAL_BITS_TOTAL):
            if bits[i]:
                data[i // 8] |= 1 << (7 - (i % 8))
        p2 = CalParams.parse_calibration_bytes(bytes(data))
        assert p2 is not None

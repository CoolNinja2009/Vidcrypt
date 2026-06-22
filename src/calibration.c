#include "calibration.h"
#include "crc16.h"
#include <string.h>
#include <stdint.h>

void build_calibration_bytes(const CalParams *params, uint8_t out[24]) {
    memset(out, 0, 24);
    out[0] = (uint8_t)(CAL_MAGIC_V2 >> 8);
    out[1] = (uint8_t)(CAL_MAGIC_V2 & 0xFF);
    out[2] = (uint8_t)(params->frame_width >> 8);
    out[3] = (uint8_t)(params->frame_width & 0xFF);
    out[4] = (uint8_t)(params->frame_height >> 8);
    out[5] = (uint8_t)(params->frame_height & 0xFF);
    out[6] = (uint8_t)(params->margin_x >> 8);
    out[7] = (uint8_t)(params->margin_x & 0xFF);
    out[8] = (uint8_t)(params->margin_y >> 8);
    out[9] = (uint8_t)(params->margin_y & 0xFF);
    out[10] = (uint8_t)(params->block_size_x >> 8);
    out[11] = (uint8_t)(params->block_size_x & 0xFF);
    out[12] = (uint8_t)(params->block_size_y >> 8);
    out[13] = (uint8_t)(params->block_size_y & 0xFF);
    out[14] = (uint8_t)(params->grid_cols >> 8);
    out[15] = (uint8_t)(params->grid_cols & 0xFF);
    out[16] = (uint8_t)(params->grid_rows >> 8);
    out[17] = (uint8_t)(params->grid_rows & 0xFF);
    out[18] = params->rs_ecc_symbols;
    out[19] = params->header_version;
    out[20] = params->sync_rows;
    out[21] = 0;
    uint16_t crc = crc16(out, 22);
    out[22] = (uint8_t)(crc >> 8);
    out[23] = (uint8_t)(crc & 0xFF);
}

bool parse_calibration_bytes(const uint8_t data[24], CalParams *params, bool *is_v2) {
    uint16_t magic = (uint16_t)(((uint16_t)data[0] << 8) | data[1]);
    uint8_t buf[24];
    memcpy(buf, data, 24);
    buf[22] = 0; buf[23] = 0;
    uint16_t stored_crc = (uint16_t)(((uint16_t)data[22] << 8) | data[23]);
    if (crc16(buf, 22) != stored_crc) return false;

    memset(params, 0, sizeof(CalParams));

    if (magic == CAL_MAGIC_V1) {
        params->frame_width    = (uint16_t)(((uint16_t)data[2] << 8) | data[3]);
        params->frame_height   = (uint16_t)(((uint16_t)data[4] << 8) | data[5]);
        params->margin_x       = (uint16_t)(((uint16_t)data[6] << 8) | data[7]);
        params->margin_y       = (uint16_t)(((uint16_t)data[8] << 8) | data[9]);
        params->block_size_x   = (uint16_t)(((uint16_t)data[10] << 8) | data[11]);
        params->block_size_y   = (uint16_t)(((uint16_t)data[12] << 8) | data[13]);
        params->grid_cols      = data[14];
        params->grid_rows      = data[15];
        params->rs_ecc_symbols = data[16];
        params->rs_data_bytes  = data[17];
        params->header_version = data[18];
        params->calibration_rows = data[19];
        params->sync_rows      = data[20];
        if (is_v2) *is_v2 = false;
        return true;
    } else if (magic == CAL_MAGIC_V2) {
        params->frame_width    = (uint16_t)(((uint16_t)data[2] << 8) | data[3]);
        params->frame_height   = (uint16_t)(((uint16_t)data[4] << 8) | data[5]);
        params->margin_x       = (uint16_t)(((uint16_t)data[6] << 8) | data[7]);
        params->margin_y       = (uint16_t)(((uint16_t)data[8] << 8) | data[9]);
        params->block_size_x   = (uint16_t)(((uint16_t)data[10] << 8) | data[11]);
        params->block_size_y   = (uint16_t)(((uint16_t)data[12] << 8) | data[13]);
        params->grid_cols      = (uint16_t)(((uint16_t)data[14] << 8) | data[15]);
        params->grid_rows      = (uint16_t)(((uint16_t)data[16] << 8) | data[17]);
        params->rs_ecc_symbols = data[18];
        params->rs_data_bytes  = (uint8_t)(255 - data[18]);
        params->header_version = data[19];
        params->calibration_rows = CAL_ROWS;
        params->sync_rows      = data[20];
        if (is_v2) *is_v2 = true;
        return true;
    }
    return false;
}

void cal_params_legacy_defaults(CalParams *params) {
    memset(params, 0, sizeof(CalParams));
    params->frame_width      = LEGACY_FRAME_WIDTH;
    params->frame_height     = LEGACY_FRAME_HEIGHT;
    params->margin_x         = LEGACY_MARGIN_X;
    params->margin_y         = LEGACY_MARGIN_Y;
    params->block_size_x     = LEGACY_BLOCK_SIZE;
    params->block_size_y     = LEGACY_BLOCK_SIZE;
    params->grid_cols        = LEGACY_GRID_COLS;
    params->grid_rows        = LEGACY_GRID_ROWS;
    params->rs_ecc_symbols   = LEGACY_RS_ECC_SYMBOLS;
    params->rs_data_bytes    = LEGACY_RS_DATA_BYTES;
    params->header_version   = LEGACY_HEADER_VERSION;
    params->calibration_rows = 0;
    params->sync_rows        = 1;
}

void write_calibration_dots(uint8_t *frame, int width, int height,
                            const CalParams *params) {
    (void)params;
    int cal_top    = (int)((float)height * CAL_TOP_FRAC);
    int cal_height = (int)((float)height * CAL_HEIGHT_FRAC);
    int cal_left   = (int)((float)width  * CAL_LEFT_FRAC);
    int cal_width  = (int)((float)width  * CAL_WIDTH_FRAC);

    if (cal_height < CAL_ROWS || cal_width < CAL_COLS) return;

    int cell_w = cal_width  / CAL_COLS;
    int cell_h = cal_height / CAL_ROWS;
    int dot_w  = (cell_w * 7) / 10; if (dot_w < 2) dot_w = 2;
    int dot_h  = (cell_h * 7) / 10; if (dot_h < 2) dot_h = 2;

    uint8_t cal_data[24];
    build_calibration_bytes(params, cal_data);

    uint8_t cal_bits[192];
    for (int i = 0; i < 24; ++i)
        for (int b = 0; b < 8; ++b)
            cal_bits[i * 8 + b] = (uint8_t)((cal_data[i] >> (7 - b)) & 1);

    for (int idx = 0; idx < 192; ++idx) {
        int row = idx / CAL_COLS;
        int col = idx % CAL_COLS;
        int cx = cal_left + col * cell_w + cell_w / 2;
        int cy = cal_top  + row * cell_h + cell_h / 2;
        uint8_t val = cal_bits[idx] ? 255 : 0;

        int x1 = cx - dot_w / 2; if (x1 < 0) x1 = 0;
        int y1 = cy - dot_h / 2; if (y1 < 0) y1 = 0;
        int x2 = x1 + dot_w; if (x2 > width)  x2 = width;
        int y2 = y1 + dot_h; if (y2 > height) y2 = height;

        for (int yy = y1; yy < y2; ++yy)
            for (int xx = x1; xx < x2; ++xx)
                frame[yy * width + xx] = val;
    }
}

int read_calibration_dots(const uint8_t *gray, int width, int height,
                          uint8_t *bits) {
    int cal_top    = (int)((float)height * CAL_TOP_FRAC);
    int cal_height = (int)((float)height * CAL_HEIGHT_FRAC);
    int cal_left   = (int)((float)width  * CAL_LEFT_FRAC);
    int cal_width  = (int)((float)width  * CAL_WIDTH_FRAC);

    if (cal_height < CAL_ROWS || cal_width < CAL_COLS) return 0;

    int cell_w = cal_width  / CAL_COLS;
    int cell_h = cal_height / CAL_ROWS;
    int sample_w = (cell_w * 60) / 100; if (sample_w < 2) sample_w = 2;
    int sample_h = (cell_h * 60) / 100; if (sample_h < 2) sample_h = 2;

    int count = 0;
    for (int row = 0; row < CAL_ROWS; ++row) {
        for (int col = 0; col < CAL_COLS; ++col) {
            int cx = cal_left + col * cell_w + cell_w / 2;
            int cy = cal_top  + row * cell_h + cell_h / 2;
            int sx = cx - sample_w / 2;
            int sy = cy - sample_h / 2;

            int sum = 0, samples = 0;
            for (int yy = sy; yy < sy + sample_h && yy < height; ++yy)
                for (int xx = sx; xx < sx + sample_w && xx < width; ++xx) {
                    sum += (int)gray[yy * width + xx];
                    samples++;
                }
            if (samples == 0) return 0;
            bits[count++] = (uint8_t)((sum / samples) >= THRESHOLD ? 1 : 0);
        }
    }
    return count;
}

bool interpret_calibration_bits(const uint8_t bits[192], CalParams *params) {
    uint8_t data[24];
    memset(data, 0, 24);
    for (int i = 0; i < 192; ++i)
        if (bits[i]) data[i / 8] |= (uint8_t)(1 << (7 - (i % 8)));
    bool is_v2 = false;
    return parse_calibration_bytes(data, params, &is_v2);
}

bool extract_calibration(const uint8_t *gray, int width, int height,
                         CalParams *params) {
    uint8_t bits[192];
    int n = read_calibration_dots(gray, width, height, bits);
    if (n >= 192 && interpret_calibration_bits(bits, params))
        return true;

    /* ── Single-bit error correction (NVDEC tolerance) ──────────────
     * Hardware decoders like NVDEC don't produce bit-exact output.
     * Even with CRF 10 lossy H.264, small pixel differences can flip
     * 1-2 calibration dots (out of 192). The 16-bit CRC rejects the
     * entire calibration if even 1 bit is wrong.
     *
     * Brute-force: try flipping each of the 192 bits and re-check
     * the CRC. This is fast (192 iterations, microsecond-scale) and
     * corrects any single-bit error introduced by NVDEC. */
    for (int flip = 0; flip < 192; ++flip) {
        bits[flip] ^= 1;
        if (interpret_calibration_bits(bits, params))
            return true;
        bits[flip] ^= 1; /* restore */
    }

    cal_params_legacy_defaults(params);
    return false;
}

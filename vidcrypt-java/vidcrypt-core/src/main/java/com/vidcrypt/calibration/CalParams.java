package com.vidcrypt.calibration;

import com.vidcrypt.hash.Crc16;
import java.util.Arrays;

/**
 * Calibration parameters embedded in every video frame.
 * Matches calibration.h CalParams struct and constants exactly.
 */
public final class CalParams {
    // Layout constants
    public static final float CAL_TOP_FRAC     = 0.02f;
    public static final float CAL_HEIGHT_FRAC  = 0.04f;
    public static final float CAL_LEFT_FRAC    = 0.04f;
    public static final float CAL_WIDTH_FRAC   = 0.92f;
    public static final int   CAL_COLS         = 48;
    public static final int   CAL_ROWS         = 4;
    public static final int   CAL_DATA_BYTES   = 24;
    public static final int   CAL_BITS_TOTAL   = CAL_COLS * CAL_ROWS;  // 192
    public static final int   CAL_MAGIC_V1     = 0xCB01;
    public static final int   CAL_MAGIC_V2     = 0xCB02;

    // Legacy defaults
    public static final int LEGACY_FRAME_WIDTH    = 1920;
    public static final int LEGACY_FRAME_HEIGHT   = 1080;
    public static final int LEGACY_MARGIN_X       = 96;
    public static final int LEGACY_MARGIN_Y       = 72;
    public static final int LEGACY_BLOCK_SIZE     = 24;
    public static final int LEGACY_GRID_COLS      = 72;
    public static final int LEGACY_GRID_ROWS      = 39;
    public static final int LEGACY_RS_ECC_SYMBOLS = 32;
    public static final int LEGACY_RS_DATA_BYTES  = 223;
    public static final int LEGACY_HEADER_VERSION = 2;

    // Defaults
    public static final int DEFAULT_FRAME_WIDTH    = 1920;
    public static final int DEFAULT_FRAME_HEIGHT   = 1080;
    public static final int DEFAULT_MARGIN_X       = 96;
    public static final int DEFAULT_MARGIN_Y       = 72;
    public static final int DEFAULT_BLOCK_SIZE     = 8;
    public static final int DEFAULT_RS_ECC_SYMBOLS = 32;
    public static final int DEFAULT_RS_DATA_BYTES  = 223;

    public static final int THRESHOLD = 128;

    // Fields — match CalParams struct exactly
    public int frameWidth;       // uint16_t frame_width
    public int frameHeight;      // uint16_t frame_height
    public int marginX;          // uint16_t margin_x
    public int marginY;          // uint16_t margin_y
    public int blockSizeX;       // uint16_t block_size_x
    public int blockSizeY;       // uint16_t block_size_y
    public int gridCols;         // uint16_t grid_cols
    public int gridRows;         // uint16_t grid_rows
    public int rsEccSymbols;     // uint8_t  rs_ecc_symbols
    public int rsDataBytes;      // uint8_t  rs_data_bytes
    public int headerVersion;    // uint8_t  header_version
    public int calibrationRows;  // uint8_t  calibration_rows
    public int syncRows;         // uint8_t  sync_rows

    public CalParams() {
        setDefaults();
    }

    public void setDefaults() {
        frameWidth      = DEFAULT_FRAME_WIDTH;
        frameHeight     = DEFAULT_FRAME_HEIGHT;
        marginX         = DEFAULT_MARGIN_X;
        marginY         = DEFAULT_MARGIN_Y;
        blockSizeX      = DEFAULT_BLOCK_SIZE;
        blockSizeY      = DEFAULT_BLOCK_SIZE;
        gridCols        = 0;  // computed
        gridRows        = 0;  // computed
        rsEccSymbols    = DEFAULT_RS_ECC_SYMBOLS;
        rsDataBytes     = DEFAULT_RS_DATA_BYTES;
        headerVersion   = 3;
        syncRows        = 1;
        calibrationRows = 4;
    }

    public void setLegacyDefaults() {
        frameWidth      = LEGACY_FRAME_WIDTH;
        frameHeight     = LEGACY_FRAME_HEIGHT;
        marginX         = LEGACY_MARGIN_X;
        marginY         = LEGACY_MARGIN_Y;
        blockSizeX      = LEGACY_BLOCK_SIZE;
        blockSizeY      = LEGACY_BLOCK_SIZE;
        gridCols        = LEGACY_GRID_COLS;
        gridRows        = LEGACY_GRID_ROWS;
        rsEccSymbols    = LEGACY_RS_ECC_SYMBOLS;
        rsDataBytes     = LEGACY_RS_DATA_BYTES;
        headerVersion   = LEGACY_HEADER_VERSION;
        syncRows        = 1;
        calibrationRows = 0;   // C: params->calibration_rows = 0
    }

    // ------------------------------------------------------------------
    // Derived helpers — match static inline functions in calibration.h
    // ------------------------------------------------------------------

    public int calTop() {
        return (int) (frameHeight * CAL_TOP_FRAC);
    }

    public int calBottom() {
        return (int) (frameHeight * (CAL_TOP_FRAC + CAL_HEIGHT_FRAC));
    }

    public int gridTopY() {
        return calBottom() + marginY;
    }

    public int legacyGridTopY() {
        return marginY;
    }

    public int payloadRows() {
        return gridRows - syncRows;
    }

    public int payloadBitsPerFrame() {
        return payloadRows() * gridCols;
    }

    // ------------------------------------------------------------------
    // Serialization / deserialization
    // ------------------------------------------------------------------

    /**
     * Build 24-byte calibration data from params.
     * Matches build_calibration_bytes() exactly.
     */
    public byte[] buildCalibrationBytes() {
        byte[] out = new byte[24];
        out[0] = (byte) (CAL_MAGIC_V2 >> 8);
        out[1] = (byte) (CAL_MAGIC_V2 & 0xFF);
        out[2]  = (byte) (frameWidth >> 8);
        out[3]  = (byte) (frameWidth & 0xFF);
        out[4]  = (byte) (frameHeight >> 8);
        out[5]  = (byte) (frameHeight & 0xFF);
        out[6]  = (byte) (marginX >> 8);
        out[7]  = (byte) (marginX & 0xFF);
        out[8]  = (byte) (marginY >> 8);
        out[9]  = (byte) (marginY & 0xFF);
        out[10] = (byte) (blockSizeX >> 8);
        out[11] = (byte) (blockSizeX & 0xFF);
        out[12] = (byte) (blockSizeY >> 8);
        out[13] = (byte) (blockSizeY & 0xFF);
        out[14] = (byte) (gridCols >> 8);
        out[15] = (byte) (gridCols & 0xFF);
        out[16] = (byte) (gridRows >> 8);
        out[17] = (byte) (gridRows & 0xFF);
        out[18] = (byte) rsEccSymbols;
        out[19] = (byte) headerVersion;
        out[20] = (byte) syncRows;
        out[21] = 0;
        short crc = Crc16.crc16(out, 0, 22);
        out[22] = (byte) (crc >> 8);
        out[23] = (byte) (crc & 0xFF);
        return out;
    }

    /**
     * Parse 24-byte calibration data into params.
     * Matches parse_calibration_bytes() exactly — CRC check first,
     * then separate V1 and V2 field layouts.
     */
    public static ParsedCalibration parseCalibrationBytes(byte[] data) {
        int magic = ((data[0] & 0xFF) << 8) | (data[1] & 0xFF);

        // CRC check over bytes [0..21]  (C: copies to buf, zeros buf[22-23], computes crc16(buf,22))
        short storedCrc = (short) (((data[22] & 0xFF) << 8) | (data[23] & 0xFF));
        short computedCrc = Crc16.crc16(data, 0, 22);
        if (storedCrc != computedCrc) {
            ParsedCalibration p = new ParsedCalibration();
            p.valid = false;
            return p;
        }

        ParsedCalibration p = new ParsedCalibration();
        p.params = new CalParams();

        if (magic == CAL_MAGIC_V1) {
            // V1 layout: grid_cols/rows are single bytes, different offsets
            p.params.frameWidth    = ((data[2] & 0xFF) << 8) | (data[3] & 0xFF);
            p.params.frameHeight   = ((data[4] & 0xFF) << 8) | (data[5] & 0xFF);
            p.params.marginX       = ((data[6] & 0xFF) << 8) | (data[7] & 0xFF);
            p.params.marginY       = ((data[8] & 0xFF) << 8) | (data[9] & 0xFF);
            p.params.blockSizeX    = ((data[10] & 0xFF) << 8) | (data[11] & 0xFF);
            p.params.blockSizeY    = ((data[12] & 0xFF) << 8) | (data[13] & 0xFF);
            p.params.gridCols      = data[14] & 0xFF;
            p.params.gridRows      = data[15] & 0xFF;
            p.params.rsEccSymbols  = data[16] & 0xFF;
            p.params.rsDataBytes   = data[17] & 0xFF;
            p.params.headerVersion = data[18] & 0xFF;
            p.params.calibrationRows = data[19] & 0xFF;
            p.params.syncRows      = data[20] & 0xFF;
            p.isV2 = false;
            p.valid = true;
            return p;
        }

        if (magic == CAL_MAGIC_V2) {
            // V2 layout: grid_cols/rows are 16-bit, rs_data_bytes computed
            p.params.frameWidth    = ((data[2] & 0xFF) << 8) | (data[3] & 0xFF);
            p.params.frameHeight   = ((data[4] & 0xFF) << 8) | (data[5] & 0xFF);
            p.params.marginX       = ((data[6] & 0xFF) << 8) | (data[7] & 0xFF);
            p.params.marginY       = ((data[8] & 0xFF) << 8) | (data[9] & 0xFF);
            p.params.blockSizeX    = ((data[10] & 0xFF) << 8) | (data[11] & 0xFF);
            p.params.blockSizeY    = ((data[12] & 0xFF) << 8) | (data[13] & 0xFF);
            p.params.gridCols      = ((data[14] & 0xFF) << 8) | (data[15] & 0xFF);
            p.params.gridRows      = ((data[16] & 0xFF) << 8) | (data[17] & 0xFF);
            p.params.rsEccSymbols  = data[18] & 0xFF;
            p.params.rsDataBytes   = 255 - p.params.rsEccSymbols;
            p.params.headerVersion = data[19] & 0xFF;
            p.params.syncRows      = data[20] & 0xFF;
            p.params.calibrationRows = CAL_ROWS;  // C: params->calibration_rows = CAL_ROWS
            p.isV2 = true;
            p.valid = true;
            return p;
        }

        // Unknown magic
        p.valid = false;
        return p;
    }

    /**
     * Parse 192-bit calibration bit array into params.
     * Matches interpret_calibration_bits() exactly.
     */
    public static ParsedCalibration interpretCalibrationBits(byte[] bits) {
        // Convert 192 bits to 24 bytes
        byte[] data = new byte[24];
        for (int i = 0; i < 192; i++) {
            if (bits[i] != 0) {
                data[i / 8] |= (byte) (1 << (7 - (i % 8)));
            }
        }
        return parseCalibrationBytes(data);
    }

    // ------------------------------------------------------------------
    // Calibration dot I/O
    // ------------------------------------------------------------------

    /**
     * Write calibration dots into a grayscale frame buffer.
     * Matches write_calibration_dots() exactly — centered dots at 70% cell size.
     */
    public static void writeCalibrationDots(byte[] frame, int width, int height, CalParams params) {
        // C: (void)params — frame dimensions come from the arguments, not params
        int calTop    = (int) ((float) height * CAL_TOP_FRAC);
        int calHeight = (int) ((float) height * CAL_HEIGHT_FRAC);
        int calLeft   = (int) ((float) width  * CAL_LEFT_FRAC);
        int calWidth  = (int) ((float) width  * CAL_WIDTH_FRAC);

        if (calHeight < CAL_ROWS || calWidth < CAL_COLS) return;

        int cellW = calWidth  / CAL_COLS;
        int cellH = calHeight / CAL_ROWS;
        int dotW  = (cellW * 7) / 10;
        if (dotW < 2) dotW = 2;
        int dotH  = (cellH * 7) / 10;
        if (dotH < 2) dotH = 2;

        byte[] calData = params.buildCalibrationBytes();
        byte[] calBits = new byte[192];
        // Unpack 24 bytes → 192 bits
        for (int i = 0; i < 24; i++) {
            for (int b = 0; b < 8; b++) {
                calBits[i * 8 + b] = (byte) ((calData[i] >> (7 - b)) & 1);
            }
        }

        for (int idx = 0; idx < 192; idx++) {
            int row = idx / CAL_COLS;
            int col = idx % CAL_COLS;
            int cx = calLeft + col * cellW + cellW / 2;
            int cy = calTop  + row * cellH + cellH / 2;
            byte val = (byte) (calBits[idx] != 0 ? 255 : 0);

            int x1 = cx - dotW / 2;
            if (x1 < 0) x1 = 0;
            int y1 = cy - dotH / 2;
            if (y1 < 0) y1 = 0;
            int x2 = x1 + dotW;
            if (x2 > width)  x2 = width;
            int y2 = y1 + dotH;
            if (y2 > height) y2 = height;

            for (int yy = y1; yy < y2; yy++) {
                int rowOff = yy * width;
                for (int xx = x1; xx < x2; xx++) {
                    frame[rowOff + xx] = val;
                }
            }
        }
    }

    /**
     * Read calibration dots from a grayscale frame. Returns bit array (192 elements).
     * Matches read_calibration_dots() exactly — centered 60% sample window, average vs THRESHOLD.
     */
    public static byte[] readCalibrationDots(byte[] gray, int width, int height) {
        int calTop    = (int) ((float) height * CAL_TOP_FRAC);
        int calHeight = (int) ((float) height * CAL_HEIGHT_FRAC);
        int calLeft   = (int) ((float) width  * CAL_LEFT_FRAC);
        int calWidth  = (int) ((float) width  * CAL_WIDTH_FRAC);

        if (calHeight < CAL_ROWS || calWidth < CAL_COLS) return new byte[0];

        int cellW    = calWidth  / CAL_COLS;
        int cellH    = calHeight / CAL_ROWS;
        int sampleW  = (cellW * 60) / 100;
        if (sampleW < 2) sampleW = 2;
        int sampleH  = (cellH * 60) / 100;
        if (sampleH < 2) sampleH = 2;

        byte[] bits = new byte[CAL_BITS_TOTAL];
        int count = 0;

        for (int row = 0; row < CAL_ROWS; row++) {
            for (int col = 0; col < CAL_COLS; col++) {
                int cx = calLeft + col * cellW + cellW / 2;
                int cy = calTop  + row * cellH + cellH / 2;
                int sx = cx - sampleW / 2;
                int sy = cy - sampleH / 2;

                int sum = 0, samples = 0;
                for (int yy = sy; yy < sy + sampleH && yy < height; yy++) {
                    int rowOff = yy * width;
                    for (int xx = sx; xx < sx + sampleW && xx < width; xx++) {
                        sum += gray[rowOff + xx] & 0xFF;
                        samples++;
                    }
                }
                if (samples == 0) return new byte[0];
                bits[count++] = (byte) ((sum / samples) >= THRESHOLD ? 1 : 0);
            }
        }
        return bits;
    }

    /**
     * Extract calibration from a grayscale frame.
     * Matches extract_calibration() exactly — includes single-bit error correction
     * for NVDEC tolerance.
     */
    public static ParsedCalibration extractCalibration(byte[] gray, int width, int height) {
        byte[] bits = readCalibrationDots(gray, width, height);
        if (bits.length >= 192) {
            ParsedCalibration p = interpretCalibrationBits(bits);
            if (p.valid) return p;

            // Single-bit error correction (NVDEC tolerance).
            // Hardware decoders like NVDEC don't produce bit-exact output.
            // Brute-force: try flipping each of the 192 bits and re-check CRC.
            for (int flip = 0; flip < 192; flip++) {
                bits[flip] ^= 1;
                p = interpretCalibrationBits(bits);
                if (p.valid) return p;
                bits[flip] ^= 1;  // restore
            }
        }

        // Fallback to legacy defaults
        ParsedCalibration p = new ParsedCalibration();
        p.params = new CalParams();
        p.params.setLegacyDefaults();
        p.valid = false;
        return p;
    }

    // ------------------------------------------------------------------
    // ParsedCalibration result wrapper
    // ------------------------------------------------------------------

    public static class ParsedCalibration {
        public CalParams params;
        public boolean valid;
        public boolean isV2;
    }
}

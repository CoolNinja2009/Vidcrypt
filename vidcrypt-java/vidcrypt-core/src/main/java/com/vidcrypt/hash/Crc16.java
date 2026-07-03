package com.vidcrypt.hash;

/** CRC-16-CCITT (poly 0x1021, init 0xFFFF). */
public final class Crc16 {
    private Crc16() {}

    private static final int POLY = 0x1021;

    /** Compute CRC-16-CCITT of data. */
    public static short crc16(byte[] data, int offset, int len) {
        int crc = 0xFFFF;
        for (int i = offset; i < offset + len; i++) {
            crc ^= (data[i] & 0xFF) << 8;
            for (int j = 0; j < 8; j++) {
                if ((crc & 0x8000) != 0) {
                    crc = (crc << 1) ^ POLY;
                } else {
                    crc <<= 1;
                }
                crc &= 0xFFFF;
            }
        }
        return (short) crc;
    }

    /** Convenience: full array. */
    public static short crc16(byte[] data) {
        return crc16(data, 0, data.length);
    }
}

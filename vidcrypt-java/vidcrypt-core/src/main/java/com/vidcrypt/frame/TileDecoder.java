package com.vidcrypt.frame;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;

/**
 * Tile-level bit extraction from grayscale frames.
 * Matches simd_decode.c — branchless counting, specialized paths for block_size 8 and 16.
 */
public final class TileDecoder {
    private TileDecoder() {}

    static final VectorSpecies<Byte> BYTE_SPEC = ByteVector.SPECIES_PREFERRED;
    static final int THRESHOLD = 128;

    // ── Specialized: block_size = 8, subsample = 4 ──
    //   off = 2, step = 4  ⇒  positions: (2,2) (2,6) (6,2) (6,6)
    private static int countWhiteBlock8(byte[] src, int offset, int stride) {
        return ((src[offset + 2 * stride + 2] & 0xFF) >> 7)
             + ((src[offset + 2 * stride + 6] & 0xFF) >> 7)
             + ((src[offset + 6 * stride + 2] & 0xFF) >> 7)
             + ((src[offset + 6 * stride + 6] & 0xFF) >> 7);
    }

    // ── Specialized: block_size = 16, subsample = 8 ──
    //   off = 4, step = 8  ⇒  positions: (4,4) (4,12) (12,4) (12,12)
    private static int countWhiteBlock16(byte[] src, int offset, int stride) {
        return ((src[offset + 4 * stride + 4] & 0xFF) >> 7)
             + ((src[offset + 4 * stride + 12] & 0xFF) >> 7)
             + ((src[offset + 12 * stride + 4] & 0xFF) >> 7)
             + ((src[offset + 12 * stride + 12] & 0xFF) >> 7);
    }

    /**
     * Count white pixels in a grayscale tile region using subsampling.
     * Branchless: (pixel >> 7) replaces conditional (pixel >= 128).
     */
    public static int tileCountWhite(byte[] src, int srcOffset, int stride,
                                      int tileWidth, int tileHeight, int subsample) {
        int off     = subsample / 2;
        int ncols   = (tileWidth  - 1 - off) / subsample + 1;
        int nrows   = (tileHeight - 1 - off) / subsample + 1;
        int total   = ncols * nrows;
        int half    = total / 2;
        int count   = 0;
        int remaining = total;

        for (int y = off; y < tileHeight; y += subsample) {
            int rowOff = srcOffset + y * stride;
            for (int x = off; x < tileWidth; x += subsample) {
                count += (src[rowOff + x] & 0xFF) >> 7;
                remaining--;
                if (count > half) return count;
                if (count + remaining <= half) return count;
            }
        }
        return count;
    }

    /**
     * Validate sync row: alternating black/white pattern.
     */
    public static boolean validateSync(byte[] gray, int stride,
                                        int syncY, int syncXStart,
                                        int blockSize, int gridCols, int subsample) {
        int baseOffset = syncY * stride;

        if (subsample > 0 && subsample * 2 == blockSize) {
            if (blockSize == 8) {
                for (int col = 0; col < gridCols; col++) {
                    int off = baseOffset + syncXStart + col * 8;
                    int c = countWhiteBlock8(gray, off, stride);
                    if ((c > 2) != ((col & 1) != 0)) return false;
                }
                return true;
            }
            if (blockSize == 16) {
                for (int col = 0; col < gridCols; col++) {
                    int off = baseOffset + syncXStart + col * 16;
                    int c = countWhiteBlock16(gray, off, stride);
                    if ((c > 2) != ((col & 1) != 0)) return false;
                }
                return true;
            }
        }

        // General fallback
        for (int col = 0; col < gridCols; col++) {
            int off = baseOffset + syncXStart + col * blockSize;
            int c = tileCountWhite(gray, off, stride, blockSize, blockSize, subsample);
            if ((c > 2) != ((col & 1) != 0)) return false;
        }
        return true;
    }

    /**
     * Decode grid of tiles into bits.
     * Returns number of bits decoded. Sets syncOk[0] to sync validation result.
     */
    public static int decodeGrid(byte[] gray, int stride,
                                  DecodeGeometry geom,
                                  byte[] bitsOut, int maxBits,
                                  boolean[] syncOk) {
        syncOk[0] = validateSync(gray, stride, geom.syncY, geom.syncXStart,
                                  geom.blockSize, geom.gridCols, geom.subsample);

        int bitIdx = 0;
        int bs = geom.blockSize;
        int ss = geom.subsample;

        if (ss > 0 && ss * 2 == bs) {
            if (bs == 8) {
                for (int row = geom.syncRows; row < geom.gridRows && bitIdx < maxBits; row++) {
                    int yOff = (geom.gridTopY + row * 8) * stride;
                    for (int col = 0; col < geom.gridCols && bitIdx < maxBits; col++) {
                        int off = yOff + geom.gridLeftX + col * 8;
                        bitsOut[bitIdx++] = (byte) (countWhiteBlock8(gray, off, stride) > 2 ? 1 : 0);
                    }
                }
                return bitIdx;
            }
            if (bs == 16) {
                for (int row = geom.syncRows; row < geom.gridRows && bitIdx < maxBits; row++) {
                    int yOff = (geom.gridTopY + row * 16) * stride;
                    for (int col = 0; col < geom.gridCols && bitIdx < maxBits; col++) {
                        int off = yOff + geom.gridLeftX + col * 16;
                        bitsOut[bitIdx++] = (byte) (countWhiteBlock16(gray, off, stride) > 2 ? 1 : 0);
                    }
                }
                return bitIdx;
            }
        }

        // General fallback
        for (int row = geom.syncRows; row < geom.gridRows && bitIdx < maxBits; row++) {
            int yOff = (geom.gridTopY + row * bs) * stride;
            for (int col = 0; col < geom.gridCols && bitIdx < maxBits; col++) {
                int off = yOff + geom.gridLeftX + col * bs;
                int c = tileCountWhite(gray, off, stride, bs, bs, ss);
                bitsOut[bitIdx++] = (byte) (c > 2 ? 1 : 0);
            }
        }
        return bitIdx;
    }

    /**
     * Expand bits into a grayscale frame.
     * Each bit fills a blockSize x blockSize region with 0 or 255.
     */
    public static void expandBits(byte[] srcBits, int bitsOffset,
                                   byte[] frame, int stride,
                                   int blockY, int blockX,
                                   int blockSize, int gridCols, int gridRows) {
        for (int row = 0; row < gridRows; row++) {
            for (int col = 0; col < gridCols; col++) {
                byte val = (byte) (srcBits[bitsOffset + row * gridCols + col] != 0 ? 255 : 0);
                int y0 = blockY + row * blockSize;
                int x0 = blockX + col * blockSize;
                for (int y = y0; y < y0 + blockSize; y++) {
                    int rowOff = y * stride + x0;
                    java.util.Arrays.fill(frame, rowOff, rowOff + blockSize, val);
                }
            }
        }
    }
}

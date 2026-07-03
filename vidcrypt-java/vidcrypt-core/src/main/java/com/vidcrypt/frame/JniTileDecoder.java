package com.vidcrypt.frame;

/**
 * JNI-callable entry point for C code to invoke Java tile decode.
 * Called from tile_decode_jni.c via JNI Invocation API.
 *
 * Signature matches C tile_decode_grid exactly:
 *   (gray, stride, gridTopY, gridLeftX, blockSize, gridCols, gridRows, syncRows, bitsOut, maxBits) → nbits
 * syncOk is an output: syncOk[0] = true/false
 */
public final class JniTileDecoder {
    private JniTileDecoder() {}

    /**
     * Decode a grid of tiles from a grayscale frame into bits.
     * Called from C via JNI. Returns number of bits decoded.
     *
     * @param gray       grayscale frame (width*height bytes, stride = width)
     * @param stride     bytes per row (= width)
     * @param gridTopY   top Y of tile grid
     * @param gridLeftX  left X of tile grid
     * @param blockSize  tile block size (typically 8)
     * @param gridCols   number of tile columns
     * @param gridRows   total grid rows (including sync)
     * @param syncRows   number of sync rows at top
     * @param bitsOut    output bit array (0 or 1 per tile)
     * @param maxBits    size of bitsOut
     * @param syncOk     output: syncOk[0] = sync validation result
     * @return number of bits decoded
     */
    public static int tileDecodeGrid(
            byte[] gray, int stride,
            int gridTopY, int gridLeftX, int blockSize,
            int gridCols, int gridRows, int syncRows,
            byte[] bitsOut, int maxBits,
            boolean[] syncOk) {

        // Build DecodeGeometry on the fly (avoids JNI object overhead)
        DecodeGeometry geom = new DecodeGeometry();
        geom.gridTopY = gridTopY;
        geom.gridLeftX = gridLeftX;
        geom.blockSize = blockSize;
        geom.gridCols = gridCols;
        geom.gridRows = gridRows;
        geom.syncRows = syncRows;
        geom.payRows = gridRows - syncRows;
        geom.payBitsPerFrame = geom.payRows * gridCols;
        geom.rsBlockBits = 0; // not needed for tile decode

        geom.subsample = blockSize / 2;
        if (geom.subsample < 2) geom.subsample = 2;

        geom.payYStart = gridTopY + syncRows * blockSize;
        geom.payYEnd = geom.payYStart + geom.payRows * blockSize;
        geom.payXStart = gridLeftX;
        geom.payXEnd = gridLeftX + gridCols * blockSize;
        geom.syncY = gridTopY;
        geom.syncXStart = gridLeftX;

        return TileDecoder.decodeGrid(gray, stride, geom, bitsOut, maxBits, syncOk);
    }
}

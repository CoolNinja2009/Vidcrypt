package com.vidcrypt.frame;

import org.graalvm.nativeimage.c.CContext;
import org.graalvm.nativeimage.c.function.CEntryPoint;
import org.graalvm.nativeimage.c.type.CCharPointer;
import org.graalvm.nativeimage.c.type.CTypeConversion;
import org.graalvm.word.Pointer;

/**
 * GraalVM Native Image entry point for tile_decode_grid.
 * Compiled to a native shared library (vidcrypt_tiles.dll/.so) via:
 *   native-image --shared -o vidcrypt_tiles -H:Name=vidcrypt_tiles \
 *     -cp vidcrypt-java/out --no-fallback \
 *     com.vidcrypt.frame.NativeTileDecoder
 *
 * The C code calls tile_decode_grid_native() which has zero JVM overhead.
 */
public final class NativeTileDecoder {

    /**
     * C-callable tile decode. Signature matches simd_decode.c tile_decode_grid.
     *
     * @param gray       pointer to grayscale frame (uint8_t*)
     * @param stride     bytes per row
     * @param gridTopY   top Y of tile grid
     * @param gridLeftX  left X of tile grid
     * @param blockSize  tile block size
     * @param gridCols   number of tile columns
     * @param gridRows   total grid rows
     * @param syncRows   number of sync rows
     * @param bitsOut    output bit array (uint8_t*)
     * @param maxBits    size of bitsOut
     * @param syncOk     output: sync validation result (int*, 0 or 1)
     * @return number of bits decoded, or -1 on error
     */
    @CEntryPoint(name = "tile_decode_grid_native")
    public static int tileDecodeGridNative(
            CCharPointer gray, int stride,
            int gridTopY, int gridLeftX, int blockSize,
            int gridCols, int gridRows, int syncRows,
            CCharPointer bitsOut, int maxBits,
            Pointer syncOk) {

        // Convert C pointers to Java arrays
        int frameSize = stride * (gridTopY + gridRows * blockSize + blockSize);
        byte[] grayArray = new byte[frameSize];
        for (int i = 0; i < frameSize; i++) {
            grayArray[i] = gray.read(i);
        }

        byte[] bitsArray = new byte[maxBits];
        boolean[] syncOkArray = new boolean[1];

        // Build geometry and decode
        DecodeGeometry geom = new DecodeGeometry();
        geom.gridTopY = gridTopY;
        geom.gridLeftX = gridLeftX;
        geom.blockSize = blockSize;
        geom.gridCols = gridCols;
        geom.gridRows = gridRows;
        geom.syncRows = syncRows;
        geom.payRows = gridRows - syncRows;
        geom.payBitsPerFrame = geom.payRows * gridCols;
        geom.subsample = blockSize / 2;
        if (geom.subsample < 2) geom.subsample = 2;
        geom.payYStart = gridTopY + syncRows * blockSize;
        geom.payYEnd = geom.payYStart + geom.payRows * blockSize;
        geom.payXStart = gridLeftX;
        geom.payXEnd = gridLeftX + gridCols * blockSize;
        geom.syncY = gridTopY;
        geom.syncXStart = gridLeftX;

        int nbits = TileDecoder.decodeGrid(grayArray, stride, geom, bitsArray, maxBits, syncOkArray);

        // Copy results back
        for (int i = 0; i < nbits; i++) {
            bitsOut.write(i, bitsArray[i]);
        }
        if (syncOk.isNonNull()) {
            syncOk.writeInt(0, syncOkArray[0] ? 1 : 0);
        }

        return nbits;
    }
}

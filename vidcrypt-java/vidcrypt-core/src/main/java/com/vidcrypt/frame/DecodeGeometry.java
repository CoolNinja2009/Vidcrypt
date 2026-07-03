package com.vidcrypt.frame;

import com.vidcrypt.calibration.CalParams;

/**
 * Frame geometry for tile decoding. Matches framedecode.h DecodeGeometry struct, initialized
 * via {@link #fromParams(CalParams, boolean)} which mirrors framedecode.c {@code decode_geometry_init}.
 */
public final class DecodeGeometry {
    public int gridTopY;
    public int gridLeftX;
    public int blockSize;
    public int gridCols;
    public int gridRows;
    public int syncRows;
    public int payRows;
    public int payBitsPerFrame;
    public int rsBlockBits;

    public int payYStart, payYEnd, payXStart, payXEnd;
    public int subsample;
    public int syncY, syncXStart;

    public DecodeGeometry() {}

    /**
     * Initialize from calibration params. Mirrors {@code decode_geometry_init} in framedecode.c.
     *
     * @param params       calibration parameters
     * @param isNewFormat  true for v2 format (grid positioned below calibration bar),
     *                     false for legacy format (grid at margin_y)
     * @return initialized geometry
     */
    public static DecodeGeometry fromParams(CalParams params, boolean isNewFormat) {
        DecodeGeometry g = new DecodeGeometry();
        g.blockSize  = params.blockSizeY;
        g.gridCols   = params.gridCols;
        g.gridRows   = params.gridRows;
        g.syncRows   = params.syncRows;
        g.payRows    = params.payloadRows();
        g.payBitsPerFrame = params.payloadBitsPerFrame();
        g.rsBlockBits = (params.rsDataBytes + params.rsEccSymbols) * 8;

        // subsample = blockSize / 2, minimum 2 (ensures at least 2x2 = 4 samples per tile)
        g.subsample = g.blockSize / 2;
        if (g.subsample < 2) g.subsample = 2;

        if (isNewFormat) {
            g.gridTopY = params.gridTopY();
        } else {
            g.gridTopY = params.legacyGridTopY();
        }
        g.gridLeftX = params.marginX;

        int bs = g.blockSize;
        int payY0 = g.gridTopY + g.syncRows * bs;
        g.payYStart = payY0;
        g.payYEnd   = payY0 + g.payRows * bs;
        g.payXStart = g.gridLeftX;
        g.payXEnd   = g.gridLeftX + g.gridCols * bs;
        g.syncY      = g.gridTopY;
        g.syncXStart = g.gridLeftX;

        return g;
    }
}

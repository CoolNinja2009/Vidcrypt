package com.vidcrypt.frame;

import com.vidcrypt.calibration.CalParams;
import java.util.Arrays;

/**
 * Precomputed frame template with calibration bar + sync row pre-rendered.
 * Payload blocks are inserted via tile expansion. Matches framegen.h/c.
 */
public final class FrameGenerator {
    private final byte[] work;       // grayscale frame buffer (width * height)
    private final int width;
    private final int height;
    private final int stride;
    private final int payYStart, payYEnd, payXStart, payXEnd;
    private final int payRows, gridCols, blockSize;
    private final CalParams params;

    public FrameGenerator(CalParams params) {
        this.params = params;
        this.width  = params.frameWidth;
        this.height = params.frameHeight;
        this.stride = width;
        this.blockSize = params.blockSizeY;

        int gridTopY = params.gridTopY();
        this.payYStart = gridTopY + params.syncRows * blockSize;
        this.payYEnd   = payYStart + params.payloadRows() * blockSize;
        this.payXStart = params.marginX;
        this.payXEnd   = payXStart + params.gridCols * blockSize;
        this.payRows   = params.payloadRows();
        this.gridCols  = params.gridCols;

        this.work = new byte[height * stride];

        // Pre-render calibration bar
        CalParams.writeCalibrationDots(work, width, height, params);

        // Pre-render sync row (alternating black/white tiles)
        for (int col = 0; col < params.gridCols; col++) {
            byte val = (byte) ((col & 1) != 0 ? -1 : 0); // 255 or 0
            int x = params.marginX + col * blockSize;
            for (int y = gridTopY; y < gridTopY + blockSize; y++) {
                Arrays.fill(work, y * stride + x, y * stride + x + blockSize, val);
            }
        }

        // Corner markers (8×8 checkerboard for decoder self-alignment)
        writeCornerMarkers();
    }

    private void writeCornerMarkers() {
        final int SIZE = 8;
        for (int y = 0; y < SIZE; y++) {
            for (int x = 0; x < SIZE; x++) {
                byte val = (byte) (((x + y) & 1) != 0 ? -1 : 0);
                work[y * stride + x] = val;                                          // TL
                work[y * stride + (width - SIZE + x)] = val;                        // TR
                work[(height - SIZE + y) * stride + x] = val;                       // BL
                work[(height - SIZE + y) * stride + (width - SIZE + x)] = val;      // BR
            }
        }
    }

    /**
     * Generate a frame from packed bits.
     * Returns a COPY of the work buffer with payload injected.
     */
    public byte[] generateFrame(byte[] bits) {
        byte[] frame = work.clone();
        TileDecoder.expandBits(bits, 0, frame, stride,
                                payYStart, payXStart, blockSize, gridCols, payRows);
        return frame;
    }

    public int frameSize() { return height * stride; }
    public int width()  { return width; }
    public int height() { return height; }
}

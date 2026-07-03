package com.vidcrypt.encoder;

import com.vidcrypt.calibration.CalParams;
import com.vidcrypt.frame.FrameGenerator;
import com.vidcrypt.hash.Sha256;
import com.vidcrypt.cbridge.NativeRsCodec;
import com.vidcrypt.bitstream.BitStream;
import com.vidcrypt.concurrent.VidcryptExecutors;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.Arrays;

/**
 * Encoder pipeline — uses C native RS encode via JNI.
 */
public final class EncoderPipeline {
    public static final int HEADER_VERSION = 3;
    public static final byte[] HEADER_MAGIC = {'L','S','Y','1'};
    public static final int MAX_FILENAME_LEN = 200;

    private final EncoderConfig config;

    static { NativeRsCodec.ensureLoaded(); }

    public EncoderPipeline(EncoderConfig config) {
        this.config = config;
    }

    public EncoderResult encode(String inputPath, FrameConsumer frameConsumer,
                                 StringBuilder errorMsg) throws IOException {
        long startNs = System.nanoTime();
        EncoderResult result = new EncoderResult();

        Path inPath = Path.of(inputPath);
        if (!Files.exists(inPath)) {
            if (errorMsg != null) errorMsg.append("Input file not found: ").append(inputPath);
            return result;
        }

        long fileSize = Files.size(inPath);
        String fileName = inPath.getFileName().toString();

        CalParams params = computeGridParams(fileSize);
        result.params = params;
        if (params.gridCols == 0 || params.gridRows == 0) {
            if (errorMsg != null) errorMsg.append("Cannot compute valid grid parameters");
            return result;
        }

        byte[] fileHash = Sha256.hashFile(inPath);
        byte[] header = buildHeader(fileName, fileSize, fileHash);
        byte[] fileBytes = Files.readAllBytes(inPath);
        byte[] payload = new byte[header.length + fileBytes.length];
        System.arraycopy(header, 0, payload, 0, header.length);
        System.arraycopy(fileBytes, 0, payload, header.length, fileBytes.length);

        // Pad to RS block boundary
        int rsDataBytes = params.rsDataBytes;
        int payloadLen = payload.length;
        int padLen = (rsDataBytes - (payloadLen % rsDataBytes)) % rsDataBytes;
        if (padLen > 0) payload = Arrays.copyOf(payload, payloadLen + padLen);
        int totalPayloadLen = payload.length;
        int numBlocks = totalPayloadLen / rsDataBytes;

        // Native C RS encode — one JNI call for all blocks
        int blockLen = 255;
        byte[][] encodedBlocks = new byte[numBlocks][blockLen];
        NativeRsCodec.nativeEncodeAll(payload, totalPayloadLen, rsDataBytes,
                                       params.rsEccSymbols, encodedBlocks);

        // Pack encoded blocks into bits (MSB-first per byte)
        int totalBits = numBlocks * blockLen * 8;
        byte[] allBits = new byte[totalBits];
        int bitOff = 0;
        for (int i = 0; i < numBlocks; i++) {
            byte[] block = encodedBlocks[i];
            for (int j = 0; j < blockLen; j++) {
                byte b = block[j];
                for (int k = 0; k < 8; k++)
                    allBits[bitOff + j * 8 + k] = (byte) ((b >> (7 - k)) & 1);
            }
            bitOff += blockLen * 8;
        }

        // Generate frames
        FrameGenerator fg = new FrameGenerator(params);
        int bitsPerFrame = params.payloadBitsPerFrame();
        int totalFrames = (totalBits + bitsPerFrame - 1) / bitsPerFrame;
        byte[] frameBits = new byte[bitsPerFrame];

        for (int frame = 0; frame < totalFrames; frame++) {
            int frameStartBit = frame * bitsPerFrame;
            int bitsThisFrame = Math.min(bitsPerFrame, totalBits - frameStartBit);
            Arrays.fill(frameBits, (byte) 0);
            System.arraycopy(allBits, frameStartBit, frameBits, 0, bitsThisFrame);
            byte[] frameData = fg.generateFrame(frameBits);
            frameConsumer.accept(frame, totalFrames, frameData);
            if (config.progressCallback != null)
                config.progressCallback.onProgress(frame + 1, totalFrames);
        }

        long elapsedNs = System.nanoTime() - startNs;
        result.totalFrames = totalFrames;
        result.totalBytesWritten = totalPayloadLen;
        result.elapsedSec = elapsedNs / 1_000_000_000.0;
        result.fps = totalFrames / result.elapsedSec;
        return result;
    }

    private CalParams computeGridParams(long fileSize) {
        CalParams params = new CalParams();
        if (config.frameWidth > 0)  params.frameWidth  = config.frameWidth;
        if (config.frameHeight > 0) params.frameHeight = config.frameHeight;
        if (config.marginX > 0)     params.marginX     = config.marginX;
        if (config.marginY > 0)     params.marginY     = config.marginY;
        if (config.blockSize > 0)   params.blockSizeX = params.blockSizeY = config.blockSize;
        if (config.rsEccSymbols > 0) params.rsEccSymbols = config.rsEccSymbols;
        params.rsDataBytes = 255 - params.rsEccSymbols;

        int usableW = params.frameWidth  - 2 * params.marginX;
        int usableH = params.frameHeight - params.marginY - params.calBottom();
        int bs = params.blockSizeY;
        params.gridCols = usableW / bs;
        params.gridRows = usableH / bs;
        params.syncRows = 1;
        return params;
    }

    private byte[] buildHeader(String filename, long fileSize, byte[] hash) {
        int fnLen = Math.min(filename.length(), MAX_FILENAME_LEN);
        byte[] fnBytes = filename.substring(0, fnLen).getBytes(StandardCharsets.UTF_8);
        fnLen = Math.min(fnBytes.length, MAX_FILENAME_LEN);
        byte[] h = new byte[4+1+4+4+2+fnLen+32]; int off = 0;
        System.arraycopy(HEADER_MAGIC, 0, h, off, 4); off += 4;
        h[off++] = HEADER_VERSION;
        h[off++] = (byte)(fileSize & 0xFF); h[off++] = (byte)((fileSize>>8)&0xFF);
        h[off++] = (byte)((fileSize>>16)&0xFF); h[off++] = (byte)((fileSize>>24)&0xFF);
        h[off++] = (byte)(fileSize & 0xFF); h[off++] = (byte)((fileSize>>8)&0xFF);
        h[off++] = (byte)((fileSize>>16)&0xFF); h[off++] = (byte)((fileSize>>24)&0xFF);
        h[off++] = (byte)(fnLen & 0xFF); h[off++] = (byte)((fnLen>>8)&0xFF);
        System.arraycopy(fnBytes, 0, h, off, fnLen); off += fnLen;
        System.arraycopy(hash, 0, h, off, 32);
        return h;
    }

    @FunctionalInterface public interface FrameConsumer {
        void accept(int frameIndex, int totalFrames, byte[] frameData);
    }
    @FunctionalInterface public interface ProgressCallback {
        void onProgress(int current, int total);
    }

    public static class EncoderConfig {
        public int frameWidth, frameHeight, marginX, marginY, blockSize, rsEccSymbols;
        public double fps = 25.0;
        public int numWorkers = Runtime.getRuntime().availableProcessors();
        public String codecName = "ffv1", outputPath = "";
        public ProgressCallback progressCallback;
    }
    public static class EncoderResult {
        public int totalFrames;
        public long totalBytesWritten;
        public double elapsedSec, fps;
        public String outputPath;
        public CalParams params;
    }
}

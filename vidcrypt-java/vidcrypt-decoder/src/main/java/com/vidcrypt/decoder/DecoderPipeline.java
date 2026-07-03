package com.vidcrypt.decoder;

import com.vidcrypt.calibration.CalParams;
import com.vidcrypt.frame.DecodeGeometry;
import com.vidcrypt.frame.TileDecoder;
import com.vidcrypt.hash.Sha256;
import com.vidcrypt.cbridge.NativeRsCodec;
import com.vidcrypt.bitstream.BitStream;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.Arrays;

/**
 * Decoder pipeline — uses C native RS decode + bitstream via JNI, Java tile decode.
 */
public final class DecoderPipeline {
    private final DecoderConfig config;

    static { NativeRsCodec.ensureLoaded(); }

    public DecoderPipeline(DecoderConfig config) { this.config = config; }

    public DecoderResult decode(int width, int height, FrameProvider frameProvider) throws IOException {
        long startNs = System.nanoTime();
        DecoderResult result = new DecoderResult();

        // Phase 1: Calibration from first frame
        byte[] frame0 = frameProvider.nextFrame();
        if (frame0 == null) return result;
        result.totalFrames++;

        CalParams.ParsedCalibration parsed = CalParams.extractCalibration(frame0, width, height);
        CalParams params;
        boolean isNewFormat;
        if (parsed != null && parsed.valid) {
            params = parsed.params;
            isNewFormat = parsed.isV2;
        } else {
            params = new CalParams();
            params.setLegacyDefaults();
            isNewFormat = false;
        }
        result.params = params;

        DecodeGeometry geom = DecodeGeometry.fromParams(params, isNewFormat);
        int bitsPerFrame = geom.payBitsPerFrame;
        int stride = width;

        // Phase 2: Decode all frames (Java tile decode)
        byte[][] allFrameBits = new byte[65536][];
        int frameCount = 0;
        byte[] bitsBuf = new byte[bitsPerFrame];
        boolean[] syncOk = new boolean[1];

        int n0 = TileDecoder.decodeGrid(frame0, stride, geom, bitsBuf, bitsPerFrame, syncOk);
        allFrameBits[frameCount++] = Arrays.copyOf(bitsBuf, n0);

        byte[] frame;
        while ((frame = frameProvider.nextFrame()) != null) {
            result.totalFrames++;
            int n = TileDecoder.decodeGrid(frame, stride, geom, bitsBuf, bitsPerFrame, syncOk);
            allFrameBits[frameCount] = Arrays.copyOf(bitsBuf, n);
            frameCount++;
            if (frameCount >= allFrameBits.length)
                allFrameBits = Arrays.copyOf(allFrameBits, allFrameBits.length * 2);
        }

        // Flatten bits
        int totalBits = 0;
        for (int i = 0; i < frameCount; i++) totalBits += allFrameBits[i].length;
        byte[] allBits = new byte[totalBits];
        int off = 0;
        for (int i = 0; i < frameCount; i++) {
            System.arraycopy(allFrameBits[i], 0, allBits, off, allFrameBits[i].length);
            off += allFrameBits[i].length;
        }

        // Phase 3: Bits to bytes — native C
        int totalBytes = totalBits / 8;
        byte[] encodedData = new byte[totalBytes];
        BitStream.bitsToBytes(allBits, totalBits, encodedData, totalBytes);

        // Phase 4: RS decode — native C bulk
        int blockLen = 255;
        int msgLen = params.rsDataBytes;
        int eccSym = params.rsEccSymbols;
        int numBlocks = totalBytes / blockLen;
        if (numBlocks == 0) numBlocks = Math.max(1, totalBytes / msgLen);

        byte[][] decodedBlocks = new byte[numBlocks][msgLen];
        int[] statuses = new int[numBlocks];

        NativeRsCodec.nativeDecodeAll(encodedData, totalBytes, blockLen, msgLen,
                                       eccSym, decodedBlocks, statuses);

        for (int s : statuses) if (s == -1) result.rsFailures++;

        // Phase 5: Reassemble
        byte[] reassembled = new byte[numBlocks * msgLen];
        for (int i = 0; i < numBlocks; i++)
            System.arraycopy(decodedBlocks[i], 0, reassembled, i * msgLen, msgLen);

        // Phase 6: Parse header
        ParsedHeader hdr = parseHeader(reassembled);
        if (hdr == null) return result;

        int headerLen = hdr.headerLength;
        byte[] fileData = Arrays.copyOfRange(reassembled, headerLen,
            (int)Math.min(headerLen + hdr.originalSize, reassembled.length));

        // SHA-256 verify
        byte[] actualHash = Sha256.hashData(fileData);
        result.checksumMatch = Arrays.equals(actualHash, hdr.fileHash);

        Path outDir = Path.of(config.outputDir);
        if (!Files.exists(outDir)) Files.createDirectories(outDir);
        Path outPath = outDir.resolve(hdr.filename);
        Files.write(outPath, fileData);

        result.outputPath = outPath.toString();
        result.originalFilename = hdr.filename;
        result.totalBytesWritten = fileData.length;

        long elapsedNs = System.nanoTime() - startNs;
        result.elapsedSec = elapsedNs / 1_000_000_000.0;
        result.fps = result.totalFrames / result.elapsedSec;
        return result;
    }

    private ParsedHeader parseHeader(byte[] data) {
        if (data.length < 4) return null;
        if (data[0]!='L'||data[1]!='S'||data[2]!='Y'||data[3]!='1') return null;
        int off = 4;
        off += 1; // version
        off += 4; // payload_size
        long origSize = (data[off]&0xFF)|((data[off+1]&0xFF)<<8)|((data[off+2]&0xFF)<<16)|((data[off+3]&0xFF)<<24);
        off += 4;
        int fnLen = (data[off]&0xFF)|((data[off+1]&0xFF)<<8); off += 2;
        byte[] fnBytes = Arrays.copyOfRange(data, off, off+fnLen); off += fnLen;
        String filename = new String(fnBytes, StandardCharsets.UTF_8);
        byte[] hash = Arrays.copyOfRange(data, off, off+32);
        ParsedHeader h = new ParsedHeader();
        h.filename = filename; h.originalSize = origSize; h.fileHash = hash; h.headerLength = off+32;
        return h;
    }

    private static class ParsedHeader { String filename; long originalSize; byte[] fileHash; int headerLength; }

    @FunctionalInterface public interface FrameProvider { byte[] nextFrame(); }
    @FunctionalInterface public interface ProgressCallback { void onProgress(int cur, int tot); }

    public static class DecoderConfig {
        public int numWorkers = Runtime.getRuntime().availableProcessors();
        public String outputDir = ".";
        public ProgressCallback progressCallback;
    }
    public static class DecoderResult {
        public int totalFrames, rsFailures;
        public long totalBytesWritten;
        public double elapsedSec, fps;
        public boolean checksumMatch;
        public String outputPath, originalFilename;
        public CalParams params;
    }
}

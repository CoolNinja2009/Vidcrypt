package com.vidcrypt.cli;

import com.vidcrypt.encoder.EncoderPipeline;
import com.vidcrypt.decoder.DecoderPipeline;

import java.io.*;
import java.nio.file.*;
import java.util.concurrent.Callable;

/**
 * CLI entry point for Vidcrypt Java rewrite.
 * Usage:
 *   encode: vidcrypt encode -i input.zip -o encoded.mkv [-c ffv1|h264]
 *   decode: vidcrypt decode -i encoded.mkv [-b cpu|gpu|auto] [-o output_dir]
 *
 * The FFmpeg pipe (video encode/decode) is handled by forking the C FFmpeg
 * process. This CLI focuses on the pure-computation Java path.
 */
public final class VidcryptMain {
    public static void main(String[] args) {
        if (args.length == 0) {
            printUsage();
            System.exit(1);
        }

        String command = args[0];
        String[] cmdArgs = new String[args.length - 1];
        System.arraycopy(args, 1, cmdArgs, 0, cmdArgs.length);

        try {
            switch (command) {
                case "encode" -> encode(cmdArgs);
                case "decode" -> decode(cmdArgs);
                case "bench"  -> bench(cmdArgs);
                case "help", "--help", "-h" -> printUsage();
                default -> {
                    System.err.println("Unknown command: " + command);
                    printUsage();
                    System.exit(1);
                }
            }
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static void encode(String[] args) throws IOException {
        String inputPath = null;
        String outputPath = "encoded.mkv";
        String codec = "ffv1";
        int blockSize = 0;
        int eccSymbols = 0;
        int workers = Runtime.getRuntime().availableProcessors();
        int width = 0, height = 0;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-i" -> inputPath = args[++i];
                case "-o" -> outputPath = args[++i];
                case "-c" -> codec = args[++i];
                case "-b", "--block-size" -> blockSize = Integer.parseInt(args[++i]);
                case "--ecc" -> eccSymbols = Integer.parseInt(args[++i]);
                case "-j", "--workers" -> workers = Integer.parseInt(args[++i]);
                case "-W", "--width" -> width = Integer.parseInt(args[++i]);
                case "-H", "--height" -> height = Integer.parseInt(args[++i]);
                default -> {
                    System.err.println("Unknown option: " + args[i]);
                    System.exit(1);
                }
            }
        }

        if (inputPath == null) {
            System.err.println("Missing required -i <input>");
            System.exit(1);
        }

        EncoderPipeline.EncoderConfig config = new EncoderPipeline.EncoderConfig();
        config.outputPath = outputPath;
        config.codecName = codec;
        config.numWorkers = workers;
        if (blockSize > 0) config.blockSize = blockSize;
        if (eccSymbols > 0) config.rsEccSymbols = eccSymbols;
        if (width > 0) config.frameWidth = width;
        if (height > 0) config.frameHeight = height;
        config.progressCallback = (current, total) -> {
            int pct = (int)((long)current * 100 / total);
            System.out.printf("\rEncoding: %d/%d frames (%d%%)", current, total, pct);
        };

        System.out.println("Vidcrypt Encoder (Java)");
        System.out.println("  Input:  " + inputPath);
        System.out.println("  Output: " + outputPath);
        System.out.println("  Codec:  " + codec);

        EncoderPipeline pipeline = new EncoderPipeline(config);

        // Write frames to raw grayscale file as proof-of-concept
        // In production, pipe to FFmpeg process
        try (FileOutputStream fos = new FileOutputStream(outputPath + ".raw")) {
            EncoderPipeline.EncoderResult result = pipeline.encode(
                inputPath,
                (frameIdx, totalFrames, frameData) -> {
                    try {
                        fos.write(frameData);
                    } catch (IOException e) {
                        throw new UncheckedIOException(e);
                    }
                },
                null  // errorMsg
            );

            System.out.println();
            System.out.println("Done: " + result.totalFrames + " frames, "
                + String.format("%.1f", result.elapsedSec) + "s, "
                + String.format("%.0f", result.fps) + " fps");
        }
    }

    private static void decode(String[] args) throws IOException {
        String inputPath = null;
        String outputDir = ".";
        int workers = Runtime.getRuntime().availableProcessors();

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-i" -> inputPath = args[++i];
                case "-o" -> outputDir = args[++i];
                case "-j", "--workers" -> workers = Integer.parseInt(args[++i]);
                default -> {
                    System.err.println("Unknown option: " + args[i]);
                    System.exit(1);
                }
            }
        }

        if (inputPath == null) {
            System.err.println("Missing required -i <input>");
            System.exit(1);
        }

        DecoderPipeline.DecoderConfig config = new DecoderPipeline.DecoderConfig();
        config.outputDir = outputDir;
        config.numWorkers = workers;

        System.out.println("Vidcrypt Decoder (Java)");
        System.out.println("  Input:  " + inputPath);
        System.out.println("  Output: " + outputDir);

        // Read raw grayscale frames from file (proof-of-concept)
        // In production, read from FFmpeg pipe
        byte[] rawData = Files.readAllBytes(Path.of(inputPath));

        // Assume default 1920x1080 frame size
        int width = 1920;
        int height = 1080;
        int frameSize = width * height;
        int totalFrames = rawData.length / frameSize;

        DecoderPipeline pipeline = new DecoderPipeline(config);
        DecoderPipeline.DecoderResult result = pipeline.decode(
            width, height,
            new DecoderPipeline.FrameProvider() {
                int frameIdx = 0;

                @Override
                public byte[] nextFrame() {
                    if (frameIdx >= totalFrames) return null;
                    int off = frameIdx * frameSize;
                    frameIdx++;
                    byte[] frame = new byte[frameSize];
                    System.arraycopy(rawData, off, frame, 0, frameSize);
                    return frame;
                }
            }
        );

        System.out.println("Done: " + result.totalFrames + " frames, "
            + String.format("%.1f", result.elapsedSec) + "s, "
            + String.format("%.0f", result.fps) + " fps");
        System.out.println("  File:     " + result.originalFilename);
        System.out.println("  Written:  " + result.totalBytesWritten + " bytes");
        System.out.println("  Checksum: " + (result.checksumMatch ? "MATCH" : "MISMATCH"));
        System.out.println("  RS fails: " + result.rsFailures);
    }

    private static void bench(String[] args) {
        System.out.println("Vidcrypt Benchmark (C native RS + bitstream, Java tiles)");
        System.out.println("========================================================");

        com.vidcrypt.cbridge.NativeRsCodec.ensureLoaded();
        

        // RS encode benchmark — native C bulk
        int ecc = 32, msgLen = 223, nBlocks = 100000;
        byte[] payload = new byte[nBlocks * msgLen];
        for (int i = 0; i < payload.length; i++) payload[i] = (byte) i;

        long start = System.nanoTime();
        byte[][] encBlocks = new byte[nBlocks][255];
        com.vidcrypt.cbridge.NativeRsCodec.nativeEncodeAll(payload, payload.length, msgLen, ecc, encBlocks);
        long encodeNs = System.nanoTime() - start;

        System.out.printf("RS encode:  %d blocks in %.2f ms (%.0f ops/s)%n",
            nBlocks, encodeNs / 1_000_000.0, nBlocks / (encodeNs / 1_000_000_000.0));

        // RS decode benchmark — native C bulk
        byte[] encData = new byte[nBlocks * 255];
        for (int i = 0; i < nBlocks; i++)
            System.arraycopy(encBlocks[i], 0, encData, i * 255, 255);

        start = System.nanoTime();
        byte[][] decBlocks = new byte[nBlocks][msgLen];
        int[] statuses = new int[nBlocks];
        com.vidcrypt.cbridge.NativeRsCodec.nativeDecodeAll(encData, encData.length, 255, msgLen, ecc, decBlocks, statuses);
        long decodeNs = System.nanoTime() - start;

        int failures = 0;
        for (int s : statuses) if (s != 0) failures++;
        System.out.printf("RS decode:  %d blocks in %.2f ms (%.0f ops/s, %d failures)%n",
            nBlocks, decodeNs / 1_000_000.0, nBlocks / (decodeNs / 1_000_000_000.0), failures);

        // Bitstream benchmark — native C
        byte[] bits = new byte[100_000_000];
        for (int i = 0; i < bits.length; i++) bits[i] = (byte) (i & 1);
        long[] words = new long[(bits.length + 63) / 64];

        start = System.nanoTime();
        com.vidcrypt.bitstream.BitStream.bitsToWords(bits, bits.length, words, words.length);
        long bitNs = System.nanoTime() - start;
        System.out.printf("Bit pack:   %d bits in %.2f ms (%.0f Mbps)%n",
            bits.length, bitNs / 1_000_000.0, bits.length / (bitNs / 1_000_000_000.0) / 1_000_000);

        // Tile decode benchmark — pure Java
        int width = 1920, height = 1080;
        byte[] frame = new byte[width * height];
        for (int i = 0; i < frame.length; i++) frame[i] = (byte) (i & 0xFF);

        com.vidcrypt.frame.DecodeGeometry geom =
            com.vidcrypt.frame.DecodeGeometry.fromParams(
                new com.vidcrypt.calibration.CalParams(), true);
        byte[] bitsOut = new byte[100_000];
        boolean[] syncOk = new boolean[1];

        start = System.nanoTime();
        for (int i = 0; i < 1000; i++)
            com.vidcrypt.frame.TileDecoder.decodeGrid(frame, width, geom, bitsOut, bitsOut.length, syncOk);
        long tileNs = System.nanoTime() - start;
        System.out.printf("Tile decode: 1000 frames in %.2f ms (%.0f FPS)%n",
            tileNs / 1_000_000.0, 1000.0 / (tileNs / 1_000_000_000.0));
    }

    private static void printUsage() {
        System.out.println("""
            Vidcrypt Java — File-to-Video Encoder/Decoder
                        
            Usage:
              vidcrypt encode -i <input> [-o <output>] [-c <codec>] [options]
              vidcrypt decode -i <input> [-o <output_dir>] [options]
              vidcrypt bench
                        
            Encode options:
              -i <path>          Input file to encode
              -o <path>          Output video path (default: encoded.mkv)
              -c <codec>         Video codec: ffv1, h264 (default: ffv1)
              -b, --block-size N Tile block size (default: 8)
              --ecc N            RS ECC symbols (default: 32)
              -j, --workers N    Worker threads (default: CPU cores)
              -W, --width N      Frame width (default: 1920)
              -H, --height N     Frame height (default: 1080)
                        
            Decode options:
              -i <path>          Input video or raw frames to decode
              -o <dir>           Output directory (default: .)
              -j, --workers N    Worker threads (default: CPU cores)
                        
            Bench:
              Runs micro-benchmarks for RS, bitstream, and tile decode.
            """);
    }
}

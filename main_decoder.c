#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "decoder.h"
#include "backend.h"
#include "logutil.h"

static void print_usage(void) {
    printf("vidcrypt-decoder v4 — Decode a video back to the original file\n");
    printf("\n");
    printf("Usage:\n");
    printf("  vidcrypt-decoder -i <video> [options]\n");
    printf("\n");
    printf("Options:\n");
    printf("  -i, --input <path>       Input video file (required)\n");
    printf("  -o, --output-dir <dir>   Output directory (default: current dir)\n");
    printf("  -j, --workers <n>        Number of worker threads (default: 4)\n");
    printf("  -b, --backend <mode>     Backend: cpu, gpu (default: cpu)\n");
    printf("  -h, --help               Show this help\n");
}

int main(int argc, char **argv) {
    DecoderConfig config;
    decoder_config_defaults(&config);

    const char *input_path = NULL;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) {
            if (++i >= argc) { fprintf(stderr, "Missing argument for %s\n", argv[i-1]); return 1; }
            input_path = argv[i];
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output-dir") == 0) {
            if (++i >= argc) return 1;
            snprintf(config.output_dir, sizeof(config.output_dir), "%s", argv[i]);
        } else if (strcmp(argv[i], "-j") == 0 || strcmp(argv[i], "--workers") == 0) {
            if (++i >= argc) return 1;
            config.num_workers = atoi(argv[i]);
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--backend") == 0) {
            if (++i >= argc) { fprintf(stderr, "Missing argument for %s\n", argv[i-1]); return 1; }
            config.backend_mode = backend_mode_from_string(argv[i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage();
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage();
            return 1;
        }
    }

    if (!input_path) {
        fprintf(stderr, "Error: -i (input) is required.\n");
        print_usage();
        return 1;
    }

    log_init();

    LOG_SEPARATOR("VIDCRYPT DECODE SESSION");
    LOG_INFO("Input: %s", input_path);
    LOG_INFO("Backend: %s", backend_mode_name(config.backend_mode));
    LOG_INFO("Workers: %d", config.num_workers);
    LOG_METRIC("config.backend", (double)config.backend_mode);
    if (config.output_dir[0])
        LOG_INFO("Output dir: %s", config.output_dir);

    DecoderResult result;
    char error_msg[512];
    error_msg[0] = '\0';

    printf("Decoding: %s\n", input_path);
    if (config.output_dir[0])
        printf("Output dir: %s\n", config.output_dir);

    LOG_INFO("Starting decode pipeline...");

    if (!decoder_decode_file(input_path, &config, &result, error_msg, sizeof(error_msg))) {
        fprintf(stderr, "Error: %s\n", error_msg);
        if (result.total_frames > 0) {
            printf("\nPartial results:\n");
            printf("  Original filename: %s\n", result.original_filename);
            printf("  Frames decoded: %d\n", result.total_frames);
            printf("  Bytes written: %lld\n", (long long)result.total_bytes_written);
            printf("  RS failures: %d\n", result.rs_failures);
            printf("  Checksum: %s\n", result.checksum_match ? "MATCH" : "MISMATCH");
        }
        LOG_ERROR("Decode failed: %s", error_msg);
        LOG_METRIC("decode.failed", 1.0);
        log_close();
        return 1;
    }

    printf("\n=== DECODE COMPLETE ===\n");
    printf("Output: %s\n", result.output_path);
    printf("Original filename: %s\n", result.original_filename);
    printf("Frames: %d\n", result.total_frames);
    printf("Bytes: %lld\n", (long long)result.total_bytes_written);
    printf("Time: %.2f seconds\n", result.elapsed_sec);
    printf("Throughput: %.1f fps\n", result.fps);
    printf("RS failures: %d\n", result.rs_failures);
    printf("Checksum: %s\n", result.checksum_match ? "MATCH" : "MISMATCH");

    LOG_SEPARATOR("DECODE COMPLETE");
    LOG_INFO("Output: %s", result.output_path);
    LOG_INFO("Filename: %s", result.original_filename);
    LOG_METRIC("frames.total",   (double)result.total_frames);
    LOG_METRIC("bytes.written",  (double)result.total_bytes_written);
    LOG_METRIC("elapsed.sec",    result.elapsed_sec);
    LOG_METRIC("throughput.fps", result.fps);
    LOG_METRIC("rs.failures",    (double)result.rs_failures);
    LOG_INFO("Checksum: %s", result.checksum_match ? "MATCH" : "MISMATCH");
    LOG_METRIC("checksum.match", result.checksum_match ? 1.0 : 0.0);

    log_close();

    return result.checksum_match ? 0 : 1;
}

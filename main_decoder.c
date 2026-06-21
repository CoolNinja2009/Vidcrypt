#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "decoder.h"

static void print_usage(void) {
    printf("vidcrypt-decoder v4 — Decode a video back to the original file (CPU, gray8)\n");
    printf("\n");
    printf("Usage:\n");
    printf("  vidcrypt-decoder -i <video> [options]\n");
    printf("\n");
    printf("Options:\n");
    printf("  -i, --input <path>       Input video file (required)\n");
    printf("  -o, --output-dir <dir>   Output directory (default: current dir)\n");
    printf("  -j, --workers <n>        Number of worker threads (default: 4)\n");
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

    DecoderResult result;
    char error_msg[512];
    error_msg[0] = '\0';

    printf("Decoding: %s\n", input_path);
    if (config.output_dir[0])
        printf("Output dir: %s\n", config.output_dir);

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

    return result.checksum_match ? 0 : 1;
}

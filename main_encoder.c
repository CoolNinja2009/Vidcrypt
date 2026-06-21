#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "encoder.h"

static void print_usage(void) {
    printf("vidcrypt-encoder v4 — Encode a file to video (CPU, gray8)\n");
    printf("\n");
    printf("Usage:\n");
    printf("  vidcrypt-encoder -i <input> -o <output> [options]\n");
    printf("\n");
    printf("Options:\n");
    printf("  -i, --input <path>       Input file (required)\n");
    printf("  -o, --output <path>      Output video file (required)\n");
    printf("  -W, --width <px>         Frame width  (default: %d)\n", DEFAULT_FRAME_WIDTH);
    printf("  -H, --height <px>        Frame height (default: %d)\n", DEFAULT_FRAME_HEIGHT);
    printf("  -m, --margin <px>        Margin (default: %d)\n", DEFAULT_MARGIN_X);
    printf("  -b, --block-size <px>    Block size   (default: %d)\n", DEFAULT_BLOCK_SIZE);
    printf("  -r, --rs <n>             RS ECC symbols: 32, 16, 8, 0 (default: %d)\n", DEFAULT_RS_ECC_SYMBOLS);
    printf("  -f, --fps <n>            Frames per second (default: 30.0)\n");
    printf("  -j, --workers <n>        Number of worker threads (default: 4)\n");
    printf("  -c, --codec <name>       FFmpeg codec name (default: ffv1)\n");
    printf("  -h, --help               Show this help\n");
}

int main(int argc, char **argv) {
    EncoderConfig config;
    encoder_config_defaults(&config);

    const char *input_path = NULL;
    const char *output_path = NULL;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) {
            if (++i >= argc) { fprintf(stderr, "Missing argument for %s\n", argv[i-1]); return 1; }
            input_path = argv[i];
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (++i >= argc) { fprintf(stderr, "Missing argument for %s\n", argv[i-1]); return 1; }
            output_path = argv[i];
        } else if (strcmp(argv[i], "-W") == 0 || strcmp(argv[i], "--width") == 0) {
            if (++i >= argc) return 1;
            config.frame_width = atoi(argv[i]);
        } else if (strcmp(argv[i], "-H") == 0 || strcmp(argv[i], "--height") == 0) {
            if (++i >= argc) return 1;
            config.frame_height = atoi(argv[i]);
        } else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--margin") == 0) {
            if (++i >= argc) return 1;
            config.margin_x = config.margin_y = atoi(argv[i]);
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--block-size") == 0) {
            if (++i >= argc) return 1;
            config.block_size = atoi(argv[i]);
        } else if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--rs") == 0) {
            if (++i >= argc) return 1;
            config.rs_ecc_symbols = atoi(argv[i]);
        } else if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--fps") == 0) {
            if (++i >= argc) return 1;
            config.fps = atof(argv[i]);
        } else if (strcmp(argv[i], "-j") == 0 || strcmp(argv[i], "--workers") == 0) {
            if (++i >= argc) return 1;
            config.num_workers = atoi(argv[i]);
        } else if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--codec") == 0) {
            if (++i >= argc) return 1;
            config.codec_name = argv[i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage();
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage();
            return 1;
        }
    }

    if (!input_path || !output_path) {
        fprintf(stderr, "Error: -i (input) and -o (output) are required.\n");
        print_usage();
        return 1;
    }

    snprintf(config.output_path, sizeof(config.output_path), "%s", output_path);

    EncoderResult result;
    char error_msg[256];
    error_msg[0] = '\0';

    printf("Encoding: %s -> %s\n", input_path, output_path);
    printf("Backend: cpu  Codec: %s\n", config.codec_name ? config.codec_name : "ffv1");
    printf("Resolution: %dx%d  Block: %dpx  RS: %d  FPS: %.1f  Workers: %d\n",
           config.frame_width, config.frame_height,
           config.block_size, config.rs_ecc_symbols, config.fps, config.num_workers);

    if (!encoder_encode_file(input_path, &config, &result, error_msg, sizeof(error_msg))) {
        fprintf(stderr, "Error: %s\n", error_msg);
        return 1;
    }

    printf("\n=== ENCODE COMPLETE ===\n");
    printf("Output: %s\n", result.output_path);
    printf("Frames: %d\n", result.total_frames);
    printf("Payload: %lld bytes\n", (long long)result.total_bytes_written);
    printf("Time: %.2f seconds\n", result.elapsed_sec);
    printf("Throughput: %.1f fps\n", result.fps);

    return 0;
}

#ifndef VIDCRYPT_ENCODER_H
#define VIDCRYPT_ENCODER_H

#include <stdint.h>
#include <stdbool.h>
#include "calibration.h"
#include "framegen.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct EncoderConfig {
    int  frame_width;
    int  frame_height;
    int  margin_x;
    int  margin_y;
    int  block_size;
    int  rs_ecc_symbols;
    double fps;
    int  num_workers;

    const char *codec_name;

    void (*progress_callback)(int64_t written, int64_t total, void *user_data);
    void *progress_user_data;

    char output_path[1024];
} EncoderConfig;

#define HEADER_VERSION      3
#define HEADER_MAGIC        "LSY1"
#define MAX_FILENAME_LEN    200
#define SHA256_DIGEST_LEN   32

typedef struct EncoderResult {
    int   total_frames;
    int64_t total_bytes_written;
    double elapsed_sec;
    double fps;
    char  output_path[2048];
    CalParams params;
} EncoderResult;

uint8_t *encoder_build_header(const char *input_path, const CalParams *params,
                              size_t *out_len);

bool encoder_encode_file(const char *input_path, const EncoderConfig *config,
                         EncoderResult *result, char *error_msg, int error_msg_size);

void encoder_config_defaults(EncoderConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* VIDCRYPT_ENCODER_H */

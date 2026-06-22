#ifndef VIDCRYPT_DECODER_H
#define VIDCRYPT_DECODER_H

#include <stdint.h>
#include <stdbool.h>
#include "calibration.h"
#include "framedecode.h"
#include "backend.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DecoderConfig {
    int  num_workers;
    BackendMode backend_mode;  /* BACKEND_CPU (default) or BACKEND_GPU */

    void (*progress_callback)(int64_t written, int64_t total, void *user_data);
    void *progress_user_data;

    char output_dir[1024];
} DecoderConfig;

typedef struct DecoderResult {
    int      total_frames;
    int64_t  total_bytes_written;
    double   elapsed_sec;
    double   fps;
    int      rs_failures;
    bool     checksum_match;
    char     output_path[2048];
    char     original_filename[256];
    CalParams params;
} DecoderResult;

void decoder_config_defaults(DecoderConfig *config);

bool decoder_parse_header(const uint8_t *payload_bits, int nbits,
                          const CalParams *params,
                          char *filename, int filename_size,
                          int64_t *orig_size,
                          uint8_t expected_checksum[32],
                          char *error_msg, int error_msg_size);

bool decoder_decode_file(const char *input_path, const DecoderConfig *config,
                         DecoderResult *result, char *error_msg, int error_msg_size);

#ifdef __cplusplus
}
#endif

#endif /* VIDCRYPT_DECODER_H */

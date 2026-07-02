#ifndef VIDCRYPT_FRAMEDECODE_H
#define VIDCRYPT_FRAMEDECODE_H

#include <stdint.h>
#include <stdbool.h>
#include "calibration.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int grid_top_y;
    int grid_left_x;
    int block_size;
    int grid_cols;
    int grid_rows;
    int sync_rows;
    int pay_rows;
    int pay_bits_per_frame;
    int rs_block_bits;

    int pay_y_start, pay_y_end, pay_x_start, pay_x_end;
    int subsample;
    int sync_y, sync_x_start;
} DecodeGeometry;

void decode_geometry_init(DecodeGeometry *geom, const CalParams *params, bool is_new_format);

bool decode_sync_check(const DecodeGeometry *geom, const uint8_t *gray, int stride);

int decode_payload_tiles(const DecodeGeometry *geom, const uint8_t *gray, int stride,
                         uint8_t *bits, int max_bits);

/* ── Frame offset detection for self-alignment ────────────────────── */

typedef struct {
    int dx;            /* horizontal pixel offset (positive = content shifted right) */
    int dy;            /* vertical pixel offset (positive = content shifted down) */
    int confidence;    /* 0-100: confidence in the detected offset */
} FrameOffset;

/* Detect frame alignment offset from corner markers.
 * Scans ±2 pixels around each expected 8×8 checkerboard marker position.
 * Returns detected offset, or {0,0,0} if markers not found (new videos
 * without markers, or non-vidcrypt content). */
FrameOffset detect_frame_offset(const uint8_t *gray, int width, int height, int stride);

/* Decode bits from a grayscale frame using precomputed geometry.
 * gray: gray8 frame data (stride = width, 1 byte per pixel). */
void decode_frame_bits(const uint8_t *gray, int width, int height, int stride,
                       const DecodeGeometry *geom,
                       bool *sync_ok, uint8_t *bits, int *nbits);

#ifdef __cplusplus
}
#endif

#endif /* VIDCRYPT_FRAMEDECODE_H */

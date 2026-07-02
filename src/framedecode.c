#include "framedecode.h"
#include "simd_decode.h"
#include <string.h>
#include <stdlib.h>

void decode_geometry_init(DecodeGeometry *geom, const CalParams *params, bool is_new_format) {
    memset(geom, 0, sizeof(DecodeGeometry));
    geom->block_size = (int)params->block_size_y;
    geom->grid_cols  = (int)params->grid_cols;
    geom->grid_rows  = (int)params->grid_rows;
    geom->sync_rows  = (int)params->sync_rows;
    geom->pay_rows   = cal_params_payload_rows(params);
    geom->pay_bits_per_frame = cal_params_payload_bits_per_frame(params);
    geom->rs_block_bits = ((int)params->rs_data_bytes + (int)params->rs_ecc_symbols) * 8;

    /* Sample a grid of pixels within each tile for majority-vote bit decoding.
     * subsample = block_size/2: for block_size=8, every 4th row/col = 4 samples.
     * Need >50% (3/4) to flip a bit — robust for lossless gray pipeline.
     * Minimum subsample of 2 ensures at least 2x2 = 4 samples per tile. */
    geom->subsample = geom->block_size / 2;
    if (geom->subsample < 2) geom->subsample = 2;

    if (is_new_format)
        geom->grid_top_y = cal_params_grid_top_y(params);
    else
        geom->grid_top_y = cal_params_legacy_grid_top_y(params);
    geom->grid_left_x = (int)params->margin_x;

    int bs = geom->block_size;
    int pay_y0 = geom->grid_top_y + geom->sync_rows * bs;
    geom->pay_y_start = pay_y0;
    geom->pay_y_end   = pay_y0 + geom->pay_rows * bs;
    geom->pay_x_start = geom->grid_left_x;
    geom->pay_x_end   = geom->grid_left_x + geom->grid_cols * bs;
    geom->sync_y       = geom->grid_top_y;
    geom->sync_x_start = geom->grid_left_x;
}

bool decode_sync_check(const DecodeGeometry *geom, const uint8_t *gray, int stride) {
    return tile_validate_sync(gray, stride,
                              geom->sync_y, geom->sync_x_start,
                              geom->block_size, geom->grid_cols,
                              geom->subsample);
}

int decode_payload_tiles(const DecodeGeometry *geom, const uint8_t *gray, int stride,
                         uint8_t *bits, int max_bits) {
    bool sync_ok = true;
    return tile_decode_grid(gray, stride,
                            geom->grid_top_y, geom->grid_left_x,
                            geom->block_size,
                            geom->grid_cols, geom->grid_rows, geom->sync_rows,
                            geom->subsample,
                            bits, max_bits, &sync_ok);
}

/* ──────────────────────────────────────────────────────────────────────
 * Corner marker offset detection — self-alignment for the decoder
 * ──────────────────────────────────────────────────────────────────────
 *
 * The encoder writes an 8×8 checkerboard at each frame corner.
 * These markers survive video compression and let the decoder detect
 * any pixel-level shift in the frame (from ffmpeg padding, alignment,
 * or re-encoding). The decoder scans a ±2 pixel window around each
 * expected marker position and picks the offset with the best match.
 *
 * Marker pattern (8×8, B=0, W=255):
 *   B W B W B W B W
 *   W B W B W B W B
 *   ...repeating...
 * ────────────────────────────────────────────────────────────────────── */

#define CORNER_SIZE 8
#define SEARCH_RANGE 2

/* Reference checkerboard pattern (0=black, 1=white) */
static const uint8_t corner_ref[CORNER_SIZE][CORNER_SIZE] = {
    {0,1,0,1,0,1,0,1},
    {1,0,1,0,1,0,1,0},
    {0,1,0,1,0,1,0,1},
    {1,0,1,0,1,0,1,0},
    {0,1,0,1,0,1,0,1},
    {1,0,1,0,1,0,1,0},
    {0,1,0,1,0,1,0,1},
    {1,0,1,0,1,0,1,0},
};

/* Score an 8×8 region against the checkerboard pattern. Returns matches (0-64). */
static int score_corner_region(const uint8_t *gray, int stride, int x0, int y0) {
    int matches = 0;
    for (int y = 0; y < CORNER_SIZE; ++y) {
        for (int x = 0; x < CORNER_SIZE; ++x) {
            int pixel_white = (gray[(size_t)(y0 + y) * stride + (size_t)(x0 + x)] >= 128) ? 1 : 0;
            if (pixel_white == corner_ref[y][x])
                matches++;
        }
    }
    return matches;
}

FrameOffset detect_frame_offset(const uint8_t *gray, int width, int height, int stride) {
    FrameOffset result = {0, 0, 0};

    /* Expected corner positions (top-left of each 8×8 marker) */
    struct { int x, y; } corners[4] = {
        {0, 0},                              /* top-left */
        {width - CORNER_SIZE, 0},            /* top-right */
        {0, height - CORNER_SIZE},           /* bottom-left */
        {width - CORNER_SIZE, height - CORNER_SIZE}  /* bottom-right */
    };

    int best_total = 0, best_dx = 0, best_dy = 0;

    /* Score the (0,0) offset first */
    int zero_total = 0;
    for (int c = 0; c < 4; ++c)
        zero_total += score_corner_region(gray, stride, corners[c].x, corners[c].y);
    best_total = zero_total;

    /* Scan ±SEARCH_RANGE around each corner */
    for (int dy = -SEARCH_RANGE; dy <= SEARCH_RANGE; ++dy) {
        for (int dx = -SEARCH_RANGE; dx <= SEARCH_RANGE; ++dx) {
            if (dx == 0 && dy == 0) continue; /* already scored */
            int total = 0;
            int valid = 1;
            for (int c = 0; c < 4; ++c) {
                int sx = corners[c].x + dx;
                int sy = corners[c].y + dy;
                if (sx < 0 || sy < 0 ||
                    sx + CORNER_SIZE > width || sy + CORNER_SIZE > height) {
                    valid = 0;
                    break;
                }
                total += score_corner_region(gray, stride, sx, sy);
            }
            if (valid && total > best_total) {
                best_total = total;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }

    int max_possible = 4 * 64; /* 4 corners × 64 pixels each */
    result.confidence = best_total * 100 / max_possible;

    /* Only trust the offset if confidence > 70% (i.e., >45/64 per corner on avg).
     * An all-black margin scores ~50% (matches half the checkerboard by chance).
     * A real marker scores ~100%. 70% is a safe threshold. */
    if (result.confidence >= 70) {
        result.dx = best_dx;
        result.dy = best_dy;
    } else {
        result.dx = 0;
        result.dy = 0;
        result.confidence = 0;
    }

    return result;
}

void decode_frame_bits(const uint8_t *gray, int width, int height, int stride,
                       const DecodeGeometry *geom,
                       bool *sync_ok, uint8_t *bits, int *nbits) {
    (void)width;
    (void)height;

    int pay_bits = geom->pay_rows * geom->grid_cols;
    if (*nbits < pay_bits) {
        *nbits = 0;
        *sync_ok = false;
        return;
    }

    *sync_ok = decode_sync_check(geom, gray, stride);
    *nbits = decode_payload_tiles(geom, gray, stride, bits, *nbits);
}

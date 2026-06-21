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

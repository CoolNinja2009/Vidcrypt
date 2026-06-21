#include "simd_decode.h"
#include "calibration.h"
#include <string.h>
#include <stdlib.h>

static int scalar_count_white(const uint8_t *src, int stride,
                               int tile_width, int tile_height,
                               int subsample) {
    int count = 0;
    /* Start sampling from the center of each tile (offset = subsample/2)
     * rather than the top-left corner (offset = 0). This avoids sampling
     * tile boundary pixels which are blurred by scaling/compression.
     * For subsample=4, samples at offsets 2,6 instead of 0,4. */
    int off = subsample / 2;
    for (int y = off; y < tile_height; y += subsample)
        for (int x = off; x < tile_width; x += subsample)
            if (src[y * stride + x] >= THRESHOLD) ++count;
    return count;
}

int tile_count_white(const uint8_t *src, int stride,
                     int tile_width, int tile_height,
                     int subsample) {
    return scalar_count_white(src, stride, tile_width, tile_height, subsample);
}

bool tile_validate_sync(const uint8_t *gray, int stride,
                        int sync_y, int sync_x_start,
                        int block_size, int grid_cols,
                        int subsample) {
    for (int col = 0; col < grid_cols; ++col) {
        int x = sync_x_start + col * block_size;
        int count = scalar_count_white(gray + sync_y * stride + x, stride,
                                        block_size, block_size, subsample);
        int total = (block_size / subsample) * (block_size / subsample);
        int bit = (count > total / 2) ? 1 : 0;
        if (bit != (col % 2)) return false;
    }
    return true;
}

int tile_decode_grid(const uint8_t *gray, int stride,
                     int grid_top_y, int grid_left_x,
                     int block_size,
                     int grid_cols, int grid_rows, int sync_rows,
                     int subsample,
                     uint8_t *bits_out, int max_bits,
                     bool *sync_ok) {
    int pay_rows = grid_rows - sync_rows;
    int total_bits = pay_rows * grid_cols;
    if (total_bits > max_bits) total_bits = max_bits;

    *sync_ok = tile_validate_sync(gray, stride, grid_top_y, grid_left_x,
                                   block_size, grid_cols, subsample);

    int bit_idx = 0;
    int sub = subsample;
    int total_samples = (block_size / sub) * (block_size / sub);

    for (int row = sync_rows; row < grid_rows && bit_idx < total_bits; ++row) {
        int y = grid_top_y + row * block_size;
        for (int col = 0; col < grid_cols && bit_idx < total_bits; ++col) {
            int x = grid_left_x + col * block_size;
            int count = scalar_count_white(gray + y * stride + x, stride,
                                            block_size, block_size, sub);
            bits_out[bit_idx++] = (uint8_t)((count > total_samples / 2) ? 1 : 0);
        }
    }
    return bit_idx;
}

int tile_read_calibration(const uint8_t *gray, int stride,
                          int width, int height,
                          uint8_t *bits, int max_bits) {
    int cal_top    = (int)((float)height * CAL_TOP_FRAC);
    int cal_height = (int)((float)height * CAL_HEIGHT_FRAC);
    int cal_left   = (int)((float)width  * CAL_LEFT_FRAC);
    int cal_width  = (int)((float)width  * CAL_WIDTH_FRAC);

    if (cal_height < CAL_ROWS || cal_width < CAL_COLS) return 0;

    int cell_w = cal_width  / CAL_COLS;
    int cell_h = cal_height / CAL_ROWS;
    int sample_w = (cell_w * 60) / 100; if (sample_w < 2) sample_w = 2;
    int sample_h = (cell_h * 60) / 100; if (sample_h < 2) sample_h = 2;

    int count = 0;
    for (int row = 0; row < CAL_ROWS && count < max_bits; ++row)
        for (int col = 0; col < CAL_COLS && count < max_bits; ++col) {
            int cx = cal_left + col * cell_w + cell_w / 2;
            int cy = cal_top  + row * cell_h + cell_h / 2;
            int sx = cx - sample_w / 2;
            int sy = cy - sample_h / 2;
            int sum = 0, samples = 0;
            for (int yy = sy; yy < sy + sample_h && yy < height; ++yy)
                for (int xx = sx; xx < sx + sample_w && xx < width; ++xx) {
                    sum += (int)gray[yy * stride + xx];
                    samples++;
                }
            if (samples == 0) return 0;
            bits[count++] = (uint8_t)((sum / samples) >= THRESHOLD ? 1 : 0);
        }
    return count;
}

void tile_expand_bits(const uint8_t *src_bits,
                      uint8_t *frame, int stride,
                      int block_y, int block_x,
                      int block_size,
                      int grid_cols, int grid_rows) {
    for (int row = 0; row < grid_rows; ++row) {
        int y = block_y + row * block_size;
        for (int col = 0; col < grid_cols; ++col) {
            int bit_idx = row * grid_cols + col;
            uint8_t val = src_bits[bit_idx] ? 255 : 0;
            int x = block_x + col * block_size;
            for (int yy = y; yy < y + block_size; ++yy)
                memset(frame + yy * stride + x, val, (size_t)block_size);
        }
    }
}

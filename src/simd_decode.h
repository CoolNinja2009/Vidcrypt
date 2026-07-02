#ifndef VIDCRYPT_SIMD_DECODE_H
#define VIDCRYPT_SIMD_DECODE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Count white pixels in a grayscale tile region using subsampling.
 * src: grayscale pixel data (stride bytes per row).
 * pixel value >= THRESHOLD (128) = white.
 * subsample: 1 = every pixel, 2 = every other row/col.
 * Returns count of white (above threshold) pixels sampled. */
int tile_count_white(const uint8_t *src, int stride,
                     int tile_width, int tile_height,
                     int subsample);

/* Validate sync row in grayscale frame. Returns true if alternating pattern matches. */
bool tile_validate_sync(const uint8_t *gray, int stride,
                        int sync_y, int sync_x_start,
                        int block_size, int grid_cols,
                        int subsample);

/* Decode grid of tiles into bits. Returns number of bits decoded. */
int tile_decode_grid(const uint8_t *gray, int stride,
                     int grid_top_y, int grid_left_x,
                     int block_size,
                     int grid_cols, int grid_rows, int sync_rows,
                     int subsample,
                     uint8_t *bits_out, int max_bits,
                     bool *sync_ok);

/* Read calibration dots from grayscale frame. Returns number of bits read. */
int tile_read_calibration(const uint8_t *gray, int stride,
                          int width, int height,
                          uint8_t *bits, int max_bits);

/* Expand bits into a grayscale frame (encoder).
 * Each bit fills a block_size x block_size region with 0 or 255.
 * frame: grayscale buffer (stride bytes per row = width). */
void tile_expand_bits(const uint8_t *src_bits,
                      uint8_t *frame, int stride,
                      int block_y, int block_x,
                      int block_size,
                      int grid_cols, int grid_rows);

#ifdef __cplusplus
}
#endif

#endif /* VIDCRYPT_SIMD_DECODE_H */

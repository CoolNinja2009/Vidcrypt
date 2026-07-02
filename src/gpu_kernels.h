#ifndef VIDCRYPT_GPU_KERNELS_H
#define VIDCRYPT_GPU_KERNELS_H

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Calibration bar layout constants ──────────────────────────────
 * Used by both the GPU render_template kernel and the CPU fallback
 * (backend.c / gpu_backend.c) for computing grid layout geometry.
 * These must match the values in render_template_impl. */

#define CAL_COLS         48
#define CAL_ROWS          4
#define CAL_BITS_TOTAL   (CAL_COLS * CAL_ROWS)  /* 192 */
#define CAL_TOP_FRAC     0.02f
#define CAL_HEIGHT_FRAC  0.04f
#define CAL_LEFT_FRAC    0.04f
#define CAL_WIDTH_FRAC   0.92f

/* Luma threshold for bit decision (must match calibration.h) */
#define THRESHOLD         128

/* ─── nv12_to_gray ───────────────────────────────────────────────────
 * Copy NV12 Y-plane to planar grayscale, handling pitch conversion.
 * Both src and dst are GPU device pointers.
 * Launched on the given CUDA stream. */
void nv12_to_gray_kernel(dim3 grid, dim3 block,
                          const uint8_t *src, int src_pitch,
                          uint8_t *dst, int dst_pitch,
                          int width, int height,
                          cudaStream_t stream);

/* ─── frame_generate ─────────────────────────────────────────────────
 * Fill payload region of a BGR frame with black/white blocks from bits.
 * Frame must be pre-cleared (calibration + sync row rendered) before call.
 * Launched on the given CUDA stream. */
void frame_generate_kernel(dim3 grid, dim3 block,
                            uint8_t *frame, int stride,
                            const uint8_t *bits,
                            int pay_y, int pay_x,
                            int block_size, int grid_cols, int payload_rows,
                            cudaStream_t stream);

/* ─── bgr24_to_bgra32 ────────────────────────────────────────────────
 * Convert packed BGR24 to BGRA32 (alpha=0xFF) for NVENC input.
 * Both src and dst are GPU device pointers.
 * dst_stride should be width * 4 (tightly packed BGRA32). */
void bgr24_to_bgra32_kernel(dim3 grid, dim3 block,
                              const uint8_t *src, int src_stride,
                              uint8_t *dst, int dst_stride,
                              int width, int height,
                              cudaStream_t stream);

/* ─── bgr24_to_gray ───────────────────────────────────────────────────
 * Extract G channel from packed BGR24 to create planar grayscale buffer.
 * Both src and dst are GPU device pointers.
 * src_stride = width * 3 (packed BGR24), dst_stride = width (grayscale).
 * Launched on the given CUDA stream. */
void bgr24_to_gray_kernel(dim3 grid, dim3 block,
                           const uint8_t *src, int src_stride,
                           uint8_t *dst, int dst_stride,
                           int width, int height,
                           cudaStream_t stream);

/* ─── render_template ─────────────────────────────────────────────────
 * Pre-render calibration dots + sync row into a frame buffer.
 * 'cal_data' points to 6 uint32 values (192 bits = 24 bytes) of
 * calibration data built by build_calibration_bytes().
 * Called once per video (not per frame). */
void render_template_kernel(dim3 grid, dim3 block,
                             uint8_t *frame, int stride,
                             int width, int height,
                             int grid_cols, int block_size,
                             int margin_x, int margin_y,
                             const uint32_t *cal_data,
                             cudaStream_t stream);

/* ─── read_calibration_dots ───────────────────────────────────────────
 * Read calibration dots from a grayscale frame and pack 192 bits into
 * 6 uint32 values (24 bytes).  The output can be parsed by
 * parse_calibration_bytes() on the CPU side.
 *
 * 'gray' is a device pointer to a grayscale frame (width * height bytes).
 * 'gray_stride' = width (tightly packed grayscale).
 * 'cal_data_out' is a device pointer to 24 bytes (6 uint32) of output.
 * The output buffer MUST be zero-initialized before calling this kernel.
 * Launched on the given CUDA stream. */
void read_calibration_dots_kernel(dim3 grid, dim3 block,
                                   const uint8_t *gray, int gray_stride,
                                   int width, int height,
                                   uint32_t *cal_data_out,
                                   cudaStream_t stream);

/* ─── extract_bits ────────────────────────────────────────────────────
 * GPU-side payload tile bit extraction.
 * Decodes grid_cols × pay_rows tiles from a grayscale frame in CUDA
 * memory, writing only the compact bit array to output.
 *
 * 'gray' is a device pointer to planar grayscale (width × height bytes).
 * 'gray_stride' = width (tightly packed).
 * 'grid_top_y', 'grid_left_x': starting pixel of the grid.
 * 'block_size': pixel size of each tile.
 * 'grid_cols', 'pay_rows': grid dimensions.
 * 'subsample': pixel subsampling factor (2 = every other pixel).
 * 'sync_rows': number of sync rows to skip (typically 1).
 * 'bits_out': device pointer to output array (at least grid_cols×pay_rows bytes).
 *
 * Launched on the given CUDA stream.
 * Grid should be (grid_cols, pay_rows) with block (1, 1). */
void extract_bits_kernel(dim3 grid, dim3 block,
                          const uint8_t *gray, int gray_stride,
                          int grid_top_y, int grid_left_x,
                          int block_size,
                          int grid_cols, int pay_rows,
                          int subsample,
                          int sync_rows,
                          uint8_t *bits_out,
                          cudaStream_t stream);

/* ─── extract_calibration_bits ────────────────────────────────────────
 * GPU-side calibration dot extraction.
 * Reads 192 calibration dots from a grayscale frame, writing one byte
 * per bit (0 or 1) to the output array.
 *
 * 'bits_out' must be at least 192 bytes (device memory).
 * Launched with 1 block, 192 threads. */
void extract_calibration_bits_kernel(
    const uint8_t *gray, int gray_stride,
    int width, int height,
    uint8_t *bits_out,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* VIDCRYPT_GPU_KERNELS_H */

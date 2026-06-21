#include <cuda_runtime.h>
#include <stdint.h>
#include "gpu_kernels.h"

/* ─── nv12_to_gray_impl ──────────────────────────────────────────────
 * Copies NV12 Y-plane to planar grayscale with pitch conversion.
 * NV12 Y-data IS grayscale — this kernel handles stride fixup only. */

__global__ void nv12_to_gray_impl(
    const uint8_t* __restrict__ src, int src_pitch,
    uint8_t* __restrict__ dst, int dst_pitch,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    dst[y * dst_pitch + x] = __ldg(&src[y * src_pitch + x]);
}

/* ─── frame_generate_impl ────────────────────────────────────────────
 * Fills payload region of BGR frame with black/white blocks.
 * Frame must have calibration + sync row pre-rendered via render_template.
 * 2D grid: (grid_cols, payload_rows) blocks, (block_size, block_size) threads. */

__global__ void frame_generate_impl(
    uint8_t* __restrict__ frame, int stride,
    const uint8_t* __restrict__ bits,
    int pay_y, int pay_x,
    int block_size, int grid_cols, int payload_rows)
{
    int col = blockIdx.x;
    int row = blockIdx.y;
    if (row >= payload_rows || col >= grid_cols) return;

    int bit_idx = row * grid_cols + col;
    uint8_t val = bits[bit_idx] ? 255 : 0;

    int x0 = pay_x + col * block_size;
    int y0 = pay_y + row * block_size;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    if (tx >= block_size || ty >= block_size) return;

    int px = x0 + tx;
    int py = y0 + ty;
    int off = py * stride + px * 3;
    frame[off]     = val; /* B */
    frame[off + 1] = val; /* G */
    frame[off + 2] = val; /* R */
}

/* ─── bgr24_to_bgra32_impl ───────────────────────────────────────────
 * Convert packed BGR24 (3 bytes/pixel) to BGRA32 (4 bytes/pixel) with
 * alpha channel set to 0xFF. This is required by NVENC's ARGB input format. *
 * Each thread handles one pixel. */

__global__ void bgr24_to_bgra32_impl(
    const uint8_t* __restrict__ src, int src_stride,
    uint8_t* __restrict__ dst, int dst_stride,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int si = y * src_stride + x * 3;
    int di = y * dst_stride + x * 4;
    dst[di + 0] = src[si + 0];  /* B */
    dst[di + 1] = src[si + 1];  /* G */
    dst[di + 2] = src[si + 2];  /* R */
    dst[di + 3] = 0xFF;         /* A */
}

/* ─── Helper: extract a calibration bit from packed uint32 data ─────
 * Calibration data is 24 bytes packed as 6 uint32_t values.
 * On little-endian GPUs (all NVIDIA GPUs), byte i within each word
 * occupies bits (i*8)..(i*8+7).  Bit order within each byte is MSB-first
 * to match the CPU serialization in build_calibration_bytes(). */

static __device__ __forceinline__
uint8_t get_cal_bit(const uint32_t* __restrict__ cal_data, int idx) {
    int word_idx    = idx / 32;
    int byte_in_word = (idx % 32) / 8;     /* 0 = LSB byte, 3 = MSB byte on LE */
    int bit_in_byte  = 7 - (idx % 8);       /* MSB first within byte */
    int bit_pos      = byte_in_word * 8 + bit_in_byte;
    return (cal_data[word_idx] & (1u << bit_pos)) ? 255 : 0;
}

/* ─── render_template_impl ───────────────────────────────────────────
 * Pre-renders calibration dots + sync row into frame buffer.
 * Calibration layout constants (CAL_COLS, CAL_TOP_FRAC, etc.) are
 * defined in gpu_kernels.h (included above).
 *
 * Calibration bar: top 2-6% of frame, 48x4 grid of black/white dots.
 * Sync row: first grid row below calibration margin, alternating 0,1,0,1...
 *
 * Each thread handles one pixel. Grid covers calibration region. */

__global__ void render_template_impl(
    uint8_t* __restrict__ frame, int stride,
    int width, int height,
    int grid_cols, int block_size,
    int margin_x, int margin_y,
    const uint32_t* __restrict__ cal_data)  /* 24 bytes = 6 uint32, device ptr */
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    int off = py * stride + px * 3;

    /* ── Calibration bar (top 2-6% of frame) ────────────────────── */
    int cal_top    = (int)(height * CAL_TOP_FRAC);
    int cal_height = (int)(height * CAL_HEIGHT_FRAC);
    int cal_left   = (int)(width  * CAL_LEFT_FRAC);
    int cal_width  = (int)(width  * CAL_WIDTH_FRAC);

    if (py >= cal_top && py < cal_top + cal_height &&
        px >= cal_left && px < cal_left + cal_width) {
        int cell_w = cal_width  / CAL_COLS;
        int cell_h = cal_height / CAL_ROWS;
        int cal_row = (py - cal_top) / cell_h;
        int cal_col = (px - cal_left) / cell_w;
        if (cal_row < CAL_ROWS && cal_col < CAL_COLS) {
            /* Compute dot bounds (70% of cell size, centered) — matching CPU write_calibration_dots */
            int dot_w = (cell_w * 7) / 10; if (dot_w < 2) dot_w = 2;
            int dot_h = (cell_h * 7) / 10; if (dot_h < 2) dot_h = 2;
            int cx = cal_left + cal_col * cell_w + cell_w / 2;
            int cy = cal_top  + cal_row * cell_h + cell_h / 2;
            int x1 = cx - dot_w / 2; if (x1 < 0) x1 = 0;
            int y1 = cy - dot_h / 2; if (y1 < 0) y1 = 0;
            int x2 = x1 + dot_w; if (x2 > width)  x2 = width;
            int y2 = y1 + dot_h; if (y2 > height) y2 = height;

            if (px >= x1 && px < x2 && py >= y1 && py < y2) {
                /* Inside dot — read bit from calibration data */
                int idx = cal_row * CAL_COLS + cal_col;  /* 0..191 */
                frame[off] = frame[off + 1] = frame[off + 2] = get_cal_bit(cal_data, idx);
            } else {
                /* Outside dot — black background */
                frame[off] = frame[off + 1] = frame[off + 2] = 0;
            }
            return;
        }
    }

    /* ── Sync row (first row of grid, below calibration margin) ─── */
    int cal_bottom = cal_top + cal_height;
    int sync_y = cal_bottom + margin_y;
    if (py >= sync_y && py < sync_y + block_size &&
        px >= margin_x && px < margin_x + grid_cols * block_size) {
        int col = (px - margin_x) / block_size;
        if (col < grid_cols) {
            uint8_t val = (col % 2) ? 255 : 0;
            frame[off] = frame[off + 1] = frame[off + 2] = val;
            return;
        }
    }

    /* ── Everything else: black background ────────────────────────── */
    frame[off] = frame[off + 1] = frame[off + 2] = 0;
}

/* ─── extract_bits_impl ────────────────────────────────────────────
 * Decode payload tiles directly from a grayscale frame in GPU memory,
 * writing only the compact bit array to output.
 *
 * This eliminates the need to copy the entire grayscale frame to CPU
 * just for bit extraction. Each thread handles one tile.
 *
 * Grid layout: (grid_cols, pay_rows) with 1 thread per tile.
 * For each tile, subsamples the center region and counts white pixels
 * against THRESHOLD (128), mirroring the CPU-side tile_decode_grid in
 * simd_decode.c. */

__global__ void extract_bits_impl(
    const uint8_t* __restrict__ gray, int gray_stride,
    int grid_top_y, int grid_left_x,
    int block_size,
    int grid_cols, int pay_rows,
    int subsample,
    int sync_rows,
    uint8_t* __restrict__ bits_out)
{
    int col = blockIdx.x;
    int row = blockIdx.y;
    if (col >= grid_cols || row >= pay_rows) return;

    int tile_y = grid_top_y + (row + sync_rows) * block_size;
    int tile_x = grid_left_x + col * block_size;

    int total_samples = 0;
    int white_count   = 0;

    for (int ty = 0; ty < block_size; ty += subsample) {
        for (int tx = 0; tx < block_size; tx += subsample) {
            int py = tile_y + ty;
            int px = tile_x + tx;
            uint8_t val = __ldg(&gray[py * gray_stride + px]);
            if (val >= THRESHOLD) ++white_count;
            ++total_samples;
        }
    }

    int bit_idx = row * grid_cols + col;
    bits_out[bit_idx] = (white_count > total_samples / 2) ? 1 : 0;
}

/* ─── extract_calibration_bits_impl ───────────────────────────────────
 * Read calibration dots from a grayscale frame in GPU memory.
 * Mirrors tile_read_calibration from simd_decode.c.
 * Each thread handles one calibration dot (1 of 192).
 * Writes bits to output array (one byte per bit). */

__global__ void extract_calibration_bits_impl(
    const uint8_t* __restrict__ gray, int gray_stride,
    int width, int height,
    uint8_t* __restrict__ bits_out)
{
    int dot_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (dot_idx >= CAL_BITS_TOTAL) return;

    int cal_top    = (int)((float)height * CAL_TOP_FRAC);
    int cal_height = (int)((float)height * CAL_HEIGHT_FRAC);
    int cal_left   = (int)((float)width  * CAL_LEFT_FRAC);
    int cal_width  = (int)((float)width  * CAL_WIDTH_FRAC);

    if (cal_height < CAL_ROWS || cal_width < CAL_COLS) return;

    int row = dot_idx / CAL_COLS;
    int col = dot_idx % CAL_COLS;

    int cell_w = cal_width  / CAL_COLS;
    int cell_h = cal_height / CAL_ROWS;
    int sample_w = (cell_w * 60) / 100; if (sample_w < 2) sample_w = 2;
    int sample_h = (cell_h * 60) / 100; if (sample_h < 2) sample_h = 2;

    int cx = cal_left + col * cell_w + cell_w / 2;
    int cy = cal_top  + row * cell_h + cell_h / 2;
    int sx = cx - sample_w / 2;
    int sy = cy - sample_h / 2;
    if (sx < 0) sx = 0;
    if (sy < 0) sy = 0;
    int ex = sx + sample_w; if (ex > width)  ex = width;
    int ey = sy + sample_h; if (ey > height) ey = height;

    int sum = 0;
    int samples = 0;
    for (int yy = sy; yy < ey; ++yy) {
        for (int xx = sx; xx < ex; ++xx) {
            sum += (int)gray[yy * gray_stride + xx];
            samples++;
        }
    }

    if (samples == 0) return;
    bits_out[dot_idx] = (uint8_t)((sum / samples) >= THRESHOLD ? 1 : 0);
}

/* ─── read_calibration_dots_impl ──────────────────────────────────────
 * Reads calibration dots from a grayscale frame and packs 192 bits
 * into 6 uint32 values (MSB-first bit order within each byte, matching
 * the layout that get_cal_bit() expects from render_template).
 *
 * Each thread handles one calibration dot (1 of 192).  The grid should
 * be launched as dim3(CAL_COLS, CAL_ROWS) with a block of 1 thread.
 *
 * For each dot, the thread samples the center 60% of its cell area
 * (matching CPU read_calibration_dots) and applies a threshold (128)
 * to determine the bit value. */

__global__ void read_calibration_dots_impl(
    const uint8_t* __restrict__ gray, int gray_stride,
    int width, int height,
    uint32_t* __restrict__ cal_data_out)  /* 6 uint32 = 24 bytes */
{
    int dot_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (dot_idx >= CAL_BITS_TOTAL) return;

    int cal_top    = (int)(height * CAL_TOP_FRAC);
    int cal_height = (int)(height * CAL_HEIGHT_FRAC);
    int cal_left   = (int)(width  * CAL_LEFT_FRAC);
    int cal_width  = (int)(width  * CAL_WIDTH_FRAC);

    /* Invalid calibration region — set to 0, caller will detect via CRC */
    if (cal_height < CAL_ROWS || cal_width < CAL_COLS) return;

    int row = dot_idx / CAL_COLS;
    int col = dot_idx % CAL_COLS;

    int cell_w = cal_width  / CAL_COLS;
    int cell_h = cal_height / CAL_ROWS;
    int sample_w = (cell_w * 60) / 100; if (sample_w < 2) sample_w = 2;
    int sample_h = (cell_h * 60) / 100; if (sample_h < 2) sample_h = 2;

    int cx = cal_left + col * cell_w + cell_w / 2;
    int cy = cal_top  + row * cell_h + cell_h / 2;
    int sx = cx - sample_w / 2;
    int sy = cy - sample_h / 2;

    /* Clamp to frame bounds */
    if (sx < 0) sx = 0;
    if (sy < 0) sy = 0;
    int ex = sx + sample_w; if (ex > width)  ex = width;
    int ey = sy + sample_h; if (ey > height) ey = height;

    int sum = 0;
    int samples = 0;
    for (int yy = sy; yy < ey; ++yy) {
        for (int xx = sx; xx < ex; ++xx) {
            sum += (int)gray[yy * gray_stride + xx];
            samples++;
        }
    }

    if (samples == 0) return;
    uint8_t bit = (sum / samples) >= THRESHOLD ? 1 : 0;

    /* Pack bit into output: MSB-first within each byte, little-endian byte
     * order within each uint32 (matching how render_template reads them).
     * bit 'dot_idx' lands in byte (dot_idx/8) at bit position 7-(dot_idx%8).
     * On LE: byte 0 of cal_data = bits 0..7 of word0, so */
    int word_idx    = dot_idx / 32;
    int byte_in_word = (dot_idx % 32) / 8;
    int bit_in_byte  = 7 - (dot_idx % 8);
    int bit_pos      = byte_in_word * 8 + bit_in_byte;

    if (bit) {
        atomicOr(&cal_data_out[word_idx], (1u << bit_pos));
    }
    /* else: bit stays 0 (already zero-initialized) */
}

/* ─── bgr24_to_gray_impl ────────────────────────────────────────────
 * Extract G channel from packed BGR24 to create planar grayscale.
 * BGR pixel format: byte 0=B, byte 1=G, byte 2=R.
 * The G channel is a good luminance approximation for our thresholding. */

__global__ void bgr24_to_gray_impl(
    const uint8_t* __restrict__ src, int src_stride,
    uint8_t* __restrict__ dst, int dst_stride,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    /* Extract G channel (byte index 1 in each BGR pixel) */
    dst[y * dst_stride + x] = __ldg(&src[y * src_stride + x * 3 + 1]);
}

/* ─── extract_bits wrapper ────────────────────────────────────────── */

extern "C" void extract_bits_kernel(
    dim3 grid, dim3 block,
    const uint8_t *gray, int gray_stride,
    int grid_top_y, int grid_left_x,
    int block_size,
    int grid_cols, int pay_rows,
    int subsample,
    int sync_rows,
    uint8_t *bits_out,
    cudaStream_t stream)
{
    /* Each block == one tile */
    extract_bits_impl<<<grid, 1, 0, stream>>>(
        gray, gray_stride,
        grid_top_y, grid_left_x,
        block_size, grid_cols, pay_rows,
        subsample, sync_rows,
        bits_out);
}

extern "C" void extract_calibration_bits_kernel(
    const uint8_t *gray, int gray_stride,
    int width, int height,
    uint8_t *bits_out,
    cudaStream_t stream)
{
    /* 192 threads, one per calibration dot */
    extract_calibration_bits_impl<<<1, 192, 0, stream>>>(
        gray, gray_stride, width, height, bits_out);
}

extern "C" void bgr24_to_gray_kernel(
    dim3 grid, dim3 block,
    const uint8_t *src, int src_stride,
    uint8_t *dst, int dst_stride,
    int width, int height,
    cudaStream_t stream)
{
    bgr24_to_gray_impl<<<grid, block, 0, stream>>>(
        src, src_stride, dst, dst_stride, width, height);
}

/* ─── extern "C" wrappers (called from gpu_backend.c) ─────────────── */

extern "C" void nv12_to_gray_kernel(
    dim3 grid, dim3 block,
    const uint8_t *src, int src_pitch,
    uint8_t *dst, int dst_pitch,
    int width, int height,
    cudaStream_t stream)
{
    nv12_to_gray_impl<<<grid, block, 0, stream>>>(
        src, src_pitch, dst, dst_pitch, width, height);
}

extern "C" void frame_generate_kernel(
    dim3 grid, dim3 block,
    uint8_t *frame, int stride,
    const uint8_t *bits,
    int pay_y, int pay_x,
    int block_size, int grid_cols, int payload_rows,
    cudaStream_t stream)
{
    frame_generate_impl<<<grid, block, 0, stream>>>(
        frame, stride, bits, pay_y, pay_x,
        block_size, grid_cols, payload_rows);
}

extern "C" void bgr24_to_bgra32_kernel(
    dim3 grid, dim3 block,
    const uint8_t *src, int src_stride,
    uint8_t *dst, int dst_stride,
    int width, int height,
    cudaStream_t stream)
{
    bgr24_to_bgra32_impl<<<grid, block, 0, stream>>>(
        src, src_stride, dst, dst_stride, width, height);
}

extern "C" void render_template_kernel(
    dim3 grid, dim3 block,
    uint8_t *frame, int stride,
    int width, int height,
    int grid_cols, int block_size,
    int margin_x, int margin_y,
    const uint32_t *cal_data,
    cudaStream_t stream)
{
    (void)grid;
    dim3 g((unsigned int)((width  + block.x - 1) / block.x),
           (unsigned int)((height + block.y - 1) / block.y));
    render_template_impl<<<g, block, 0, stream>>>(
        frame, stride, width, height,
        grid_cols, block_size, margin_x, margin_y,
        cal_data);
}

extern "C" void read_calibration_dots_kernel(
    dim3 grid, dim3 block,
    const uint8_t *gray, int gray_stride,
    int width, int height,
    uint32_t *cal_data_out,
    cudaStream_t stream)
{
    (void)grid; (void)block;
    /* Launch 192 threads, one per calibration dot */
    read_calibration_dots_impl<<<1, 192, 0, stream>>>(
        gray, gray_stride, width, height, cal_data_out);
}

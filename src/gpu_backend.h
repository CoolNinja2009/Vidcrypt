#ifndef VIDCRYPT_GPU_BACKEND_H
#define VIDCRYPT_GPU_BACKEND_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "gpu_presets.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Forward declarations ────────────────────────────────────────── */
struct GpuBackend;
typedef struct GpuBackend GpuBackend;

/* ─── GPU Backend ───────────────────────────────────────────────────
 * Manages CUDA device, memory pool, streams, NVDEC/NVENC lifecycle. */

typedef enum {
    GPU_INIT_OK = 0,
    GPU_ERR_NO_DEVICE,
    GPU_ERR_DRIVER,
    GPU_ERR_MEMORY,
    GPU_ERR_NVDEC,
    GPU_ERR_NVENC,
} GpuStatus;

/* ─── Lifecycle ───────────────────────────────────────────────────── */

/* Initialize GPU backend. Detects GPU, selects preset, allocates pool.
 * Returns status code. On failure, error_out (if non-NULL) contains message. */
GpuStatus gpu_backend_init(GpuBackend **backend, int device_id,
                           char *error_out, int error_size);
void      gpu_backend_destroy(GpuBackend *backend);

/* Get the active preset for this GPU */
const GpuPreset* gpu_backend_preset(const GpuBackend *backend);

/* Get device name string */
const char* gpu_backend_device_name(const GpuBackend *backend);

/* Open/close backend-owned NVDEC and NVENC instances without exposing
 * GpuBackend internals to callers. */
bool gpu_backend_open_nvdec(GpuBackend *backend, const char *path,
                            char *error_out, int error_size);
void gpu_backend_close_nvdec(GpuBackend *backend);

bool gpu_backend_open_nvenc(GpuBackend *backend, int width, int height,
                            double fps, int bitrate, int gop_length,
                            char *error_out, int error_size);
void gpu_backend_close_nvenc(GpuBackend *backend);
int  gpu_backend_get_nvenc_packet(GpuBackend *backend, const uint8_t **packet_out);
int  gpu_backend_flush_nvenc(GpuBackend *backend);

uint32_t* gpu_backend_calibration_buffer(GpuBackend *backend);

/* ─── Decode pipeline helpers ─────────────────────────────────────── */

/* Decode a video frame via FFmpeg+CUDA hwaccel, returning a CUdeviceptr
 * to the NV12 Y-plane in GPU memory. The frame stays in GPU memory —
 * no CPU transfer. FFmpeg handles demuxing, codec detection, and
 * NVDEC initialization automatically. */
bool gpu_backend_decode_frame(GpuBackend *backend,
                              const uint8_t *encoded_packet, int packet_size,
                              uint8_t **y_plane_out, int *pitch_out,
                              int *width_out, int *height_out);

void gpu_backend_unmap_decode_frame(GpuBackend *backend);

/* Convert BGR24 frame to planar grayscale on GPU (extracts G channel).
 * 'd_bgr' is a CUdeviceptr to packed BGR24 (width*3 bytes per row).
 * 'bgr_stride' = width * 3.
 * Returns device pointer to grayscale buffer (width * height bytes),
 * or NULL on failure. */
uint8_t* gpu_backend_bgr24_to_gray(GpuBackend *backend,
                                    const uint8_t *d_bgr, int bgr_stride,
                                    int width, int height);

/* Convert NV12 frame to planar grayscale (both in GPU memory).
 * Returns device pointer to grayscale buffer. */
uint8_t* gpu_backend_nv12_to_gray(GpuBackend *backend,
                                   const uint8_t *nv12_y, int pitch,
                                   int width, int height);

/* ─── GPU-side bit extraction ─────────────────────────────────────── */

/* Extract payload bits from a grayscale GPU frame.
 * Launch GPU kernel to decode all tiles, returns device pointer to bits.
 * Only the compact bit array (~KB) needs to be transferred to CPU.
 * Parameters match the DecodeGeometry layout. */
uint8_t* gpu_backend_extract_bits(GpuBackend *backend,
                                   const uint8_t *d_gray, int gray_stride,
                                   int width, int height,
                                   int grid_top_y, int grid_left_x,
                                   int block_size,
                                   int grid_cols, int pay_rows,
                                   int subsample, int sync_rows);

/* Copy GPU-extracted bits to pinned CPU buffer.
 * Must call gpu_backend_sync_decode() before reading.
 * Returns pointer to host memory with bits (0 or 1 per byte). */
uint8_t* gpu_backend_get_bits_cpu(GpuBackend *backend, int total_bits);

/* Extract calibration bits (192 dots) from a grayscale GPU frame.
 * Returns pointer to pinned host buffer (192 bytes, one per bit).
 * Must sync before reading. */
uint8_t* gpu_backend_extract_calibration(GpuBackend *backend,
                                          const uint8_t *d_gray,
                                          int gray_stride,
                                          int width, int height);

/* ─── Encode pipeline helpers ─────────────────────────────────────── */

/* Generate a BGR frame from payload bits, in GPU memory.
 * Returns device pointer to BGR frame. */
uint8_t* gpu_backend_generate_frame(GpuBackend *backend,
                                     const uint8_t *bits, int nbits,
                                     int width, int height,
                                     int grid_cols, int payload_rows,
                                     int block_size,
                                     int margin_x, int margin_y);

/* Write a GPU frame to video via NVENC (or fallback FFmpeg pipe).
 * Returns true on success. */
bool gpu_backend_write_frame(GpuBackend *backend,
                              const uint8_t *gpu_frame, int stride);

/* ─── Calibration ──────────────────────────────────────────────────── */

/* Read calibration dots from a grayscale frame in GPU memory.
 * 'd_gray' is a device pointer to grayscale data (width * height bytes).
 * 'gray_stride' = width (tightly packed).
 * 'cal_data_out' is a device pointer to 24 bytes of output (zero-initialized).
 * The result can be parsed by parse_calibration_bytes() on CPU.
 * Returns true on success. */
bool gpu_backend_read_calibration(GpuBackend *backend,
                                   const uint8_t *d_gray, int gray_stride,
                                   int width, int height,
                                   uint32_t *d_cal_data_out);

/* ─── CPU grayscale upload / download ─────────────────────────────── */

/* Upload a CPU grayscale frame to the GPU working buffer for kernel processing.
 * 'gray': host pointer to grayscale data (width * height bytes).
 * 'stride': bytes per row (typically = width for tightly packed grayscale).
 * 'width', 'height': frame dimensions.
 * Returns device pointer to uploaded frame, or NULL on failure.
 * The returned pointer is valid until the next upload or gpu_backend_destroy. */
uint8_t* gpu_backend_upload_gray(GpuBackend *backend,
                                  const uint8_t *gray, int stride,
                                  int width, int height);

/* Download a GPU grayscale frame to CPU memory.
 * 'd_gray': device pointer to grayscale data (must belong to this backend).
 * 'gray_stride': bytes per row on GPU (typically = width).
 * 'width', 'height': frame dimensions.
 * 'h_dst': host destination buffer (must be at least width * height bytes).
 * 'dst_stride': bytes per row in host buffer.
 * The copy is queued on the decode stream — call gpu_backend_sync_decode()
 * before reading the host buffer.
 * Returns true on success. */
bool gpu_backend_download_gray(GpuBackend *backend,
                                const uint8_t *d_gray, int gray_stride,
                                int width, int height,
                                uint8_t *h_dst, int dst_stride);

/* ─── Synchronization ─────────────────────────────────────────────── */

/* Ensure all pending GPU work is complete */
void gpu_backend_sync(GpuBackend *backend);

/* Sync only the encode stream (lighter than syncing all streams).
 * Call before reading encode results on CPU or submitting frames to NVENC. */
void gpu_backend_sync_encode(GpuBackend *backend);

/* Sync only the decode stream. Call before reading decode results on CPU. */
void gpu_backend_sync_decode(GpuBackend *backend);

/* Get the decode stream handle (for external synchronization).
 * Returns opaque cudaStream_t as void*. */
void* gpu_backend_get_decode_stream(GpuBackend *backend);

#ifdef __cplusplus
}
#endif

#endif /* VIDCRYPT_GPU_BACKEND_H */

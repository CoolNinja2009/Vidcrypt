#ifndef VIDCRYPT_GPU_NVENC_H
#define VIDCRYPT_GPU_NVENC_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── NVENC Encoder ──────────────────────────────────────────────────
 * Hardware-accelerated H.264 video encoder using NVIDIA's NVENC engine.
 *
 * Zero-Copy Pipeline (no PCIe transfers, all GPU-resident):
 *   1. BGR24 GPU frame → BGR24→BGRA32 conversion kernel (in-place on GPU)
 *   2. NVENC registered resource: reads BGRA32 directly from CUDA device memory
 *      via nvEncRegisterResource + nvEncMapInputResource — zero copy
 *   3. NVENC ASIC encodes to H.264 bitstream (hardware-accelerated)
 *   4. Encoded output stored in ring buffer, retrieved by caller
 *
 * Constraints (RTX 5070, driver 610.62, CUDA 12.8):
 *   - NV_ENC_BUFFER_FORMAT_NV12 → NV_ENC_ERR_INVALID_PARAM
 *     Workaround: use NV_ENC_BUFFER_FORMAT_ARGB with BGR24→BGRA32 kernel
 *   - enableEncodeAsync=1 → NV_ENC_ERR_INVALID_PARAM
 *     Workaround: sync mode only
 *   - NvEncGetSequenceParams() → segfault (API version mismatch)
 *     Workaround: NAL parser extracts SPS/PPS from first IDR bitstream,
 *     with manual construction fallback
 *   - NV_ENC_PIC_FLAG_FORCEIDR: emits AUD, SPS/PPS must come from
 *     separate NAL units in initial bitstream */

struct GpuNvencEncoder;
typedef struct GpuNvencEncoder GpuNvencEncoder;

/* ─── Lifecycle ───────────────────────────────────────────────────── */

/* Create an NVENC encoder session using zero-copy registered resource.
 * 'width', 'height': frame dimensions (must match generated frames).
 * 'fps': target frame rate.
 * 'cuda_device': CUDA device ID.
 * 'bitrate': target bitrate in kbps (0 = auto ~50 Mbps for quality).
 * 'gop_length': GOP length in frames (0 = auto ~fps*2).
 * 'error_out': optional error buffer (256 bytes recommended).
 * Returns NULL on failure. */
GpuNvencEncoder* gpu_nvenc_create(int width, int height, double fps,
                                   int cuda_device, int bitrate,
                                   int gop_length,
                                   char *error_out, int error_size);

/* Destroy the encoder and free all resources.
 * Flushes any remaining frames before cleanup. */
void gpu_nvenc_destroy(GpuNvencEncoder *enc);

/* ─── Zero-Copy Encoding ──────────────────────────────────────────── */

/* Encode a BGR24 frame from GPU device memory — zero PCIe copies.
 *
 * 'd_frame_bgr24': CUDA device pointer to tightly-packed BGR24 frame
 *                  (width * height * 3 bytes). Must be in GPU memory.
 * 'stride': bytes per row (width * 3 for tightly packed BGR24).
 * 'packet_out': receives pointer to encoded H.264 bitstream data
 *               (valid until next call to this function or destroy).
 *               Set to NULL if caller will use gpu_nvenc_get_packet().
 * 'packet_size_out': receives size in bytes of the encoded packet.
 *
 * Pipeline (all on GPU, sync mode):
 *   1. BGR24 → BGRA32 via CUDA kernel (d_bgra is NVENC-registered resource)
 *   2. nvEncMapInputResource → maps d_bgra for NVENC read access
 *   3. nvEncEncodePicture → NVENC reads BGRA32 directly from GPU memory
 *   4. nvEncLockBitstream → retrieve encoded H.264 bitstream
 *
 * First frame: synchronous encode + SPS/PPS extraction from bitstream.
 * Returns 0 on success, -1 on error.
 * Caller must drain packets via gpu_nvenc_get_packet() after each call. */
int gpu_nvenc_encode_frame_zerocopy(GpuNvencEncoder *enc,
                                     const uint8_t *d_frame_bgr24,
                                     int stride,
                                     const uint8_t **packet_out,
                                     int *packet_size_out);

/* Retrieve the next available encoded bitstream packet.
 * 'packet_out': receives pointer to the encoded data (valid until next
 *               call to gpu_nvenc_encode_frame_zerocopy or destroy).
 * Returns the size in bytes of the packet, or 0 if no packet available.
 * The returned pointer points to internal memory. */
int gpu_nvenc_get_packet(GpuNvencEncoder *enc,
                          const uint8_t **packet_out);

/* Flush the encoder — drains all buffered frames.
 * Call after all frames have been submitted.
 * After flush, call gpu_nvenc_get_packet repeatedly until it returns 0. */
int gpu_nvenc_flush(GpuNvencEncoder *enc);

/* ─── SPS/PPS Extradata ───────────────────────────────────────────── */

/* Retrieve SPS + PPS NAL units in AVCC format (4-byte length-prefixed).
 * Extracted from first encoded frame's bitstream via NAL parser (types 7/8).
 * Falls back to manual construction if extraction fails.
 * Returns size in bytes, or 0 if not yet available.
 * Call after the first gpu_nvenc_encode_frame_zerocopy() has completed. */
int gpu_nvenc_get_sps_pps(GpuNvencEncoder *enc, const uint8_t **data_out);

/* ─── Statistics ──────────────────────────────────────────────────── */

/* Get total encoded bytes so far. */
int64_t gpu_nvenc_encoded_bytes(GpuNvencEncoder *enc);

/* Get total frames encoded so far. */
int gpu_nvenc_encoded_frames(GpuNvencEncoder *enc);

/* ─── Legacy API (kept for compatibility) ─────────────────────────── */

/* Encode a BGR24 GPU frame (old path with D2H copy — deprecated).
 * Use gpu_nvenc_encode_frame_zerocopy() for zero-copy path. */
int gpu_nvenc_encode_frame(GpuNvencEncoder *enc,
                            const uint8_t *gpu_frame_bgr24, int stride);

/* ─── NV12 path (broken on this driver — NV_ENC_ERR_INVALID_PARAM) ── */
int gpu_nvenc_encode_frame_nv12(GpuNvencEncoder *enc,
                                 const uint8_t *d_y,
                                 const uint8_t *d_uv,
                                 void *cuda_stream);
int gpu_nvenc_encode_frame_nv12_submit(GpuNvencEncoder *enc,
                                        const uint8_t *d_y,
                                        const uint8_t *d_uv,
                                        void *cuda_stream);
int gpu_nvenc_drain_output(GpuNvencEncoder *enc);

#ifdef __cplusplus
}
#endif

#endif /* VIDCRYPT_GPU_NVENC_H */

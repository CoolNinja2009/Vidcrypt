#ifndef VIDCRYPT_GPU_NVENC_H
#define VIDCRYPT_GPU_NVENC_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── NVENC Encoder ──────────────────────────────────────────────────
 * Hardware-accelerated video encoder using NVIDIA's NVENC engine.
 * Replaces the ffmpeg pipe encode pipeline (which was 95% of encode time).
 *
 * Pipeline:
 *   1. BGR24 GPU frame → BGR24→BGRA32 conversion kernel
 *   2. NVENC encodes BGRA32 input to H.264 bitstream (GPU-accelerated)
 *   3. Encoded bitstream output is stored in a ring buffer of output packets
 *   4. Caller writes bitstream to file or pipes to ffmpeg for containerization
 *
 * NVENC handles the entire video encoding pipeline on GPU hardware,
 * eliminating the CPU-side ffmpeg encode + pipe overhead entirely. */

struct GpuNvencEncoder;
typedef struct GpuNvencEncoder GpuNvencEncoder;

/* ─── Lifecycle ───────────────────────────────────────────────────── */

/* Create an NVENC encoder session.
 * 'width', 'height': frame dimensions (must match generated frames).
 * 'fps': target frame rate.
 * 'cuda_device': CUDA device ID.
 * 'bitrate': target bitrate in kbps (0 = auto ~8 Mbps for 1080p).
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

/* ─── Encoding ────────────────────────────────────────────────────── */

/* Encode a BGR24 GPU frame.
 * 'gpu_frame_bgr24': CUdeviceptr to the BGR24 frame in GPU memory.
 * 'stride': bytes per row (width * 3 for tightly packed BGR24).
 * 
 * Internally converts BGR24 → BGRA32 via CUDA kernel and submits
 * to NVENC. Encoded bitstream is stored internally.
 * Returns the size of the encoded packet (0 if frame was buffered,
 * >0 if a packet is available for retrieval). */
int gpu_nvenc_encode_frame(GpuNvencEncoder *enc,
                            const uint8_t *gpu_frame_bgr24, int stride);

/* Retrieve the next available encoded bitstream packet.
 * 'packet_out': receives pointer to the encoded data (valid until next
 *               call to gpu_nvenc_encode_frame or gpu_nvenc_destroy).
 * Returns the size in bytes of the packet, or 0 if no packet available.
 * The returned pointer points to internal memory. */
int gpu_nvenc_get_packet(GpuNvencEncoder *enc,
                          const uint8_t **packet_out);

/* Flush the encoder — drains all buffered frames.
 * Call after all frames have been submitted via gpu_nvenc_encode_frame.
 * After flush, call gpu_nvenc_get_packet repeatedly until it returns 0. */
int gpu_nvenc_flush(GpuNvencEncoder *enc);

/* Get total encoded bytes so far. */
int64_t gpu_nvenc_encoded_bytes(GpuNvencEncoder *enc);

/* Get total frames encoded so far. */
int gpu_nvenc_encoded_frames(GpuNvencEncoder *enc);

#ifdef __cplusplus
}
#endif

#endif /* VIDCRYPT_GPU_NVENC_H */

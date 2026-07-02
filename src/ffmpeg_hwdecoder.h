#ifndef VIDCRYPT_FFMPEG_HWDECODER_H
#define VIDCRYPT_FFMPEG_HWDECODER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── FFmpeg HW Decoder ──────────────────────────────────────────────
 * Hardware-accelerated video decoder using FFmpeg's CUDA hwaccel
 * infrastructure. Replaces the custom NVDEC decoder (gpu_nvdec.c).
 *
 * Design:
 *   1. Uses libavformat for container parsing and demuxing
 *   2. Uses libavcodec with AV_HWDEVICE_TYPE_CUDA for NVDEC hwaccel
 *   3. Decoded frames are in CUDA memory (AV_PIX_FMT_CUDA)
 *   4. CUdeviceptr extracted from AVFrame.data[0-3] for zero-copy
 *      CUDA kernel access
 *   5. Supports H.264, H.265, AV1, VP9 — all codecs FFmpeg+NVDEC supports
 *   6. Automatic CPU fallback for unsupported codecs (FFV1, etc.)
 */

struct FfmpegHwDecoder;
typedef struct FfmpegHwDecoder FfmpegHwDecoder;

/* ─── Codec info ──────────────────────────────────────────────────── */
typedef enum {
    FFMPEG_HW_CODEC_UNKNOWN = 0,
    FFMPEG_HW_CODEC_H264,
    FFMPEG_HW_CODEC_H265,
    FFMPEG_HW_CODEC_AV1,
    FFMPEG_HW_CODEC_VP9,
    FFMPEG_HW_CODEC_MPEG2,
    FFMPEG_HW_CODEC_VC1,
} FfmpegHwCodec;

/* ─── Lifecycle ───────────────────────────────────────────────────── */

/* Open a video file for FFmpeg+CUDA hardware decoding.
 * 'path': path to the encoded video file (any container: MP4, MKV, AVI, etc.)
 * 'cuda_device': CUDA device ID to use (typically 0)
 * 'error_out': optional error buffer (256 bytes recommended)
 * 'error_size': size of error buffer
 * Returns NULL on failure.
 *
 * Internally:
 *   1. Opens container with avformat_open_input
 *   2. Finds best video stream
 *   3. Creates CUDA hwdevice context (av_hwdevice_ctx_create)
 *   4. Opens decoder with hw_device_ctx set
 *   5. Decoder outputs AV_PIX_FMT_CUDA frames in GPU memory
 *
 * The returned FfmpegHwDecoder manages:
 *   - AVFormatContext (demuxer)
 *   - AVCodecContext (decoder with CUDA hwaccel)
 *   - AVHWDeviceContext (CUDA device context)
 *   - Internal AVPacket reuse
 */
FfmpegHwDecoder* ffmpeg_hwdecoder_open(const char *path, int cuda_device,
                                        char *error_out, int error_size);

/* Close the decoder and free all resources (FFmpeg contexts, CUDA). */
void ffmpeg_hwdecoder_close(FfmpegHwDecoder *dec);

/* ─── Frame decoding ──────────────────────────────────────────────── */

/* Decode and retrieve the next frame as a linear NV12 Y-plane.
 * 'y_plane_out': receives pointer to the NV12 Y-plane (grayscale) in
 *                system memory. Data is linear (not tiled) — handles
 *                av_hwframe_transfer_data() internally.
 * 'pitch_out': receives the pitch (bytes per row) of the Y-plane.
 *              May be > width if aligned (handle when copying).
 * 'width_out', 'height_out': receive frame dimensions.
 * Returns true if a frame was decoded, false on end of stream or error.
 *
 * The returned pointer is valid until the next call to
 * ffmpeg_hwdecoder_read_frame(). The Y-plane IS grayscale (NV12 luma).
 * No CUDA context management needed — data is CPU-accessible. */
bool ffmpeg_hwdecoder_read_frame(FfmpegHwDecoder *dec,
                                  uint8_t **y_plane_out,
                                  int *pitch_out,
                                  int *width_out,
                                  int *height_out);

/* ─── Metadata ────────────────────────────────────────────────────── */

/* Get total frame count (from avformat stream nb_frames, may be 0). */
int ffmpeg_hwdecoder_frame_count(FfmpegHwDecoder *dec);

/* Get frame rate (fps). */
double ffmpeg_hwdecoder_fps(FfmpegHwDecoder *dec);

/* Get the CUDA stream used by this decoder (from the hwdevice context). */
void* ffmpeg_hwdecoder_stream(FfmpegHwDecoder *dec);

/* Get video dimensions (coded width/height vs display width/height). */
void ffmpeg_hwdecoder_dimensions(FfmpegHwDecoder *dec,
                                  int *coded_width, int *coded_height,
                                  int *display_width, int *display_height);

/* Get the codec type of the video stream. */
FfmpegHwCodec ffmpeg_hwdecoder_codec(FfmpegHwDecoder *dec);

/* Get the raw pixel format of the decoded frames (AVPixelFormat as int). */
int ffmpeg_hwdecoder_pixel_format(FfmpegHwDecoder *dec);

/* Returns true if the decoder is using hardware acceleration.
 * false if it fell back to software decoding (e.g., FFV1 input). */
bool ffmpeg_hwdecoder_is_hardware(FfmpegHwDecoder *dec);

#ifdef __cplusplus
}
#endif

#endif /* VIDCRYPT_FFMPEG_HWDECODER_H */

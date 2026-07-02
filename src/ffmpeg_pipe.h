#ifndef VIDCRYPT_FFMPEG_PIPE_H
#define VIDCRYPT_FFMPEG_PIPE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Video Reader (decoder input) via ffmpeg subprocess ──────────── */

typedef struct VideoReader VideoReader;

/* Open a video file for reading. Uses gray8 rawvideo output from ffmpeg.
 * Each frame is width * height bytes (1 byte per pixel, grayscale). */
VideoReader *video_reader_open(const char *path, const char *ffmpeg_path);

/* Open with NVDEC hardware acceleration (-hwaccel cuda -c:v h264_cuvid).
 * Requires a CUDA-capable ffmpeg binary with cuvid decoders.
 * Falls back to software decode if hwaccel init fails.
 * Note: Only works for H.264 content (the most common case). */
VideoReader *video_reader_open_nvdec(const char *path, const char *ffmpeg_path);

/* Open with optional output scale. If target_width>0 and target_height>0,
 * adds -vf scale=W:H to upscale/downscale to the target resolution.
 * Useful when YouTube has re-encoded at a different resolution. */
VideoReader *video_reader_open_scaled(const char *path, const char *ffmpeg_path,
                                       int target_width, int target_height);

/* Read next frame. frame_out points to gray8 data (width * height bytes).
 * stride_out = width (1 byte per pixel). Returns true on success. */
bool video_reader_read_frame(VideoReader *vr,
                             const uint8_t **frame_out,
                             int *stride_out,
                             int *width_out,
                             int *height_out);

int video_reader_frame_count(VideoReader *vr);
double video_reader_fps(VideoReader *vr);
void video_reader_close(VideoReader *vr);

/* ─── Video Writer (encoder output) via ffmpeg subprocess ─────────── */

typedef struct VideoWriter VideoWriter;

/* Create a video writer. Writes gray8 rawvideo frames to ffmpeg stdin.
 * Output codec defaults to ffv1 (lossless). */
VideoWriter *video_writer_create(const char *ffmpeg_path,
                                  const char *path,
                                  int width, int height,
                                  double fps,
                                  const char *codec_name);

/* Write a gray8 frame (width * height bytes). stride is ignored (assumed = width). */
bool video_writer_write_frame(VideoWriter *vw,
                              const uint8_t *frame, int stride);

/* Write raw data (e.g., encoded H.264 packet) to the writer's pipe.
 * Used for NVENC → ffmpeg muxer pipeline. Returns true on success. */
bool video_writer_write_data(VideoWriter *vw, const uint8_t *data, int size);

void video_writer_close(VideoWriter *vw);

/* ─── H.264 Muxer (for NVENC output) ────────────────────────────── */

/* Create an ffmpeg subprocess that muxes raw H.264 bitstream into a
 * container (MKV/MP4). NVENC outputs raw H.264 AVCC-format packets;
 * ffmpeg handles containerization with -c:v copy (no re-encode).
 *
 * The caller writes raw H.264 NAL units (4-byte length-prefixed AVCC
 * format) to the returned pipe's stdin.
 *
 * Returns NULL on failure. The writer should be closed with
 * video_writer_close(). */
VideoWriter *video_writer_create_h264_muxer(const char *ffmpeg_path,
                                             const char *path,
                                             int width, int height,
                                             double fps);

/* Same as video_writer_create_h264_muxer, but writes SPS+PPS extradata
 * (in AVCC format with 4-byte length prefixes) to the pipe immediately
 * after opening. This allows the H.264 demuxer to initialize before
 * any frame data arrives — essential for async NVENC pipelines where
 * the muxer must be opened before the first async frame completes.
 *
 * 'extradata': SPS + PPS NAL units in AVCC format (4-byte length-prefixed).
 *              May be NULL if no extradata is available.
 * 'extradata_size': size of extradata in bytes. */
VideoWriter *video_writer_create_h264_muxer_extradata(const char *ffmpeg_path,
                                                       const char *path,
                                                       int width, int height,
                                                       double fps,
                                                       const uint8_t *extradata,
                                                       int extradata_size);

/* ─── Utility ─────────────────────────────────────────────────────── */

char *find_ffmpeg_binary(const char *hint, const char *name);
const char *ffmpeg_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* VIDCRYPT_FFMPEG_PIPE_H */

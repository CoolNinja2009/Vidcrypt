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

void video_writer_close(VideoWriter *vw);

/* ─── Utility ─────────────────────────────────────────────────────── */

char *find_ffmpeg_binary(const char *hint, const char *name);
const char *ffmpeg_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* VIDCRYPT_FFMPEG_PIPE_H */

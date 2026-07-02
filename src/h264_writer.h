#ifndef VIDCRYPT_H264_WRITER_H
#define VIDCRYPT_H264_WRITER_H

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Raw H.264 Bitstream Writer ────────────────────────────────────
 * Writes NVENC-encoded H.264 AVCC packets directly to a file.
 * No ffmpeg dependency — zero pipe/subprocess overhead.
 *
 * The output is a raw .h264 elementary stream with SPS/PPS at the
 * beginning, suitable for:
 *   - Direct playback with ffplay: ffplay output.h264
 *   - Muxing into container: ffmpeg -i output.h264 -c copy output.mkv
 *   - Any H.264 decoder that accepts raw Annex B or AVCC streams
 *
 * Output format: AVCC (4-byte length-prefixed NAL units).
 * SPS and PPS are written first as extradata, followed by frame data. */

typedef struct H264Writer H264Writer;

/* Create an H.264 writer.
 * 'path': output file path (should end in .h264).
 * 'extradata': SPS+PPS NAL units in AVCC format (4-byte length-prefixed).
 *              Written at the very beginning of the file.
 * 'extradata_size': size of extradata in bytes.
 * Returns NULL on failure. */
H264Writer* h264_writer_create(const char *path,
                                const uint8_t *extradata,
                                int extradata_size);

/* Write one H.264 NAL unit packet (AVCC format: 4-byte length prefix +
 * NAL unit data) to the output file.
 * Returns true on success. */
bool h264_writer_write_packet(H264Writer *w,
                               const uint8_t *data, int size);

/* Close the writer and flush all pending data.
 * Returns true on success. */
bool h264_writer_close(H264Writer *w);

/* Get total bytes written so far. */
int64_t h264_writer_bytes_written(H264Writer *w);

#ifdef __cplusplus
}
#endif

#endif /* VIDCRYPT_H264_WRITER_H */

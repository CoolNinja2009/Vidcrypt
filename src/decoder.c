#include "decoder.h"
#include "bitstream.h"
#include "reedsolomon.h"
#include "simd_decode.h"
#include "threadpool.h"
#include "ffmpeg_pipe.h"
#include "sha256.h"
#include "profiling.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void decoder_config_defaults(DecoderConfig *config) {
    memset(config, 0, sizeof(DecoderConfig));
    config->num_workers = 1;
    config->output_dir[0] = '\0';
}

typedef struct {
    const uint8_t *gray;
    int            stride;
    DecodeGeometry geom;
    uint8_t       *bits;
    int            pay_bits;
} DecodeJob;

static void decoder_worker_func(WorkItem *item) {
    DecodeJob *job = (DecodeJob *)item->data;
#ifdef ENABLE_PROFILING
    int64_t t0 = prof_clock_ns();
#endif
    decode_payload_tiles(&job->geom, job->gray, job->stride,
                          job->bits, job->pay_bits);
#ifdef ENABLE_PROFILING
    int64_t elapsed = prof_clock_ns() - t0;
    ProfCtx *pctx = (ProfCtx *)item->user_data;
    if (pctx) prof_worker_add(pctx, PROF_DECODE_BIT_EXTRACT, elapsed);
#endif
}

bool decoder_parse_header(const uint8_t *payload_bits, int nbits,
                          const CalParams *params,
                          char *filename, int filename_size,
                          int64_t *orig_size,
                          uint8_t expected_checksum[32],
                          char *error_msg, int error_msg_size) {
    uint8_t raw[512];
    int nbytes = (int)bits_to_bytes(payload_bits, (size_t)nbits, raw, 512);

    int ecc = (int)params->rs_ecc_symbols;
    if (ecc <= 0) ecc = 32;
    const RSCodec *codec = rs_get_codec(ecc);
    if (codec && nbytes >= (int)codec->block_length) {
        RSDecodeResult rs_result;
        rs_decode(codec, raw, &rs_result);
        memcpy(raw, rs_result.decoded, codec->msg_length);
        nbytes = (int)codec->msg_length;
    }

    if (nbytes < 6) { snprintf(error_msg, (size_t)error_msg_size, "Header too short"); return false; }

    int version = raw[0];
    if (version != 2 && version != 3) { snprintf(error_msg, (size_t)error_msg_size, "Unknown header version %d", version); return false; }
    if (memcmp(raw + 1, "LSY1", 4) != 0) { snprintf(error_msg, (size_t)error_msg_size, "Not a LSY1-encoded video"); return false; }

    int fname_len = raw[5];
    if (fname_len > 200 || (size_t)(6 + fname_len + 40) > (size_t)nbytes) { snprintf(error_msg, (size_t)error_msg_size, "Header truncated"); return false; }

    memcpy(filename, raw + 6, (size_t)fname_len);
    filename[fname_len] = '\0';

    int off = 6 + fname_len;
    uint64_t size = 0;
    for (int i = 0; i < 8; ++i) size = (size << 8) | raw[off + i];
    *orig_size = (int64_t)size;
    memcpy(expected_checksum, raw + off + 8, 32);
    return true;
}

bool decoder_decode_file(const char *input_path, const DecoderConfig *config,
                         DecoderResult *result, char *error_msg, int error_msg_size) {
    memset(result, 0, sizeof(DecoderResult));
    double start_time = (double)clock() / CLOCKS_PER_SEC;
    ProfCtx prof_ctx;
    (void)prof_ctx;

    /* Open video reader (ffmpeg pipe, gray8 rawvideo) */
    prof_stage_begin(&prof_ctx, PROF_DECODE_VIDEO_OPEN);
    VideoReader *vr = video_reader_open(input_path, NULL);
    prof_stage_end(&prof_ctx, PROF_DECODE_VIDEO_OPEN);
    if (!vr) {
        snprintf(error_msg, (size_t)error_msg_size, "Cannot open video file: %s", ffmpeg_last_error());
        return false;
    }

    /* Read first frame */
    prof_stage_begin(&prof_ctx, PROF_DECODE_FIRST_FRAME);
    int vw, vh, vstride;
    const uint8_t *first_frame = NULL;
    bool first_ok = video_reader_read_frame(vr, &first_frame, &vstride, &vw, &vh);
    prof_stage_end(&prof_ctx, PROF_DECODE_FIRST_FRAME);
    if (!first_ok) { video_reader_close(vr); snprintf(error_msg, (size_t)error_msg_size, "Cannot read first frame"); return false; }

    /* Extract calibration directly from grayscale frame */
    prof_stage_begin(&prof_ctx, PROF_DECODE_CALIBRATION);
    CalParams params;
    bool is_new_format = extract_calibration(first_frame, vw, vh, &params);
    prof_stage_end(&prof_ctx, PROF_DECODE_CALIBRATION);

    /* If video resolution differs from original encode (e.g., YouTube re-encode),
     * reopen ffmpeg with -vf scale to upscale to the original resolution.
     * This ensures tile geometry positions match exactly, without needing
     * adaptive geometry scaling. ffmpeg's SIMD bilayer is very fast. */
    if (vw != (int)params.frame_width || vh != (int)params.frame_height) {
        prof_stage_begin(&prof_ctx, PROF_DECODE_VIDEO_OPEN);
        video_reader_close(vr);
        vr = video_reader_open_scaled(input_path, NULL,
                                       (int)params.frame_width,
                                       (int)params.frame_height);
        prof_stage_end(&prof_ctx, PROF_DECODE_VIDEO_OPEN);
        if (!vr) {
            snprintf(error_msg, (size_t)error_msg_size,
                     "Cannot reopen video with scale filter: %s", ffmpeg_last_error());
            return false;
        }
        /* Read first frame again, now at original resolution */
        prof_stage_begin(&prof_ctx, PROF_DECODE_FIRST_FRAME);
        first_ok = video_reader_read_frame(vr, &first_frame, &vstride, &vw, &vh);
        prof_stage_end(&prof_ctx, PROF_DECODE_FIRST_FRAME);
        if (!first_ok) {
            video_reader_close(vr);
            snprintf(error_msg, (size_t)error_msg_size,
                     "Cannot read first frame after rescale");
            return false;
        }
        /* Re-extract calibration from the upscaled frame — the calibration
         * dots are clearer at the original resolution, so the CRC should pass.
         * The previous extraction (from the downscaled YouTube frame) may have
         * failed or had bit errors. */
        prof_stage_begin(&prof_ctx, PROF_DECODE_CALIBRATION);
        is_new_format = extract_calibration(first_frame, vw, vh, &params);
        prof_stage_end(&prof_ctx, PROF_DECODE_CALIBRATION);
    }

    DecodeGeometry geom;
    decode_geometry_init(&geom, &params, is_new_format);
    int pay_bits = geom.pay_rows * geom.grid_cols;


    /* Decode header bits from first frame */
    uint8_t *header_bits = (uint8_t *)malloc((size_t)pay_bits);
    if (!header_bits) { video_reader_close(vr); snprintf(error_msg, (size_t)error_msg_size, "Out of memory"); return false; }

    /* Validate sync pattern. For upscaled videos (YouTube re-encode), the
     * tile boundaries are blurred by the resize, so adjacent tiles' boundary
     * pixels are interpolated. The strict 100% sync check would fail because
     * samples at the top-left corner of each sync tile hit the blurred
     * boundary. Instead, use a relaxed check: count matching tiles and
     * require > 50% to pass. Native resolution videos pass easily (>95%). */
    prof_stage_begin(&prof_ctx, PROF_DECODE_HEADER_PARSE);
    int sync_matches = 0, sync_total = 0;
    int sub = geom.subsample;
    int total_sync_samples = (geom.block_size / sub) * (geom.block_size / sub);
    if (total_sync_samples < 1) total_sync_samples = 1;
    for (int col = 0; col < geom.grid_cols; ++col) {
        int x = geom.sync_x_start + col * geom.block_size;
        int count = tile_count_white(first_frame + geom.sync_y * vstride + x,
                                      vstride,
                                      geom.block_size, geom.block_size, sub);
        int bit = (count > total_sync_samples / 2) ? 1 : 0;
        if (bit == (col % 2)) sync_matches++;
        sync_total++;
    }
    bool sync_ok = sync_total > 0 && (double)sync_matches / sync_total > 0.5;
    prof_stage_end(&prof_ctx, PROF_DECODE_HEADER_PARSE);

    prof_stage_begin(&prof_ctx, PROF_DECODE_BIT_EXTRACT);
    int nbits = decode_payload_tiles(&geom, first_frame, vstride, header_bits, pay_bits);
    prof_stage_end(&prof_ctx, PROF_DECODE_BIT_EXTRACT);

    if (!sync_ok) {
        free(header_bits); video_reader_close(vr);
        snprintf(error_msg, (size_t)error_msg_size, "Sync pattern not found"); return false;
    }

    prof_stage_begin(&prof_ctx, PROF_DECODE_HEADER_PARSE);
    char filename[256];
    int64_t orig_size = 0;
    uint8_t expected_checksum[32];
    bool header_ok = decoder_parse_header(header_bits, nbits, &params,
                      filename, sizeof(filename), &orig_size, expected_checksum,
                      error_msg, error_msg_size);
    prof_stage_end(&prof_ctx, PROF_DECODE_HEADER_PARSE);
    if (!header_ok) { free(header_bits); video_reader_close(vr); return false; }

    char output_path[2048];
    if (config->output_dir[0])
        snprintf(output_path, sizeof(output_path), "%s/%s", config->output_dir, filename);
    else
        snprintf(output_path, sizeof(output_path), "%s", filename);

    int rs_ecc = (int)params.rs_ecc_symbols;
    int pay_bytes = pay_bits / 8;
    int chunk_bits = (rs_ecc > 0) ? ((int)params.rs_data_bytes + rs_ecc) * 8 : pay_bits;

    int64_t written = 0;
    int bit_buffer_cap = 65536;
    uint8_t *bit_buffer = (uint8_t *)malloc((size_t)bit_buffer_cap);
    int bit_buffer_len = 0;
    if (!bit_buffer) { video_reader_close(vr); snprintf(error_msg, (size_t)error_msg_size, "Out of memory"); return false; }

    FILE *out = fopen(output_path, "wb");
    if (!out) {
        free(bit_buffer); free(header_bits); video_reader_close(vr);
        snprintf(error_msg, (size_t)error_msg_size, "Cannot open output file"); return false;
    }

    int frame_count = 0;
    int rs_failures = 0;

    /* Thread pool */
    int num_workers = config->num_workers;
    if (num_workers <= 0) num_workers = 1;

    ThreadPool *pool = threadpool_create(num_workers, decoder_worker_func);
    if (!pool) {
        fclose(out); free(bit_buffer); free(header_bits); video_reader_close(vr);
        snprintf(error_msg, (size_t)error_msg_size, "Failed to create thread pool"); return false;
    }

    int frames_per_chunk = num_workers * 4;
    if (frames_per_chunk < 8) frames_per_chunk = 8;
    if (frames_per_chunk > 128) frames_per_chunk = 128;

    size_t gray_chunk_size = (size_t)frames_per_chunk * (size_t)vh * (size_t)vstride;
    size_t bits_chunk_size = (size_t)frames_per_chunk * (size_t)pay_bits;

    uint8_t *gray_chunk = (uint8_t *)malloc(gray_chunk_size);
    uint8_t *bits_chunk = (uint8_t *)calloc(1, bits_chunk_size);
    DecodeJob *jobs = (DecodeJob *)calloc((size_t)frames_per_chunk, sizeof(DecodeJob));
    if (!gray_chunk || !bits_chunk || !jobs) {
        free(gray_chunk); free(bits_chunk); free(jobs);
        threadpool_destroy(pool); fclose(out); free(bit_buffer);
        free(header_bits); video_reader_close(vr);
        snprintf(error_msg, (size_t)error_msg_size, "Out of memory"); return false;
    }

    /* Pre-allocate decoded buffer (avoids malloc/free per chunk in inner loop) */
    int decoded_buf_size = (rs_ecc > 0) ? 255 : pay_bytes;
    uint8_t *decoded_buf = (uint8_t *)malloc((size_t)decoded_buf_size);
    if (!decoded_buf) {
        free(gray_chunk); free(bits_chunk); free(jobs);
        threadpool_destroy(pool); fclose(out); free(bit_buffer);
        free(header_bits); video_reader_close(vr);
        snprintf(error_msg, (size_t)error_msg_size, "Out of memory"); return false;
    }

    /* --- Sequential reader + parallel decoder pipeline --- */
    /* In gray8 mode, the ffmpeg pipe already delivers raw grayscale frames
     * at the target resolution. No normalize or grayscale conversion needed. */
    const uint8_t *frame_data;
    int fw, fh, fstride;
    int chunk_idx = 0;

    while (1) {
        prof_stage_begin(&prof_ctx, PROF_DECODE_FRAME_ACQUIRE);
        bool has_frame = video_reader_read_frame(vr, &frame_data, &fstride, &fw, &fh);
        prof_stage_end(&prof_ctx, PROF_DECODE_FRAME_ACQUIRE);
        if (!has_frame) break;

        /* Copy grayscale frame directly into chunk buffer */
        uint8_t *gray_buf = gray_chunk + (size_t)chunk_idx * (size_t)fh * (size_t)fstride;
        for (int y = 0; y < fh; ++y)
            memcpy(gray_buf + y * fstride, frame_data + y * fstride, (size_t)fstride);

        /* Set up decode job */
        jobs[chunk_idx].gray     = gray_buf;
        jobs[chunk_idx].stride   = fstride;
        jobs[chunk_idx].geom     = geom;
        jobs[chunk_idx].bits     = bits_chunk + (size_t)chunk_idx * (size_t)pay_bits;
        jobs[chunk_idx].pay_bits = pay_bits;
        chunk_idx++;

        if (chunk_idx >= frames_per_chunk) {
            for (int i = 0; i < chunk_idx; ++i) {
                WorkItem item;
                memset(&item, 0, sizeof(item));
                item.data = &jobs[i];
                item.data_size = 0;
                item.user_data = (void*)&prof_ctx;
                threadpool_submit(pool, &item);
            }
            threadpool_wait(pool);

            for (int i = 0; i < chunk_idx; ++i) {
                if (bit_buffer_len + pay_bits > bit_buffer_cap) {
                    bit_buffer_cap = (bit_buffer_len + pay_bits) * 2;
                    uint8_t *new_buf = (uint8_t *)realloc(bit_buffer, (size_t)bit_buffer_cap);
                    if (!new_buf) goto flush_remaining;
                    bit_buffer = new_buf;
                }
                memcpy(bit_buffer + bit_buffer_len, jobs[i].bits, (size_t)pay_bits);
                bit_buffer_len += pay_bits;
                frame_count++;

                const RSCodec *codec = rs_ecc > 0 ? rs_get_codec(rs_ecc) : NULL;
                while (bit_buffer_len >= chunk_bits && written < orig_size) {
                    uint8_t raw_block[255];
                    bits_to_bytes(bit_buffer, (size_t)chunk_bits, raw_block, 255);
                    int data_bytes;
                    if (codec) {
                        RSDecodeResult rs_res;
                        prof_stage_begin(&prof_ctx, PROF_DECODE_RS_DECODE);
                        rs_decode(codec, raw_block, &rs_res);
                        prof_stage_end(&prof_ctx, PROF_DECODE_RS_DECODE);
                        if (rs_res.status < 0) {
                            rs_failures++;
                            int msg_len = (int)codec->msg_length;
                            memcpy(decoded_buf, raw_block, (size_t)msg_len);
                            data_bytes = msg_len;
                        } else {
                            memcpy(decoded_buf, rs_res.decoded, codec->msg_length);
                            data_bytes = (int)codec->msg_length;
                            if (rs_res.corrected > 0) rs_failures++;
                        }
                    } else {
                        data_bytes = (int)bits_to_bytes(bit_buffer, (size_t)pay_bits, decoded_buf, (size_t)pay_bytes);
                    }
                    int64_t remaining = orig_size - written;
                    int to_write = (int)((int64_t)data_bytes < remaining ? data_bytes : remaining);
                    prof_stage_begin(&prof_ctx, PROF_DECODE_FILE_WRITE);
                    fwrite(decoded_buf, 1, (size_t)to_write, out);
                    prof_stage_end(&prof_ctx, PROF_DECODE_FILE_WRITE);
                    written += to_write;
                    memmove(bit_buffer, bit_buffer + chunk_bits, (size_t)(bit_buffer_len - chunk_bits));
                    bit_buffer_len -= chunk_bits;
                    if (config->progress_callback)
                        config->progress_callback(written, orig_size, config->progress_user_data);
                }
            }
            chunk_idx = 0;
        }
    }

    /* Process remaining frames */
    if (chunk_idx > 0) {
        for (int i = 0; i < chunk_idx; ++i) {
            WorkItem item;
            memset(&item, 0, sizeof(item));
            item.data = &jobs[i];
            item.data_size = 0;
            item.user_data = (void*)&prof_ctx;
            threadpool_submit(pool, &item);
        }
        threadpool_wait(pool);

        for (int i = 0; i < chunk_idx; ++i) {
            if (bit_buffer_len + pay_bits > bit_buffer_cap) {
                bit_buffer_cap = (bit_buffer_len + pay_bits) * 2;
                uint8_t *new_buf = (uint8_t *)realloc(bit_buffer, (size_t)bit_buffer_cap);
                if (!new_buf) goto flush_remaining;
                bit_buffer = new_buf;
            }
            memcpy(bit_buffer + bit_buffer_len, jobs[i].bits, (size_t)pay_bits);
            bit_buffer_len += pay_bits;
            frame_count++;

            const RSCodec *codec = rs_ecc > 0 ? rs_get_codec(rs_ecc) : NULL;
            while (bit_buffer_len >= chunk_bits && written < orig_size) {
                uint8_t raw_block[255];
                bits_to_bytes(bit_buffer, (size_t)chunk_bits, raw_block, 255);
                int data_bytes;
                if (codec) {
                    RSDecodeResult rs_res;
                    prof_stage_begin(&prof_ctx, PROF_DECODE_RS_DECODE);
                    rs_decode(codec, raw_block, &rs_res);
                    prof_stage_end(&prof_ctx, PROF_DECODE_RS_DECODE);
                    if (rs_res.status < 0) {
                        rs_failures++;
                        int msg_len = (int)codec->msg_length;
                        memcpy(decoded_buf, raw_block, (size_t)msg_len);
                        data_bytes = msg_len;
                    } else {
                        memcpy(decoded_buf, rs_res.decoded, codec->msg_length);
                        data_bytes = (int)codec->msg_length;
                        if (rs_res.corrected > 0) rs_failures++;
                    }
                } else {
                    data_bytes = (int)bits_to_bytes(bit_buffer, (size_t)pay_bits, decoded_buf, (size_t)pay_bytes);
                }
                int64_t remaining = orig_size - written;
                int to_write = (int)((int64_t)data_bytes < remaining ? data_bytes : remaining);
                prof_stage_begin(&prof_ctx, PROF_DECODE_FILE_WRITE);
                fwrite(decoded_buf, 1, (size_t)to_write, out);
                prof_stage_end(&prof_ctx, PROF_DECODE_FILE_WRITE);
                written += to_write;
                memmove(bit_buffer, bit_buffer + chunk_bits, (size_t)(bit_buffer_len - chunk_bits));
                bit_buffer_len -= chunk_bits;
                if (config->progress_callback)
                    config->progress_callback(written, orig_size, config->progress_user_data);
            }
        }
    }

flush_remaining:
    if (written < orig_size && bit_buffer_len > 0) {
        uint8_t raw_block[255];
        bits_to_bytes(bit_buffer, (size_t)bit_buffer_len, raw_block, 255);
        int64_t remaining = orig_size - written;
        int to_write = (int)((int64_t)255 < remaining ? 255 : remaining);
        fwrite(raw_block, 1, (size_t)to_write, out);
        written += to_write;
    }

    /* Cleanup */
    threadpool_destroy(pool);
    free(gray_chunk);
    free(bits_chunk);
    free(jobs);
    fclose(out);
    free(bit_buffer);
    free(header_bits);
    free(decoded_buf);
    video_reader_close(vr);

    double end_time = (double)clock() / CLOCKS_PER_SEC;
    result->elapsed_sec = end_time - start_time;
    result->total_frames = frame_count;
    result->total_bytes_written = written;
    result->fps = (frame_count > 0 && result->elapsed_sec > 0.0) ? (double)frame_count / result->elapsed_sec : 0.0;
    result->rs_failures = rs_failures;
    snprintf(result->output_path, sizeof(result->output_path), "%s", output_path);
    snprintf(result->original_filename, sizeof(result->original_filename), "%s", filename);
    result->params = params;

    prof_stage_begin(&prof_ctx, PROF_DECODE_SHA256);
    uint8_t actual_checksum[32];
    sha256_file(output_path, actual_checksum);
    prof_stage_end(&prof_ctx, PROF_DECODE_SHA256);
    result->checksum_match = (memcmp(actual_checksum, expected_checksum, 32) == 0);

#ifdef ENABLE_PROFILING
    { ProfReport prof_report; prof_report_build(&prof_ctx, &prof_report); prof_report_print(&prof_report); }
#endif

    if (!result->checksum_match) {
        if (error_msg) snprintf(error_msg, (size_t)error_msg_size, "SHA-256 mismatch");
        return false;
    }
    return true;
}

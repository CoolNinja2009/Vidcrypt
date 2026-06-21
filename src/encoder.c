#include "encoder.h"
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
#include <pthread.h>

#if defined(_WIN32)
#define fseeko _fseeki64
#define ftello _ftelli64
#endif

void encoder_config_defaults(EncoderConfig *config) {
    memset(config, 0, sizeof(EncoderConfig));
    config->frame_width     = DEFAULT_FRAME_WIDTH;
    config->frame_height    = DEFAULT_FRAME_HEIGHT;
    config->margin_x        = DEFAULT_MARGIN_X;
    config->margin_y        = DEFAULT_MARGIN_Y;
    config->block_size      = DEFAULT_BLOCK_SIZE;
    config->rs_ecc_symbols  = DEFAULT_RS_ECC_SYMBOLS;
    config->fps             = 30.0;  /* default: 30 fps for higher density */
    config->num_workers     = 1;
    config->codec_name      = NULL;
}

uint8_t *encoder_build_header(const char *input_path, const CalParams *params,
                              size_t *out_len) {
    const char *fname = strrchr(input_path, '/');
    if (!fname) fname = strrchr(input_path, '\\');
    if (!fname) fname = input_path; else fname++;

    size_t fname_len = strlen(fname);
    if (fname_len > MAX_FILENAME_LEN) fname_len = MAX_FILENAME_LEN;

    FILE *f = fopen(input_path, "rb");
    if (!f) return NULL;
    fseeko(f, 0, SEEK_END);
    int64_t file_size = (int64_t)ftello(f);
    fclose(f);

    uint8_t sha256_hash[SHA256_DIGEST_LEN];
    sha256_file(input_path, sha256_hash);

    size_t header_size = (size_t)(1 + 4 + 1) + fname_len + (size_t)(8 + SHA256_DIGEST_LEN);
    uint8_t *header = (uint8_t *)malloc(header_size);
    if (!header) return NULL;

    size_t off = 0;
    header[off++] = params->header_version;
    memcpy(header + off, "LSY1", 4); off += 4;
    header[off++] = (uint8_t)fname_len;
    memcpy(header + off, fname, fname_len); off += fname_len;

    uint64_t size_be = (uint64_t)file_size;
    for (int i = 7; i >= 0; --i) {
        header[off + (size_t)i] = (uint8_t)(size_be & 0xFF);
        size_be >>= 8;
    }
    off += 8;
    memcpy(header + off, sha256_hash, SHA256_DIGEST_LEN);
    off += SHA256_DIGEST_LEN;
    *out_len = header_size;
    return header;
}

static bool compute_grid_params(const EncoderConfig *config, CalParams *params,
                                char *error_msg, int error_msg_size) {
    int cal_bottom = (int)((float)config->frame_height * (CAL_TOP_FRAC + CAL_HEIGHT_FRAC));
    int grid_cols = (config->frame_width - 2 * config->margin_x) / config->block_size;
    int grid_rows = (config->frame_height - cal_bottom - 2 * config->margin_y) / config->block_size;

    if (grid_cols < 4) { snprintf(error_msg, (size_t)error_msg_size, "Grid too narrow: %d cols", grid_cols); return false; }
    if (grid_rows < 4) { snprintf(error_msg, (size_t)error_msg_size, "Grid too short: %d rows", grid_rows); return false; }

    int sync_rows = 1;
    int payload_rows = grid_rows - sync_rows;
    int payload_bits = payload_rows * grid_cols;
    int payload_bytes = payload_bits / 8;
    if (payload_bytes < 1) { snprintf(error_msg, (size_t)error_msg_size, "Resolution too small"); return false; }

    memset(params, 0, sizeof(CalParams));
    params->frame_width      = (uint16_t)config->frame_width;
    params->frame_height     = (uint16_t)config->frame_height;
    params->margin_x         = (uint16_t)config->margin_x;
    params->margin_y         = (uint16_t)config->margin_y;
    params->block_size_x     = (uint16_t)config->block_size;
    params->block_size_y     = (uint16_t)config->block_size;
    params->grid_cols        = (uint16_t)grid_cols;
    params->grid_rows        = (uint16_t)grid_rows;
    params->rs_ecc_symbols   = (uint8_t)config->rs_ecc_symbols;
    params->rs_data_bytes    = (uint8_t)(config->rs_ecc_symbols > 0 ? 255 - config->rs_ecc_symbols : (uint8_t)payload_bytes);
    params->header_version   = HEADER_VERSION;
    params->calibration_rows = CAL_ROWS;
    params->sync_rows        = (uint8_t)sync_rows;
    return true;
}

typedef struct {
    PrecomputedFrame pf;
    int worker_id;
} FrameGenContext;

typedef struct {
    uint8_t *frame_bits;
    uint8_t *frame_out;
    int      seq;
    int      worker_id;
} EncoderJob;

static struct {
    FrameGenContext *gen_ctx;
    int frame_size;
    int pay_bits;
} g_encoder_ctx;

static void encoder_worker_func(WorkItem *item) {
    EncoderJob *job = (EncoderJob *)item->data;
    FrameGenContext *ctx = &g_encoder_ctx.gen_ctx[job->worker_id];
    uint8_t *frame = precomputed_frame_generate(&ctx->pf, job->frame_bits);
    memcpy(job->frame_out, frame, (size_t)g_encoder_ctx.frame_size);
}

bool encoder_encode_file(const char *input_path, const EncoderConfig *config,
                         EncoderResult *result, char *error_msg, int error_msg_size) {
    memset(result, 0, sizeof(EncoderResult));
    double start_time = (double)clock() / CLOCKS_PER_SEC;
    ProfCtx prof_ctx;
    (void)prof_ctx;

    CalParams params;
    if (!compute_grid_params(config, &params, error_msg, error_msg_size)) return false;


    int grid_cols    = (int)params.grid_cols;
    int grid_rows    = (int)params.grid_rows;
    int sync_rows    = (int)params.sync_rows;
    int pay_rows     = grid_rows - sync_rows;
    int pay_bits     = pay_rows * grid_cols;
    int pay_bytes    = pay_bits / 8;

    int effective_ecc = config->rs_ecc_symbols;
    int chunk_size;
    if (effective_ecc > 0) {
        const RSCodec *codec = rs_get_codec(effective_ecc);
        if (!codec) { snprintf(error_msg, (size_t)error_msg_size, "Failed to init RS codec"); return false; }
        chunk_size = (int)codec->msg_length;
    } else {
        chunk_size = pay_bytes;
    }

    FILE *in = fopen(input_path, "rb");
    if (!in) { snprintf(error_msg, (size_t)error_msg_size, "Cannot open input file"); return false; }
    fseeko(in, 0, SEEK_END);
    int64_t total_bytes = (int64_t)ftello(in);
    fseeko(in, 0, SEEK_SET);

    int n_workers = config->num_workers;
    if (n_workers <= 0) n_workers = 4;

    FrameGenContext *gen_ctx = (FrameGenContext *)calloc((size_t)n_workers, sizeof(FrameGenContext));
    if (!gen_ctx) { fclose(in); snprintf(error_msg, (size_t)error_msg_size, "Out of memory"); return false; }
    for (int i = 0; i < n_workers; ++i)
        if (!precomputed_frame_init(&gen_ctx[i].pf, &params)) {
            for (int j = 0; j < i; ++j) precomputed_frame_destroy(&gen_ctx[j].pf);
            free(gen_ctx); fclose(in);
            snprintf(error_msg, (size_t)error_msg_size, "Failed to allocate frame generator"); return false;
        }

    char output_path[1024];
    snprintf(output_path, sizeof(output_path), "%s", config->output_path);
    size_t olen = strlen(output_path);
    if (olen < 4 || (output_path[olen-4] != '.')) {
        snprintf(output_path + olen, sizeof(output_path) - olen, ".avi");
    }

    VideoWriter *vw = video_writer_create(NULL, output_path,
        (int)params.frame_width, (int)params.frame_height,
        config->fps, config->codec_name);
    if (!vw) {
        for (int i = 0; i < n_workers; ++i) precomputed_frame_destroy(&gen_ctx[i].pf);
        free(gen_ctx); fclose(in);
        snprintf(error_msg, (size_t)error_msg_size, "Failed to create video writer: %s", ffmpeg_last_error());
        return false;
    }

    /* Build and encode header frame */
    size_t header_raw_len;
    prof_stage_begin(&prof_ctx, PROF_ENCODE_HEADER_BUILD);
    uint8_t *header_raw = encoder_build_header(input_path, &params, &header_raw_len);
    prof_stage_end(&prof_ctx, PROF_ENCODE_HEADER_BUILD);
    if (!header_raw) {
        video_writer_close(vw);
        for (int i = 0; i < n_workers; ++i) precomputed_frame_destroy(&gen_ctx[i].pf);
        free(gen_ctx); fclose(in);
        snprintf(error_msg, (size_t)error_msg_size, "Failed to build header"); return false;
    }

    int header_ecc = effective_ecc > 0 ? effective_ecc : 32;
    const RSCodec *header_codec = rs_get_codec(header_ecc);
    uint8_t header_enc[255];
    int header_enc_len;
    if (header_codec) {
        prof_stage_begin(&prof_ctx, PROF_ENCODE_RS_ENCODE);
        header_enc_len = rs_encode(header_codec, header_raw, (int)header_raw_len, header_enc);
        prof_stage_end(&prof_ctx, PROF_ENCODE_RS_ENCODE);
    } else {
        memcpy(header_enc, header_raw, header_raw_len);
        header_enc_len = (int)header_raw_len;
    }

    uint8_t *header_bits = (uint8_t *)calloc((size_t)pay_bits, 1);
    int header_nbits = header_enc_len * 8;
    if (header_nbits > pay_bits) header_nbits = pay_bits;
    for (int i = 0; i < header_nbits; ++i)
        header_bits[i] = (uint8_t)((header_enc[i / 8] >> (7 - (i % 8))) & 1);


    prof_stage_begin(&prof_ctx, PROF_ENCODE_FRAME_GENERATE);
    uint8_t *header_frame = precomputed_frame_generate(&gen_ctx[0].pf, header_bits);
    prof_stage_end(&prof_ctx, PROF_ENCODE_FRAME_GENERATE);

    prof_stage_begin(&prof_ctx, PROF_ENCODE_VIDEO_WRITE);
    bool hdr_write_ok = video_writer_write_frame(vw, header_frame, (int)params.frame_width);
    prof_stage_end(&prof_ctx, PROF_ENCODE_VIDEO_WRITE);
    if (!hdr_write_ok) {
        video_writer_close(vw); free(header_bits); free(header_raw);
        for (int i = 0; i < n_workers; ++i) precomputed_frame_destroy(&gen_ctx[i].pf);
        free(gen_ctx); fclose(in);
        snprintf(error_msg, (size_t)error_msg_size, "Failed to write header frame"); return false;
    }
    result->total_frames++;
    free(header_bits);
    free(header_raw);

    /* Streaming encode pipeline */
    int64_t written = 0;
    int bit_buffer_len = 0;
    int batch_size = n_workers * 4;
    if (batch_size < 8) batch_size = 8;
    if (batch_size > 64) batch_size = 64;

    int bit_buf_cap = batch_size * pay_bits;
    uint8_t *bit_buffer = (uint8_t *)malloc((size_t)bit_buf_cap);
    if (!bit_buffer) {
        video_writer_close(vw);
        for (int i = 0; i < n_workers; ++i) precomputed_frame_destroy(&gen_ctx[i].pf);
        free(gen_ctx); fclose(in);
        snprintf(error_msg, (size_t)error_msg_size, "Out of memory"); return false;
    }

    int frame_size = (int)params.frame_width * (int)params.frame_height; /* gray8 */
    EncoderJob *jobs = (EncoderJob *)calloc((size_t)batch_size, sizeof(EncoderJob));
    uint8_t *batch_bits   = (uint8_t *)malloc((size_t)batch_size * (size_t)pay_bits);
    uint8_t *batch_frames = (uint8_t *)malloc((size_t)batch_size * (size_t)frame_size);
    if (!jobs || !batch_bits || !batch_frames) {
        free(jobs); free(batch_bits); free(batch_frames);
        free(bit_buffer); video_writer_close(vw);
        for (int i = 0; i < n_workers; ++i) precomputed_frame_destroy(&gen_ctx[i].pf);
        free(gen_ctx); fclose(in);
        snprintf(error_msg, (size_t)error_msg_size, "Out of memory"); return false;
    }

    g_encoder_ctx.gen_ctx    = gen_ctx;
    g_encoder_ctx.frame_size = frame_size;
    g_encoder_ctx.pay_bits   = pay_bits;

    ThreadPool *pool = threadpool_create(n_workers, encoder_worker_func);
    if (!pool) {
        free(jobs); free(batch_bits); free(batch_frames);
        free(bit_buffer); video_writer_close(vw);
        for (int i = 0; i < n_workers; ++i) precomputed_frame_destroy(&gen_ctx[i].pf);
        free(gen_ctx); fclose(in);
        snprintf(error_msg, (size_t)error_msg_size, "Failed to create thread pool"); return false;
    }

    uint8_t *chunk_data = (uint8_t *)malloc((size_t)chunk_size);
    if (!chunk_data) {
        threadpool_destroy(pool); free(jobs); free(batch_bits); free(batch_frames);
        free(bit_buffer); video_writer_close(vw);
        for (int i = 0; i < n_workers; ++i) precomputed_frame_destroy(&gen_ctx[i].pf);
        free(gen_ctx); fclose(in);
        snprintf(error_msg, (size_t)error_msg_size, "Out of memory"); return false;
    }

    while (1) {
        prof_stage_begin(&prof_ctx, PROF_ENCODE_FILE_READ);
        int nread = (int)fread(chunk_data, 1, (size_t)chunk_size, in);
        prof_stage_end(&prof_ctx, PROF_ENCODE_FILE_READ);
        if (nread <= 0) break;

        uint8_t enc_block[255];
        int enc_len;
        if (effective_ecc > 0 && header_codec) {
            int msg_len = (int)header_codec->msg_length;
            uint8_t *padded = (uint8_t *)calloc((size_t)msg_len, 1);
            if (!padded) {
                fclose(in); free(chunk_data);
                threadpool_destroy(pool); free(jobs); free(batch_bits); free(batch_frames);
                free(bit_buffer); video_writer_close(vw);
                for (int i = 0; i < n_workers; ++i) precomputed_frame_destroy(&gen_ctx[i].pf);
                free(gen_ctx);
                snprintf(error_msg, (size_t)error_msg_size, "Out of memory"); return false;
            }
            memcpy(padded, chunk_data, (size_t)nread);
            prof_stage_begin(&prof_ctx, PROF_ENCODE_RS_ENCODE);
            enc_len = rs_encode(header_codec, padded, msg_len, enc_block);
            prof_stage_end(&prof_ctx, PROF_ENCODE_RS_ENCODE);
            free(padded);
        } else {
            memcpy(enc_block, chunk_data, (size_t)nread);
            enc_len = nread;
        }

        int enc_bits = enc_len * 8;
        if (bit_buffer_len + enc_bits > bit_buf_cap) {
            while (bit_buffer_len >= pay_bits) {
                int frames_avail = bit_buffer_len / pay_bits;
                int count = frames_avail < batch_size ? frames_avail : batch_size;

                for (int i = 0; i < count; ++i) {
                    memcpy(batch_bits + (size_t)i * (size_t)pay_bits,
                           bit_buffer + (size_t)i * (size_t)pay_bits, (size_t)pay_bits);
                    jobs[i].frame_bits = batch_bits + (size_t)i * (size_t)pay_bits;
                    jobs[i].frame_out  = batch_frames + (size_t)i * (size_t)frame_size;
                    jobs[i].worker_id  = (result->total_frames + i) % n_workers;
                    jobs[i].seq        = result->total_frames + i;
                }

                prof_stage_begin(&prof_ctx, PROF_ENCODE_FRAME_GENERATE);
                for (int i = 0; i < count; ++i) {
                    WorkItem item;
                    memset(&item, 0, sizeof(item));
                    item.data = &jobs[i];
                    item.data_size = 0;
                    threadpool_submit(pool, &item);
                }
                threadpool_wait(pool);
                prof_stage_end(&prof_ctx, PROF_ENCODE_FRAME_GENERATE);

                prof_stage_begin(&prof_ctx, PROF_ENCODE_VIDEO_WRITE);
                for (int i = 0; i < count; ++i) {
                    video_writer_write_frame(vw, jobs[i].frame_out, frame_size);
                    result->total_frames++;
                }
                prof_stage_end(&prof_ctx, PROF_ENCODE_VIDEO_WRITE);

                int consumed = count * pay_bits;
                memmove(bit_buffer, bit_buffer + consumed, (size_t)(bit_buffer_len - consumed));
                bit_buffer_len -= consumed;

                if (config->progress_callback)
                    config->progress_callback(written, total_bytes, config->progress_user_data);
            }
        }

        for (int i = 0; i < enc_bits; ++i)
            bit_buffer[bit_buffer_len++] = (uint8_t)((enc_block[i / 8] >> (7 - (i % 8))) & 1);
        written += nread;

        if (config->progress_callback)
            config->progress_callback(written, total_bytes, config->progress_user_data);
    }
    fclose(in);
    free(chunk_data);

    /* Flush remaining complete frames */
    while (bit_buffer_len >= pay_bits) {
        int frames_avail = bit_buffer_len / pay_bits;
        int count = frames_avail < batch_size ? frames_avail : batch_size;

        for (int i = 0; i < count; ++i) {
            memcpy(batch_bits + (size_t)i * (size_t)pay_bits,
                   bit_buffer + (size_t)i * (size_t)pay_bits, (size_t)pay_bits);
            jobs[i].frame_bits = batch_bits + (size_t)i * (size_t)pay_bits;
            jobs[i].frame_out  = batch_frames + (size_t)i * (size_t)frame_size;
            jobs[i].worker_id  = (result->total_frames + i) % n_workers;
            jobs[i].seq        = result->total_frames + i;
        }

        prof_stage_begin(&prof_ctx, PROF_ENCODE_FRAME_GENERATE);
        for (int i = 0; i < count; ++i) {
            WorkItem item;
            memset(&item, 0, sizeof(item));
            item.data = &jobs[i];
            item.data_size = 0;
            threadpool_submit(pool, &item);
        }
        threadpool_wait(pool);
        prof_stage_end(&prof_ctx, PROF_ENCODE_FRAME_GENERATE);

        prof_stage_begin(&prof_ctx, PROF_ENCODE_VIDEO_WRITE);
        for (int i = 0; i < count; ++i) {
            video_writer_write_frame(vw, jobs[i].frame_out, frame_size);
            result->total_frames++;
        }
        prof_stage_end(&prof_ctx, PROF_ENCODE_VIDEO_WRITE);

        int consumed = count * pay_bits;
        memmove(bit_buffer, bit_buffer + consumed, (size_t)(bit_buffer_len - consumed));
        bit_buffer_len -= consumed;

        if (config->progress_callback)
            config->progress_callback(written, total_bytes, config->progress_user_data);
    }

    /* Flush remaining partial frame */
    if (bit_buffer_len > 0 || result->total_frames == 0) {
        for (int i = bit_buffer_len; i < pay_bits; ++i) bit_buffer[i] = 0;
        int worker_idx = result->total_frames % n_workers;
        uint8_t *frame = precomputed_frame_generate(&gen_ctx[worker_idx].pf, bit_buffer);
        if (video_writer_write_frame(vw, frame, frame_size)) result->total_frames++;
    }

    threadpool_destroy(pool);
    free(jobs);
    free(batch_bits);
    free(batch_frames);
    free(bit_buffer);
    video_writer_close(vw);

    for (int i = 0; i < n_workers; ++i) precomputed_frame_destroy(&gen_ctx[i].pf);
    free(gen_ctx);

    double end_time = (double)clock() / CLOCKS_PER_SEC;
    result->elapsed_sec = end_time - start_time;
    result->fps = (double)result->total_frames / result->elapsed_sec;
    result->total_bytes_written = total_bytes;
    snprintf(result->output_path, sizeof(result->output_path), "%s", output_path);
    result->params = params;

#ifdef ENABLE_PROFILING
    { ProfReport prof_report; prof_report_build(&prof_ctx, &prof_report); prof_report_print(&prof_report); }
#endif

    return true;
}

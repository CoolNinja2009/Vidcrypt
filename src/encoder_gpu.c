/* ═══════════════════════════════════════════════════════════════════════════
 * GPU-accelerated encode pipeline
 *
 * Implements encoder_encode_file_gpu() using:
 *   1. Tiled GPU RS encoder (shared-memory, warp-shuffle reduction)
 *   2. Double-buffered streaming pipeline (overlaps RS + NVENC)
 *   3. Zero-copy NVENC hardware encode (registered CUDA resource)
 *   4. H.264 bitstream writer (.h264 raw output, remuxable to .mkv)
 *
 * Architecture:
 *   File → Read chunk → GPU RS encode → Bit expand → Frame gen → NVENC → Write
 *          ↑                         ↑              ↑            ↑        ↑
 *        (CPU)               (stream_rs)    (stream_fg)   (NVENC)   (CPU)
 *
 * The double-buffer pipeline keeps the CUDA SMs and NVENC ASIC concurrently
 * busy: while frame N is being NVENC-encoded, frame N+1 is being generated,
 * and chunk N+2 is being RS-encoded.
 *
 * Key optimizations over the old monolithic encoder:
 *   - Tiled RS kernel: 8 codewords/block × 32 threads each = 256 threads/block
 *     vs old 1 codeword/block × 32 threads (8× more work per block)
 *   - Warp shuffle reduction for parity XOR (eliminates global atomics)
 *   - Streaming file I/O: no full-file memory load
 *   - Double-buffered GPU memory: RS stage runs concurrently with NVENC
 *   - Single CUDA stream for all GPU work (avoids context-switch overhead
 *     for our workload; NVENC has its own independent hardware pipeline)
 * ═══════════════════════════════════════════════════════════════════════════ */

#include "encoder.h"
#include "bitstream.h"
#include "reedsolomon.h"
#include "ffmpeg_pipe.h"
#include "gpu_backend.h"
#include "calibration.h"
#include "gpu_nvenc.h"
#include "gpu_encode_pipeline.h"
#include "h264_writer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#if defined(_WIN32)
#define fseeko _fseeki64
#define ftello _ftelli64
#include <windows.h>
#endif

/* ─── Default chunk size for streaming file reads ──────────────────────
 * Large enough to amortize I/O overhead but small enough to keep memory
 * bounded. 64 MB gives ~290K RS codewords per chunk (RS223,32),
 * producing ~2.3 billion bits → ~1000+ frames at typical grid density. */
#define STREAM_CHUNK_SIZE  ((int64_t)(64 * 1024 * 1024))

/* ─── Forward declarations ───────────────────────────────────────────── */

static bool encode_header_frame(const char *input_path,
                                 const EncoderConfig *config,
                                 CalParams *params,
                                 GpuBackend *backend,
                                 GpuNvencEncoder *nvenc,
                                 H264Writer **writer_out,
                                 int *frames_out,
                                 char *error_msg, int error_msg_size);

static bool stream_encode_rs(GpuEncodePipeline *pipeline,
                              FILE *in, int64_t total_bytes,
                              H264Writer *writer,
                              int *frames_out,
                              const EncoderConfig *config,
                              char *error_msg, int error_msg_size);

static bool stream_encode_raw(GpuEncodePipeline *pipeline,
                               FILE *in, int64_t total_bytes,
                               H264Writer *writer,
                               int *frames_out,
                               char *error_msg, int error_msg_size);

/* ═══════════════════════════════════════════════════════════════════════════
 * Main entry point: encoder_encode_file_gpu()
 * ═══════════════════════════════════════════════════════════════════════════ */

bool encoder_encode_file_gpu(const char *input_path,
                              const EncoderConfig *config,
                              EncoderResult *result,
                              char *error_msg, int error_msg_size) {
#ifdef USE_CUDA
    double start_time = (double)clock() / CLOCKS_PER_SEC;
    memset(result, 0, sizeof(EncoderResult));

    /* ── 1. Compute grid parameters ─────────────────────────────────── */
    CalParams params;
    if (!compute_grid_params(config, &params, error_msg, error_msg_size))
        return false;

    int grid_cols = (int)params.grid_cols;
    int pay_rows  = cal_params_payload_rows(&params);
    int pay_bits  = cal_params_payload_bits_per_frame(&params);
    int pay_bytes = pay_bits / 8;

    if (pay_bytes < 1) {
        snprintf(error_msg, (size_t)error_msg_size,
                 "Payload too small (%d bytes)", pay_bytes);
        return false;
    }

    /* ── 2. Open input file ─────────────────────────────────────────── */
    FILE *in = fopen(input_path, "rb");
    if (!in) {
        snprintf(error_msg, (size_t)error_msg_size, "Cannot open input");
        return false;
    }
    fseeko(in, 0, SEEK_END);
    int64_t total_bytes = (int64_t)ftello(in);
    fseeko(in, 0, SEEK_SET);

    /* ── 3. Determine RS codec ──────────────────────────────────────── */
    int effective_ecc = config->rs_ecc_symbols;
    int k = pay_bytes;   /* no-RS: raw bytes per frame */
    int n_k = 0;
    const RSCodec *payload_codec = NULL;

    if (effective_ecc > 0) {
        payload_codec = rs_get_codec(effective_ecc);
        if (!payload_codec) {
            fclose(in);
            snprintf(error_msg, (size_t)error_msg_size, "RS codec init failed");
            return false;
        }
        k   = (int)payload_codec->msg_length;
        n_k = (int)payload_codec->ecc_symbols;
    }

    /* ── 4. Init GPU backend ────────────────────────────────────────── */
    GpuBackend *backend = NULL;
    if (gpu_backend_init(&backend, 0, error_msg, error_msg_size) != GPU_INIT_OK) {
        fclose(in); return false;
    }

    /* ── 5. Init NVENC encoder ──────────────────────────────────────── */
    char nvenc_err[256] = {0};
    if (!gpu_backend_open_nvenc(backend,
            (int)params.frame_width, (int)params.frame_height,
            config->fps, 50000, (int)(config->fps * 2),
            nvenc_err, sizeof(nvenc_err))) {
        gpu_backend_destroy(backend); fclose(in);
        snprintf(error_msg, (size_t)error_msg_size,
                 "NVENC init failed: %s", nvenc_err);
        return false;
    }
    GpuNvencEncoder *nvenc = gpu_backend_get_nvenc_encoder(backend);

    /* ── 6. Determine output path ───────────────────────────────────── */
    char output_path[1024];
    snprintf(output_path, sizeof(output_path), "%s", config->output_path);
    size_t olen = strlen(output_path);
    /* Add .h264 extension if not already there */
    if (olen < 5 || strcmp(output_path + olen - 5, ".h264") != 0)
        snprintf(output_path + olen, sizeof(output_path) - olen, ".h264");

    /* ── 7. Encode header frame + extract SPS/PPS ───────────────────── */
    H264Writer *writer = NULL;
    int header_frames = 0;
    if (!encode_header_frame(input_path, config, &params,
                              backend, nvenc, &writer,
                              &header_frames,
                              error_msg, error_msg_size)) {
        gpu_backend_close_nvenc(backend);
        gpu_backend_destroy(backend); fclose(in);
        return false;
    }
    result->total_frames += header_frames;

    /* ── 8. Build pipeline config ───────────────────────────────────── */
    GpuPipelineConfig pipe_cfg;
    memset(&pipe_cfg, 0, sizeof(pipe_cfg));
    pipe_cfg.frame_width  = (int)params.frame_width;
    pipe_cfg.frame_height = (int)params.frame_height;
    pipe_cfg.grid_cols    = grid_cols;
    pipe_cfg.pay_rows     = pay_rows;
    pipe_cfg.pay_bits     = pay_bits;
    pipe_cfg.block_size   = (int)params.block_size_x;
    pipe_cfg.margin_x     = (int)params.margin_x;
    pipe_cfg.margin_y     = (int)params.margin_y;
    pipe_cfg.rs_k         = k;
    pipe_cfg.rs_n_k       = n_k;
    pipe_cfg.rs_n         = 255;
    pipe_cfg.cal_params   = params;

    if (payload_codec && payload_codec->parity_dict) {
        pipe_cfg.rs_dict      = payload_codec->parity_dict;
        pipe_cfg.rs_dict_size = payload_codec->parity_dict_size;
    }

    /* ── 9. Create streaming pipeline ────────────────────────────────── */
    GpuEncodePipeline *pipeline = gpu_pipeline_create(
        backend, nvenc, &pipe_cfg, error_msg, error_msg_size);
    if (!pipeline) {
        h264_writer_close(writer);
        gpu_backend_close_nvenc(backend);
        gpu_backend_destroy(backend); fclose(in);
        return false;
    }

    /* ── 10. Stream file through pipeline ───────────────────────────── */
    int stream_frames = 0;
    bool ok;
    if (effective_ecc > 0 && payload_codec) {
        ok = stream_encode_rs(pipeline, in, total_bytes, writer,
                               &stream_frames, config,
                               error_msg, error_msg_size);
    } else {
        ok = stream_encode_raw(pipeline, in, total_bytes, writer,
                                &stream_frames,
                                error_msg, error_msg_size);
    }

    if (!ok) {
        gpu_pipeline_destroy(pipeline);
        h264_writer_close(writer);
        gpu_backend_close_nvenc(backend);
        gpu_backend_destroy(backend); fclose(in);
        return false;
    }
    result->total_frames += stream_frames;

    /* ── 11. Flush remaining pipeline frames ────────────────────────── */
    int flushed = gpu_pipeline_flush(pipeline, writer, error_msg, error_msg_size);
    if (flushed < 0) {
        gpu_pipeline_destroy(pipeline);
        h264_writer_close(writer);
        gpu_backend_close_nvenc(backend);
        gpu_backend_destroy(backend); fclose(in);
        return false;
    }
    result->total_frames += flushed;

    /* ── 12. Final NVENC drain ──────────────────────────────────────── */
    gpu_nvenc_flush(nvenc);
    const uint8_t *pkt; int sz;
    while ((sz = gpu_nvenc_get_packet(nvenc, &pkt)) > 0) {
        h264_writer_write_packet(writer, pkt, sz);
    }

    /* ── 13. Compute timing before stats print ──────────────────────── */
    double end_time = (double)clock() / CLOCKS_PER_SEC;
    result->elapsed_sec = end_time - start_time;
    result->fps = result->elapsed_sec > 0.0
        ? (double)result->total_frames / result->elapsed_sec : 0.0;
    result->total_bytes_written = total_bytes;
    snprintf(result->output_path, sizeof(result->output_path),
             "%s", output_path);
    result->params = params;

    /* ── 14. Pipeline statistics ────────────────────────────────────── */
    GpuPipelineStats stats;
    gpu_pipeline_get_stats(pipeline, &stats);
    fprintf(stderr, "\n=== GPU Encode Pipeline Stats ===\n");
    fprintf(stderr, "  Total bytes:       %lld\n",
            (long long)stats.total_bytes_submitted);
    fprintf(stderr, "  Total frames:      %d\n",
            result->total_frames);
    fprintf(stderr, "  Codewords encoded:  %d\n",
            stats.total_codewords_encoded);
    fprintf(stderr, "  RS encode time:    %.1f ms\n", stats.rs_encode_ms);
    fprintf(stderr, "  Frame gen time:    %.1f ms\n", stats.frame_gen_ms);
    fprintf(stderr, "  NVENC time:        %.1f ms\n", stats.nvenc_ms);
    fprintf(stderr, "  Throughput:        %.1f fps\n", result->fps);
    fprintf(stderr, "  Wall time:         %.2f s\n", result->elapsed_sec);

    /* ── 15. Cleanup ────────────────────────────────────────────────── */
    fclose(in);
    gpu_pipeline_destroy(pipeline);
    h264_writer_close(writer);
    gpu_backend_close_nvenc(backend);
    gpu_backend_destroy(backend);

    return true;
#else
    (void)input_path; (void)config; (void)result;
    snprintf(error_msg, (size_t)error_msg_size, "CUDA not compiled");
    return false;
#endif
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Header frame encoder
 *
 * Builds the LSY1 header, RS-encodes it, generates a GPU frame, encodes
 * via NVENC, extracts SPS/PPS from the first IDR bitstream, and creates
 * the H264Writer with extradata.
 * ═══════════════════════════════════════════════════════════════════════════ */

static bool encode_header_frame(const char *input_path,
                                 const EncoderConfig *config,
                                 CalParams *params,
                                 GpuBackend *backend,
                                 GpuNvencEncoder *nvenc,
                                 H264Writer **writer_out,
                                 int *frames_out,
                                 char *error_msg, int error_msg_size) {
#ifdef USE_CUDA
    int pay_bits = cal_params_payload_bits_per_frame(params);
    int grid_cols = (int)params->grid_cols;
    int pay_rows  = cal_params_payload_rows(params);

    /* Build header bytes (LSY1 format) */
    size_t header_raw_len = 0;
    uint8_t *header_raw = encoder_build_header(input_path, params,
                                                &header_raw_len);
    if (!header_raw) {
        snprintf(error_msg, (size_t)error_msg_size, "Header build failed");
        return false;
    }

    /* RS-encode header (CPU-side, small data) */
    int header_ecc = config->rs_ecc_symbols > 0 ? config->rs_ecc_symbols : 32;
    uint8_t header_enc[255];
    int header_enc_len;
    {
        const RSCodec *hc = rs_get_codec(header_ecc);
        if (hc) {
            header_enc_len = rs_encode(hc, header_raw,
                                        (int)header_raw_len, header_enc);
        } else {
            memcpy(header_enc, header_raw, header_raw_len);
            header_enc_len = (int)header_raw_len;
        }
    }
    free(header_raw);

    /* LUT bit expansion (CPU, header is tiny) */
    uint8_t *header_bits = (uint8_t *)calloc((size_t)pay_bits, 1);
    if (!header_bits) {
        snprintf(error_msg, (size_t)error_msg_size, "OOM (header bits)");
        return false;
    }
    int nbytes = header_enc_len;
    if (nbytes * 8 > pay_bits) nbytes = pay_bits / 8;
    for (int i = 0; i < nbytes; ++i)
        ((uint64_t *)header_bits)[i] = LUT64[header_enc[i]];

    /* Generate header frame on GPU */
    uint8_t *d_header_frame = gpu_backend_generate_frame(
        backend, header_bits, pay_bits,
        (int)params->frame_width, (int)params->frame_height,
        grid_cols, pay_rows,
        (int)params->block_size_x,
        (int)params->margin_x, (int)params->margin_y,
        params);
    free(header_bits);

    if (!d_header_frame) {
        snprintf(error_msg, (size_t)error_msg_size,
                 "GPU header frame gen failed");
        return false;
    }
    gpu_backend_sync_encode(backend);

    /* NVENC encode header frame (zero-copy, uses d_header_frame directly).
     * First frame: forces IDR → SPS/PPS emitted in bitstream. */
    const uint8_t *first_pkt = NULL;
    int first_pkt_size = 0;
    if (gpu_nvenc_encode_frame_zerocopy(nvenc, d_header_frame,
            (int)params->frame_width * 3,
            &first_pkt, &first_pkt_size) < 0) {
        snprintf(error_msg, (size_t)error_msg_size,
                 "NVENC header encode failed");
        return false;
    }

    /* Extract SPS/PPS from first encoded frame */
    const uint8_t *sps_pps = NULL;
    int sps_pps_size = gpu_backend_get_nvenc_sps_pps(backend, &sps_pps);

    /* Create H.264 writer with extradata */
    H264Writer *writer = h264_writer_create(config->output_path,
                                              sps_pps, sps_pps_size);
    if (!writer) {
        snprintf(error_msg, (size_t)error_msg_size,
                 "H264Writer creation failed");
        return false;
    }

    /* Write header frame's encoded data */
    if (first_pkt && first_pkt_size > 0) {
        if (!h264_writer_write_packet(writer, first_pkt, first_pkt_size)) {
            h264_writer_close(writer);
            snprintf(error_msg, (size_t)error_msg_size,
                     "Header packet write failed");
            return false;
        }
    }

    /* Drain any queued NVENC packets */
    const uint8_t *extra_pkt;
    int extra_sz;
    while ((extra_sz = gpu_nvenc_get_packet(nvenc, &extra_pkt)) > 0) {
        if (!h264_writer_write_packet(writer, extra_pkt, extra_sz)) {
            h264_writer_close(writer);
            snprintf(error_msg, (size_t)error_msg_size,
                     "Header drain failed");
            return false;
        }
    }

    *writer_out = writer;
    *frames_out = 1;
    return true;
#else
    (void)input_path; (void)config; (void)params;
    (void)backend; (void)nvenc; (void)writer_out; (void)frames_out;
    if (error_msg) snprintf(error_msg, (size_t)error_msg_size,
                            "CUDA not compiled");
    return false;
#endif
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Stream encode with RS (double-buffered GPU RS + frame gen + NVENC)
 *
 * Reads file in STREAM_CHUNK_SIZE chunks, submits each to the GPU pipeline
 * for RS encoding + bit expansion, then drains completed frames to the
 * H.264 writer. The pipeline overlaps RS on chunk N+1 with frame generation
 * + NVENC on chunk N.
 * ═══════════════════════════════════════════════════════════════════════════ */

static bool stream_encode_rs(GpuEncodePipeline *pipeline,
                              FILE *in, int64_t total_bytes,
                              H264Writer *writer,
                              int *frames_out,
                              const EncoderConfig *config,
                              char *error_msg, int error_msg_size) {
    *frames_out = 0;
    int64_t remaining = total_bytes;
    uint8_t *read_buf = (uint8_t *)malloc((size_t)STREAM_CHUNK_SIZE);
    if (!read_buf) {
        snprintf(error_msg, (size_t)error_msg_size, "OOM (read buffer)");
        return false;
    }

    bool is_last = false;
    int chunk_idx = 0;

    while (remaining > 0 && !is_last) {
        /* Read chunk */
        int64_t to_read = remaining < STREAM_CHUNK_SIZE
                          ? remaining : STREAM_CHUNK_SIZE;
        size_t nread = fread(read_buf, 1, (size_t)to_read, in);
        if (nread == 0) break;

        remaining -= (int64_t)nread;
        is_last = (remaining <= 0);

        if (config->progress_callback) {
            config->progress_callback(total_bytes - remaining, total_bytes,
                                       config->progress_user_data);
        }

        /* Submit chunk to GPU pipeline */
        if (!gpu_pipeline_submit(pipeline, read_buf, (int64_t)nread,
                                  is_last, error_msg, error_msg_size)) {
            free(read_buf);
            return false;
        }

        /* Drain completed frames from pipeline (non-blocking) */
        int drained = 0;
        if (!gpu_pipeline_drain(pipeline, writer, &drained,
                                 error_msg, error_msg_size)) {
            free(read_buf);
            return false;
        }
        *frames_out += drained;
        chunk_idx++;

        /* Progress callback */
        if (config->progress_callback && chunk_idx % 4 == 0) {
            config->progress_callback(total_bytes - remaining, total_bytes,
                                       config->progress_user_data);
        }
    }

    /* Final drain: wait for remaining frames */
    {
        int drained = 0;
        if (!gpu_pipeline_drain(pipeline, writer, &drained,
                                 error_msg, error_msg_size)) {
            free(read_buf);
            return false;
        }
        *frames_out += drained;
    }

    free(read_buf);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Stream encode without RS (CPU LUT expansion + GPU frame gen + NVENC)
 *
 * No RS encoding — raw bytes are LUT-expanded on CPU in 64-bit chunks,
 * then fed through the pipeline for GPU frame generation and NVENC encode.
 * ═══════════════════════════════════════════════════════════════════════════ */

static bool stream_encode_raw(GpuEncodePipeline *pipeline,
                               FILE *in, int64_t total_bytes,
                               H264Writer *writer,
                               int *frames_out,
                               char *error_msg, int error_msg_size) {
    *frames_out = 0;
    int64_t remaining = total_bytes;
    uint8_t *read_buf = (uint8_t *)malloc((size_t)STREAM_CHUNK_SIZE);
    if (!read_buf) {
        snprintf(error_msg, (size_t)error_msg_size, "OOM (read buffer)");
        return false;
    }

    bool is_last = false;
    while (remaining > 0 && !is_last) {
        int64_t to_read = remaining < STREAM_CHUNK_SIZE
                          ? remaining : STREAM_CHUNK_SIZE;
        size_t nread = fread(read_buf, 1, (size_t)to_read, in);
        if (nread == 0) break;

        remaining -= (int64_t)nread;
        is_last = (remaining <= 0);

        /* Submit raw bytes (pipeline does LUT expansion internally) */
        if (!gpu_pipeline_submit(pipeline, read_buf, (int64_t)nread,
                                  is_last, error_msg, error_msg_size)) {
            free(read_buf);
            return false;
        }

        /* Drain completed frames */
        int drained = 0;
        if (!gpu_pipeline_drain(pipeline, writer, &drained,
                                 error_msg, error_msg_size)) {
            free(read_buf);
            return false;
        }
        *frames_out += drained;
    }

    /* Final drain */
    {
        int drained = 0;
        if (!gpu_pipeline_drain(pipeline, writer, &drained,
                                 error_msg, error_msg_size)) {
            free(read_buf);
            return false;
        }
        *frames_out += drained;
    }

    free(read_buf);
    return true;
}

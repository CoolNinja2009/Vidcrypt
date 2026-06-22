#include "decoder.h"
#include "bitstream.h"
#include "reedsolomon.h"
#include "simd_decode.h"
#include "threadpool.h"
#include "ffmpeg_pipe.h"
#include "sha256.h"
#include "profiling.h"
#include "gpu_backend.h"
#include "ffmpeg_hwdecoder.h"
#include "libav_decoder.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void decoder_config_defaults(DecoderConfig *config) {
    memset(config, 0, sizeof(DecoderConfig));
    config->num_workers = 1;
    config->backend_mode = BACKEND_CPU;
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

/* ─── Async GPU pipeline slot ────────────────────────────────────────
 * Double-buffered slot for overlapping CPU decode with GPU processing.
 * Each slot holds one frame's GPU buffers and a sync event.
 * The struct uses void* for CUDA event handles to avoid pulling in
 * cuda_runtime.h for the C compiler (not compiled as CUDA). */
typedef struct {
    uint8_t    *d_gray;        /* GPU device: grayscale frame */
    uint8_t    *d_bits;        /* GPU device: extracted bits */
    uint8_t    *h_bits;        /* Pinned host: bits for CPU RS decode */
    int         h_bits_size;   /* allocated size of h_bits */
    void       *event_upload;  /* cudaEvent_t (opaque ptr) */
    void       *event_copyback;/* cudaEvent_t (opaque ptr) */
    bool        valid;         /* slot has a frame queued */
} GpuDecodeSlot;

#ifdef USE_CUDA
#include <cuda_runtime.h>

/* Allocate a decode slot for the given frame size and bit count.
 * Returns true on success. */
static bool gpu_slot_init(GpuDecodeSlot *slot, int width, int height,
                           int max_bits) {
    memset(slot, 0, sizeof(GpuDecodeSlot));
    size_t gray_size = (size_t)width * (size_t)height;
    cudaError_t e;
    e = cudaMalloc((void **)&slot->d_gray, gray_size);  if (e) return false;
    e = cudaMalloc((void **)&slot->d_bits, (size_t)max_bits); if (e) return false;
    e = cudaHostAlloc((void **)&slot->h_bits, (size_t)max_bits,
                       cudaHostAllocDefault);            if (e) return false;
    slot->h_bits_size = max_bits;
    { cudaEvent_t e_up = NULL;
      e = cudaEventCreate(&e_up); if (e) return false;
      slot->event_upload = (void*)e_up; }
    { cudaEvent_t e_cb = NULL;
      e = cudaEventCreate(&e_cb); if (e) return false;
      slot->event_copyback = (void*)e_cb; }
    slot->valid = false;
    return true;
}

static void gpu_slot_destroy(GpuDecodeSlot *slot) {
    if (!slot) return;
    cudaFree(slot->d_gray);   slot->d_gray = NULL;
    cudaFree(slot->d_bits);   slot->d_bits = NULL;
    cudaFreeHost(slot->h_bits); slot->h_bits = NULL;
    cudaEventDestroy((cudaEvent_t)slot->event_upload);
    cudaEventDestroy((cudaEvent_t)slot->event_copyback);
    memset(slot, 0, sizeof(GpuDecodeSlot));
}

/* Submit a frame to the GPU pipeline slot.
 * Queued on the decode stream: upload → extract_bits → copyback.
 * event_copyback will signal when bits are ready for CPU. */
static void gpu_slot_submit(GpuDecodeSlot *slot,
                             const uint8_t *gray_cpu, int gray_pitch,
                             int width, int height,
                             const DecodeGeometry *geom,
                             int pay_bits,
                             GpuBackend *gpu_backend) {
    /* CRITICAL: All GPU operations (upload, kernel, D2H copy) must be on
     * the SAME stream to ensure ordering. gpu_backend_extract_bits uses
     * the backend's internal stream_decode stream internally. We use the
     * same stream for upload and D2H copy by getting it from the backend. */
    cudaStream_t stream = (cudaStream_t)gpu_backend_get_decode_stream(gpu_backend);

    /* Upload grayscale frame to GPU */
    cudaMemcpy2DAsync(slot->d_gray, (size_t)width,
                      gray_cpu, (size_t)gray_pitch,
                      (size_t)width, (size_t)height,
                      cudaMemcpyHostToDevice, stream);

    /* Extract bits via GPU kernel — writes to backend's internal
     * d_bits_out buffer on stream_decode. The returned pointer
     * is the device address of that buffer. */
    uint8_t *d_bits = gpu_backend_extract_bits(gpu_backend, slot->d_gray, width,
                                                width, height,
                                                geom->grid_top_y, geom->grid_left_x,
                                                geom->block_size,
                                                geom->grid_cols, geom->pay_rows,
                                                geom->subsample, geom->sync_rows);

    /* Copy bits from GPU backend's d_bits_out directly to this slot's
     * pinned host buffer. All operations are on the same stream, so
     * the D2H copy is ordered AFTER the extract kernel completes. */
    cudaMemcpyAsync(slot->h_bits, d_bits,
                    (size_t)pay_bits,
                    cudaMemcpyDeviceToHost, stream);

    cudaEventRecord((cudaEvent_t)slot->event_copyback, stream);
    slot->valid = true;
}

/* Wait for the slot's copyback to complete.
 * Returns the host pointer to the extracted bits. */
static uint8_t* gpu_slot_wait(GpuDecodeSlot *slot) {
    cudaEventSynchronize((cudaEvent_t)slot->event_copyback);
    slot->valid = false;
    return slot->h_bits;
}
#else
/* Stubs for non-CUDA builds — use void* for stream type to avoid CUDA header dependency */
static bool gpu_slot_init(GpuDecodeSlot *slot, int w, int h, int mb) {
    (void)slot; (void)w; (void)h; (void)mb; return false; }
static void gpu_slot_destroy(GpuDecodeSlot *slot) { (void)slot; }
static void gpu_slot_submit(GpuDecodeSlot *slot, const uint8_t *g, int gp,
                             int w, int h, const DecodeGeometry *geom,
                             int pb, GpuBackend *gb) {
    (void)slot; (void)g; (void)gp; (void)w; (void)h; (void)geom;
    (void)pb; (void)gb; }
static uint8_t* gpu_slot_wait(GpuDecodeSlot *slot) {
    (void)slot; return NULL; }
#endif

bool decoder_decode_file(const char *input_path, const DecoderConfig *config,
                         DecoderResult *result, char *error_msg, int error_msg_size) {
    memset(result, 0, sizeof(DecoderResult));
    double start_time = (double)clock() / CLOCKS_PER_SEC;
    ProfCtx prof_ctx;
    (void)prof_ctx;

    /* ── GPU init: libavcodec C API + CUDA backend ────────────────────
     * Uses direct libavcodec C API for video frame reading, replacing the
     * ffmpeg CLI pipe (popen) approach. Eliminates process creation
     * (~50ms startup), pipe I/O context switches (~50µs per frame), and
     * extra memory copies.
     *
     * Decoder selection (auto):
     *   1. Tries h264_cuvid/hevc_cuvid for hardware-accelerated decode
     *      (NVDEC handles GPU→CPU transfer internally — no tiled memory
     *       issue because cuvid outputs linear NV12 in system memory)
     *   2. Falls back to software decoder with multi-threading
     *      (FF_THREAD_SLICE | FF_THREAD_FRAME for maximum CPU throughput)
     *
     * GPU pipeline (async double-buffered):
     *   Slot A: GPU upload + extract_bits kernel + D2H copyback
     *   Slot B: CPU libavcodec decode of next frame
     *   While slot A processes on GPU, slot B decodes next frame on CPU
     *   CUDA events synchronize without blocking the pipeline */
    GpuBackend *gpu_backend = NULL;
    bool use_gpu = (config->backend_mode == BACKEND_GPU);
    bool first_ok = false;
    int vw = 0, vh = 0, vstride = 0;
    uint8_t *first_frame_cpu = NULL;
    CalParams params;
    bool is_new_format = false;
    LibAvDecoder *lav_dec = NULL;
    VideoReader *vr = NULL;

    /* ── Init GPU backend (CUDA context, memory pool) ──────────────── */
    if (use_gpu) {
        char gpu_err[256];
        GpuStatus gpu_status = gpu_backend_init(&gpu_backend, 0,
                                                 gpu_err, sizeof(gpu_err));
        if (gpu_status != GPU_INIT_OK) {
            fprintf(stderr, "GPU init failed: %s — falling back to CPU\n", gpu_err);
            use_gpu = false;
            gpu_backend = NULL;
        } else {
            printf("GPU backend: %s\n", gpu_backend_device_name(gpu_backend));
        }
    }

    /* ── Open video ─────────────────────────────────────────────────── */
    prof_stage_begin(&prof_ctx, PROF_DECODE_VIDEO_OPEN);
    if (use_gpu) {
        /* Open with libavcodec C API — tries h264_cuvid first */
        char lav_err[256];
        lav_dec = libav_decoder_open(input_path, 0, true, /* prefer HW */
                                      0, 0,  /* no rescale yet */
                                      lav_err, sizeof(lav_err));
        if (!lav_dec) {
            fprintf(stderr, "libav_decoder open failed: %s — falling back to CPU\n", lav_err);
            use_gpu = false;
        } else {
            printf("Decoder: %s (%s)\n",
                   libav_decoder_name(lav_dec),
                   libav_decoder_using_hardware(lav_dec) ? "HW" : "SW multi-thread");
        }
    }
    if (!use_gpu) {
        /* CPU fallback: use ffmpeg CLI pipe */
        vr = video_reader_open(input_path, NULL);
        if (!vr) {
            snprintf(error_msg, (size_t)error_msg_size,
                     "Cannot open video file: %s", ffmpeg_last_error());
            return false;
        }
    }
    prof_stage_end(&prof_ctx, PROF_DECODE_VIDEO_OPEN);

    /* ── Read first frame ───────────────────────────────────────────── */
    prof_stage_begin(&prof_ctx, PROF_DECODE_FIRST_FRAME);
    if (use_gpu) {
        uint8_t *gray = NULL;
        int pitch = 0;
        first_ok = libav_decoder_read_frame(lav_dec, &gray, &pitch, &vw, &vh);
        if (first_ok) {
            vstride = pitch;
            first_frame_cpu = (uint8_t *)malloc((size_t)vw * (size_t)vh);
            if (first_frame_cpu) {
                for (int y = 0; y < vh; ++y)
                    memcpy(first_frame_cpu + y * vstride, gray + y * vstride, (size_t)vstride);
            }
        }
    } else {
        const uint8_t *ff_frame = NULL;
        first_ok = video_reader_read_frame(vr, &ff_frame, &vstride, &vw, &vh);
        if (first_ok) {
            first_frame_cpu = (uint8_t *)malloc((size_t)vw * (size_t)vh);
            if (first_frame_cpu) {
                for (int y = 0; y < vh; ++y)
                    memcpy(first_frame_cpu + y * vstride, ff_frame + y * vstride, (size_t)vstride);
            }
        }
    }
    prof_stage_end(&prof_ctx, PROF_DECODE_FIRST_FRAME);
    if (!first_ok || !first_frame_cpu) {
        if (lav_dec) libav_decoder_close(lav_dec);
        if (vr) video_reader_close(vr);
        if (gpu_backend) gpu_backend_destroy(gpu_backend);
        snprintf(error_msg, (size_t)error_msg_size, "Cannot read first frame");
        return false;
    }

    /* Extract calibration from the first frame */
    prof_stage_begin(&prof_ctx, PROF_DECODE_CALIBRATION);
    is_new_format = extract_calibration(first_frame_cpu, vw, vh, &params);
    prof_stage_end(&prof_ctx, PROF_DECODE_CALIBRATION);

    /* If resolution differs from original, reopen with scale.
     * First check: only if we have a valid reader (vr || lav_dec).
     * Second check below is the fallback for YouTube re-encodes. */
    if ((vr || lav_dec) && (vw != (int)params.frame_width || vh != (int)params.frame_height)) {
        /* Close old decoder, reopen with scale */
        if (use_gpu) {
            libav_decoder_close(lav_dec);
            char lav_err[256];
            lav_dec = libav_decoder_open(input_path, 0, true,
                                          (int)params.frame_width,
                                          (int)params.frame_height,
                                          lav_err, sizeof(lav_err));
            if (!lav_dec) {
                if (gpu_backend) gpu_backend_destroy(gpu_backend);
                free(first_frame_cpu);
                snprintf(error_msg, (size_t)error_msg_size,
                         "Cannot reopen video with scale: %s", lav_err);
                return false;
            }
        } else {
            video_reader_close(vr);
            vr = video_reader_open_scaled(input_path, NULL,
                                           (int)params.frame_width,
                                           (int)params.frame_height);
            if (!vr) {
                if (gpu_backend) gpu_backend_destroy(gpu_backend);
                free(first_frame_cpu);
                snprintf(error_msg, (size_t)error_msg_size,
                         "Cannot reopen video with scale: %s", ffmpeg_last_error());
                return false;
            }
        }

        /* Re-read first frame at new resolution */
        prof_stage_begin(&prof_ctx, PROF_DECODE_FIRST_FRAME);
        first_ok = false;
        if (use_gpu) {
            uint8_t *gray = NULL;
            int pitch = 0;
            first_ok = libav_decoder_read_frame(lav_dec, &gray, &pitch, &vw, &vh);
            if (first_ok) {
                vstride = pitch;
                free(first_frame_cpu);
                first_frame_cpu = (uint8_t *)malloc((size_t)vw * (size_t)vh);
                if (first_frame_cpu) {
                    for (int y = 0; y < vh; ++y)
                        memcpy(first_frame_cpu + y * vstride, gray + y * vstride, (size_t)vstride);
                }
            }
        } else {
            const uint8_t *ff_frame = NULL;
            first_ok = video_reader_read_frame(vr, &ff_frame, &vstride, &vw, &vh);
            if (first_ok) {
                free(first_frame_cpu);
                first_frame_cpu = (uint8_t *)malloc((size_t)vw * (size_t)vh);
                if (first_frame_cpu) {
                    for (int y = 0; y < vh; ++y)
                        memcpy(first_frame_cpu + y * vstride, ff_frame + y * vstride, (size_t)vstride);
                }
            }
        }
        prof_stage_end(&prof_ctx, PROF_DECODE_FIRST_FRAME);
        if (!first_ok || !first_frame_cpu) {
            if (gpu_backend) gpu_backend_destroy(gpu_backend);
            snprintf(error_msg, (size_t)error_msg_size,
                     "Cannot read first frame after rescale");
            return false;
        }

        prof_stage_begin(&prof_ctx, PROF_DECODE_CALIBRATION);
        is_new_format = extract_calibration(first_frame_cpu, vw, vh, &params);
        prof_stage_end(&prof_ctx, PROF_DECODE_CALIBRATION);
    }

    /* Handle non-scaled mismatch (e.g., YouTube re-encode at different res) */ 
    if (vw != (int)params.frame_width || vh != (int)params.frame_height) {
        /* Same as above but without the outer condition check */
        if (use_gpu) {
            libav_decoder_close(lav_dec);
            char lav_err[256];
            lav_dec = libav_decoder_open(input_path, 0, true,
                                          (int)params.frame_width,
                                          (int)params.frame_height,
                                          lav_err, sizeof(lav_err));
            if (!lav_dec) {
                if (gpu_backend) gpu_backend_destroy(gpu_backend);
                free(first_frame_cpu);
                snprintf(error_msg, (size_t)error_msg_size,
                         "Cannot reopen video with scale: %s", lav_err);
                return false;
            }
        } else {
            video_reader_close(vr);
            vr = video_reader_open_scaled(input_path, NULL,
                                           (int)params.frame_width,
                                           (int)params.frame_height);
            if (!vr) {
                free(first_frame_cpu);
                snprintf(error_msg, (size_t)error_msg_size,
                         "Cannot reopen video with scale: %s", ffmpeg_last_error());
                return false;
            }
        }

        prof_stage_begin(&prof_ctx, PROF_DECODE_FIRST_FRAME);
        if (use_gpu) {
            uint8_t *gray = NULL;
            int pitch = 0;
            first_ok = libav_decoder_read_frame(lav_dec, &gray, &pitch, &vw, &vh);
            if (first_ok) {
                vstride = pitch;
                free(first_frame_cpu);
                first_frame_cpu = (uint8_t *)malloc((size_t)vw * (size_t)vh);
                if (first_frame_cpu) {
                    for (int y = 0; y < vh; ++y)
                        memcpy(first_frame_cpu + y * vstride, gray + y * vstride, (size_t)vstride);
                }
            }
        } else {
            const uint8_t *ff_frame = NULL;
            first_ok = video_reader_read_frame(vr, &ff_frame, &vstride, &vw, &vh);
            if (first_ok) {
                free(first_frame_cpu);
                first_frame_cpu = (uint8_t *)malloc((size_t)vw * (size_t)vh);
                if (first_frame_cpu) {
                    for (int y = 0; y < vh; ++y)
                        memcpy(first_frame_cpu + y * vstride, ff_frame + y * vstride, (size_t)vstride);
                }
            }
        }
        prof_stage_end(&prof_ctx, PROF_DECODE_FIRST_FRAME);
        if (!first_ok || !first_frame_cpu) {
            snprintf(error_msg, (size_t)error_msg_size,
                     "Cannot read first frame after rescale");
            return false;
        }

        prof_stage_begin(&prof_ctx, PROF_DECODE_CALIBRATION);
        is_new_format = extract_calibration(first_frame_cpu, vw, vh, &params);
        prof_stage_end(&prof_ctx, PROF_DECODE_CALIBRATION);
    }

    DecodeGeometry geom;
    decode_geometry_init(&geom, &params, is_new_format);
    int pay_bits = geom.pay_rows * geom.grid_cols;

    /* ── Detect frame alignment offset from corner markers ───────────── */
    prof_stage_begin(&prof_ctx, PROF_DECODE_HEADER_PARSE);
    FrameOffset offset = detect_frame_offset(first_frame_cpu, vw, vh, vstride);
    if (offset.confidence >= 70) {
        fprintf(stderr, "Frame alignment offset: (%+d, %+d) (confidence %d%%) — adjusting grid\n",
                offset.dx, offset.dy, offset.confidence);
        geom.grid_left_x  += offset.dx;
        geom.grid_top_y   += offset.dy;
        geom.pay_x_start  += offset.dx;
        geom.pay_x_end    += offset.dx;
        geom.pay_y_start  += offset.dy;
        geom.pay_y_end    += offset.dy;
        geom.sync_x_start += offset.dx;
        geom.sync_y       += offset.dy;
    }
    prof_stage_end(&prof_ctx, PROF_DECODE_HEADER_PARSE);

    /* Decode header bits from first frame */
    uint8_t *header_bits = (uint8_t *)malloc((size_t)pay_bits);
    if (!header_bits) {
        if (lav_dec) libav_decoder_close(lav_dec);
        if (vr) video_reader_close(vr);
        if (gpu_backend) gpu_backend_destroy(gpu_backend);
        free(first_frame_cpu);
        snprintf(error_msg, (size_t)error_msg_size, "Out of memory");
        return false;
    }

    /* Validate sync pattern */
    prof_stage_begin(&prof_ctx, PROF_DECODE_HEADER_PARSE);
    int sync_matches = 0, sync_total = 0;
    int sub = geom.subsample;
    int total_sync_samples = (geom.block_size / sub) * (geom.block_size / sub);
    if (total_sync_samples < 1) total_sync_samples = 1;
    for (int col = 0; col < geom.grid_cols; ++col) {
        int x = geom.sync_x_start + col * geom.block_size;
        int count = tile_count_white(first_frame_cpu + geom.sync_y * vstride + x,
                                      vstride,
                                      geom.block_size, geom.block_size, sub);
        int bit = (count > total_sync_samples / 2) ? 1 : 0;
        if (bit == (col % 2)) sync_matches++;
        sync_total++;
    }
    double sync_threshold = 0.50;
    bool sync_ok = sync_total > 0 && (double)sync_matches / sync_total > sync_threshold;
    fprintf(stderr, "Sync: %d/%d = %.1f%% (threshold %.2f, geom: %dx%d bs=%d)\n",
            sync_matches, sync_total,
            sync_total > 0 ? (double)sync_matches / sync_total * 100.0 : 0.0,
            sync_threshold,
            geom.grid_cols, geom.pay_rows, geom.block_size);
    prof_stage_end(&prof_ctx, PROF_DECODE_HEADER_PARSE);

    prof_stage_begin(&prof_ctx, PROF_DECODE_BIT_EXTRACT);
    int nbits = decode_payload_tiles(&geom, first_frame_cpu, vstride, header_bits, pay_bits);
    prof_stage_end(&prof_ctx, PROF_DECODE_BIT_EXTRACT);

    if (!sync_ok) {
        free(header_bits);
        if (lav_dec) libav_decoder_close(lav_dec);
        if (vr) video_reader_close(vr);
        if (gpu_backend) gpu_backend_destroy(gpu_backend);
        free(first_frame_cpu);
        snprintf(error_msg, (size_t)error_msg_size, "Sync pattern not found");
        return false;
    }

    prof_stage_begin(&prof_ctx, PROF_DECODE_HEADER_PARSE);
    char filename[256];
    int64_t orig_size = 0;
    uint8_t expected_checksum[32];
    bool header_ok = decoder_parse_header(header_bits, nbits, &params,
                      filename, sizeof(filename), &orig_size, expected_checksum,
                      error_msg, error_msg_size);
    prof_stage_end(&prof_ctx, PROF_DECODE_HEADER_PARSE);
    if (!header_ok) {
        free(header_bits);
        if (lav_dec) libav_decoder_close(lav_dec);
        if (vr) video_reader_close(vr);
        if (gpu_backend) gpu_backend_destroy(gpu_backend);
        free(first_frame_cpu);
        return false;
    }

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
    if (!bit_buffer) {
        free(header_bits);
        if (lav_dec) libav_decoder_close(lav_dec);
        if (vr) video_reader_close(vr);
        if (gpu_backend) gpu_backend_destroy(gpu_backend);
        free(first_frame_cpu);
        snprintf(error_msg, (size_t)error_msg_size, "Out of memory");
        return false;
    }

    FILE *out = fopen(output_path, "wb");
    if (!out) {
        free(bit_buffer); free(header_bits);
        if (lav_dec) libav_decoder_close(lav_dec);
        if (vr) video_reader_close(vr);
        if (gpu_backend) gpu_backend_destroy(gpu_backend);
        free(first_frame_cpu);
        snprintf(error_msg, (size_t)error_msg_size, "Cannot open output file");
        return false;
    }

    int frame_count = 0;
    int rs_failures = 0;
    int rs_hard_failures = 0;
    int rs_corrections = 0;

    /* Thread pool (CPU path only) */
    int num_workers = config->num_workers;
    if (num_workers <= 0) num_workers = 1;

    ThreadPool *pool = NULL;
    if (!use_gpu) {
        pool = threadpool_create(num_workers, decoder_worker_func);
        if (!pool) {
            fclose(out); free(bit_buffer); free(header_bits);
            if (vr) video_reader_close(vr);
            free(first_frame_cpu);
            snprintf(error_msg, (size_t)error_msg_size, "Failed to create thread pool");
            return false;
        }
    }

    int frames_per_chunk = use_gpu ? 1 : (num_workers * 4);
    if (!use_gpu && frames_per_chunk < 8) frames_per_chunk = 8;
    if (!use_gpu && frames_per_chunk > 128) frames_per_chunk = 128;

    size_t gray_chunk_size = (size_t)frames_per_chunk * (size_t)vh * (size_t)vstride;
    size_t bits_chunk_size = (size_t)frames_per_chunk * (size_t)pay_bits;

    uint8_t *gray_chunk = NULL;
    uint8_t *bits_chunk = NULL;
    DecodeJob *jobs = NULL;
    if (!use_gpu) {
        gray_chunk = (uint8_t *)malloc(gray_chunk_size);
        bits_chunk = (uint8_t *)calloc(1, bits_chunk_size);
        jobs = (DecodeJob *)calloc((size_t)frames_per_chunk, sizeof(DecodeJob));
        if (!gray_chunk || !bits_chunk || !jobs) {
            free(gray_chunk); free(bits_chunk); free(jobs);
            threadpool_destroy(pool); fclose(out); free(bit_buffer);
            free(header_bits); free(first_frame_cpu);
            if (vr) video_reader_close(vr);
            snprintf(error_msg, (size_t)error_msg_size, "Out of memory");
            return false;
        }
    }

    /* Pre-allocate decoded buffer */
    int decoded_buf_size = (rs_ecc > 0) ? 255 : pay_bytes;
    uint8_t *decoded_buf = (uint8_t *)malloc((size_t)decoded_buf_size);
    if (!decoded_buf) {
        free(gray_chunk); free(bits_chunk); free(jobs);
        threadpool_destroy(pool); fclose(out); free(bit_buffer);
        free(header_bits); free(first_frame_cpu);
        if (vr) video_reader_close(vr);
        snprintf(error_msg, (size_t)error_msg_size, "Out of memory");
        return false;
    }

    /* ─── Async GPU decode slots (double-buffered) ────────────────── */
    GpuDecodeSlot gpu_slots[2];
    memset(gpu_slots, 0, sizeof(gpu_slots));
    if (use_gpu) {
        if (!gpu_slot_init(&gpu_slots[0], vw, vh, pay_bits) ||
            !gpu_slot_init(&gpu_slots[1], vw, vh, pay_bits)) {
            gpu_slot_destroy(&gpu_slots[0]);
            gpu_slot_destroy(&gpu_slots[1]);
            threadpool_destroy(pool); fclose(out); free(bit_buffer);
            free(header_bits); free(decoded_buf); free(first_frame_cpu);
            if (gpu_backend) gpu_backend_destroy(gpu_backend);
            if (lav_dec) libav_decoder_close(lav_dec);
            snprintf(error_msg, (size_t)error_msg_size, "Out of GPU memory");
            return false;
        }
    }

    /* ─── RS decode + file write helper ────────────────────────────── */
#define PROCESS_BITS(bits_ptr, bits_count) do { \
    if (bit_buffer_len + (bits_count) > bit_buffer_cap) { \
        bit_buffer_cap = (bit_buffer_len + (bits_count)) * 2; \
        uint8_t *new_buf = (uint8_t *)realloc(bit_buffer, (size_t)bit_buffer_cap); \
        if (!new_buf) { fprintf(stderr, "OOM in bit buffer\\n"); break; } \
        bit_buffer = new_buf; \
    } \
    memcpy(bit_buffer + bit_buffer_len, (bits_ptr), (size_t)(bits_count)); \
    bit_buffer_len += (bits_count); \
    frame_count++; \
    const RSCodec *codec = rs_ecc > 0 ? rs_get_codec(rs_ecc) : NULL; \
    while (bit_buffer_len >= chunk_bits && written < orig_size) { \
        uint8_t raw_block[255]; \
        bits_to_bytes(bit_buffer, (size_t)chunk_bits, raw_block, 255); \
        int data_bytes; \
        if (codec) { \
            RSDecodeResult rs_res; \
            prof_stage_begin(&prof_ctx, PROF_DECODE_RS_DECODE); \
            rs_decode(codec, raw_block, &rs_res); \
            prof_stage_end(&prof_ctx, PROF_DECODE_RS_DECODE); \
            if (rs_res.status < 0) { \
                rs_failures++; \
                rs_hard_failures++; \
                memcpy(decoded_buf, raw_block, (size_t)codec->msg_length); \
                data_bytes = (int)codec->msg_length; \
            } else { \
                memcpy(decoded_buf, rs_res.decoded, (size_t)codec->msg_length); \
                data_bytes = (int)codec->msg_length; \
                if (rs_res.corrected > 0) { rs_failures++; rs_corrections++; } \
            } \
        } else { \
            data_bytes = (int)bits_to_bytes(bit_buffer, (size_t)pay_bits, decoded_buf, (size_t)pay_bytes); \
        } \
        int64_t remaining = orig_size - written; \
        int to_write = (int)((int64_t)data_bytes < remaining ? (int64_t)data_bytes : remaining); \
        prof_stage_begin(&prof_ctx, PROF_DECODE_FILE_WRITE); \
        fwrite(decoded_buf, 1, (size_t)to_write, out); \
        prof_stage_end(&prof_ctx, PROF_DECODE_FILE_WRITE); \
        written += to_write; \
        memmove(bit_buffer, bit_buffer + chunk_bits, (size_t)(bit_buffer_len - chunk_bits)); \
        bit_buffer_len -= chunk_bits; \
        if (config->progress_callback) \
            config->progress_callback(written, orig_size, config->progress_user_data); \
    } \
} while(0)

    /* ─── Frame decode loop ──────────────────────────────────────────
     * GPU path: async double-buffered pipeline
     *   Slot A: GPU upload + extract_bits kernel + D2H copyback
     *   Slot B: CPU libavcodec decode of next frame
     *   CPU RS decode + file write processes previous slot's bits
     *
     * CPU path: batch frames → threadpool → RS decode + file write */
    if (use_gpu) {
        /* ── Prime: decode first frame and submit to slot 0 ───────── */
        int current_slot = 0;
        uint8_t *gray_ptr = NULL;
        int gray_p = 0, fw = 0, fh = 0;
        bool more_frames;

        more_frames = libav_decoder_read_frame(lav_dec, &gray_ptr, &gray_p, &fw, &fh);
        if (more_frames) {
            gpu_slot_submit(&gpu_slots[current_slot], gray_ptr, gray_p,
                            fw, fh, &geom, pay_bits,
                            gpu_backend);
        }
        current_slot = 1 - current_slot;

        /* ── Async pipeline loop ─────────────────────────────────── */
        while (more_frames) {
            /* Decode the NEXT frame on CPU (overlaps with GPU) */
            more_frames = libav_decoder_read_frame(lav_dec, &gray_ptr, &gray_p,
                                                    &fw, &fh);

            /* Submit current frame to GPU (if any) */
            if (more_frames) {
                gpu_slot_submit(&gpu_slots[current_slot], gray_ptr, gray_p,
                                fw, fh, &geom, pay_bits,
                                gpu_backend);
            }

            /* Wait for the PREVIOUS slot's bits (GPU processing finished
             * while CPU was decoding the next frame) */
            int prev_slot = 1 - current_slot;
            uint8_t *h_bits = gpu_slot_wait(&gpu_slots[prev_slot]);

            /* RS decode + file write on CPU */
            if (h_bits)
                PROCESS_BITS(h_bits, pay_bits);

            current_slot = 1 - current_slot;
        }

        /* ── Drain: wait for remaining slots ─────────────────────── */
        for (int i = 0; i < 2; ++i) {
            int slot = (current_slot + i) % 2;
            if (gpu_slots[slot].valid) {
                uint8_t *h_bits = gpu_slot_wait(&gpu_slots[slot]);
                if (h_bits)
                    PROCESS_BITS(h_bits, pay_bits);
            }
        }

#ifdef USE_CUDA
        /* Sync GPU stream to ensure all work complete.
         * Uses backend's internal decode stream which is shared by
         * upload, kernel, and copyback operations. */
        cudaStreamSynchronize((cudaStream_t)gpu_backend_get_decode_stream(gpu_backend));
#endif
    } else {
        /* ── CPU path: batch frames, submit to threadpool ────────── */
        const uint8_t *frame_data;
        int fw, fh, fstride;
        int chunk_idx = 0;

        while (1) {
            prof_stage_begin(&prof_ctx, PROF_DECODE_FRAME_ACQUIRE);
            bool has_frame = video_reader_read_frame(vr, &frame_data,
                                                      &fstride, &fw, &fh);
            prof_stage_end(&prof_ctx, PROF_DECODE_FRAME_ACQUIRE);
            if (!has_frame) break;

            uint8_t *gray_buf = gray_chunk + (size_t)chunk_idx * (size_t)fh * (size_t)fstride;
            for (int y = 0; y < fh; ++y)
                memcpy(gray_buf + y * fstride, frame_data + y * fstride, (size_t)fstride);

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

                for (int i = 0; i < chunk_idx; ++i)
                    PROCESS_BITS(jobs[i].bits, pay_bits);

                chunk_idx = 0;
            }
        }

        /* Drain remaining CPU frames */
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

            for (int i = 0; i < chunk_idx; ++i)
                PROCESS_BITS(jobs[i].bits, pay_bits);
        }
    }

#undef PROCESS_BITS

    /* ── Flush remaining bits to output ───────────────────────────── */
    if (written < orig_size && bit_buffer_len > 0) {
        uint8_t raw_block[255];
        bits_to_bytes(bit_buffer, (size_t)bit_buffer_len, raw_block, 255);
        int64_t remaining = orig_size - written;
        int to_write = (int)((int64_t)255 < remaining ? 255 : remaining);
        fwrite(raw_block, 1, (size_t)to_write, out);
        written += to_write;
    }

    /* ── Cleanup ──────────────────────────────────────────────────── */
    threadpool_destroy(pool);
    free(gray_chunk);
    free(bits_chunk);
    free(jobs);
    if (gpu_backend) gpu_backend_destroy(gpu_backend);
    if (lav_dec) libav_decoder_close(lav_dec);
    if (vr) video_reader_close(vr);
    fclose(out);
    free(bit_buffer);
    free(header_bits);
    free(decoded_buf);
    free(first_frame_cpu);
#ifdef USE_CUDA
    /* gpu_decode_stream is owned by gpu_backend, destroyed in gpu_backend_destroy */
#endif
    gpu_slot_destroy(&gpu_slots[0]);
    gpu_slot_destroy(&gpu_slots[1]);

    double end_time = (double)clock() / CLOCKS_PER_SEC;
    result->elapsed_sec = end_time - start_time;
    result->total_frames = frame_count;
    result->total_bytes_written = written;
    result->fps = (frame_count > 0 && result->elapsed_sec > 0.0) ? (double)frame_count / result->elapsed_sec : 0.0;
    result->rs_failures = rs_failures;
    fprintf(stderr, "RS diagnostics: %d total events (%d hard failures, %d corrections)\n",
            rs_failures, rs_hard_failures, rs_corrections);
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

/* ═══════════════════════════════════════════════════════════════════════════
 * Streaming GPU encode pipeline orchestrator
 *
 * Implements a double-buffered pipeline that overlaps GPU RS encode with
 * frame generation + NVENC hardware encode:
 *
 *   Chunk N:   [RS encode] → [Bit expand]
 *   Chunk N-1:               [Frame gen] → [NVENC encode] → [Write]
 *
 * This keeps both the CUDA SMs and the NVENC ASIC busy simultaneously.
 *
 * GPU memory usage (double-buffered, worst case RS(255,223,32)):
 *   Per RS buffer: k*N_cw + 255*N_cw + 255*N_cw*8 bytes
 *   For 64MB chunk with k=223: ~288K codewords
 *   Raw:    64 MB
 *   Encoded: 73.5 MB (288K × 255)
 *   Bits:    588 MB (288K × 2040)
 *
 * At 64MB chunks, total GPU memory ≈ 64+73.5+588 = ~725 MB per buffer.
 * Double-buffered = 1.45 GB — acceptable on RTX 5070 (12 GB).
 *
 * For streaming on lower-memory GPUs, chunk size is configurable.
 * ═══════════════════════════════════════════════════════════════════════════ */

#include "gpu_encode_pipeline.h"
#include "gpu_backend.h"
#include "gpu_nvenc.h"
#include "gpu_rs_encode.cuh"
#include "h264_writer.h"
#include "bitstream.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

/* ─── Internal pipeline slot ─────────────────────────────────────────── */
typedef struct {
    uint8_t  *d_raw;        /* [chunk_cw * k] padded raw data */
    uint8_t  *d_enc;        /* [chunk_cw * 255] RS-encoded */
    uint64_t *d_bits;       /* [chunk_cw * 255 * 8] bit-expanded */
    uint8_t  *h_bits;       /* pinned host pointer for bit data download */

    int64_t   bit_len;      /* total bit bytes in this chunk */
    int       num_cw;       /* number of codewords */
    int       num_frames;   /* frames this chunk produces */
    int       frame_idx;    /* current frame being drained */
    bool      active;       /* chunk has been submitted */
    bool      rs_done;      /* RS encode + bit expand complete */
    bool      drained;      /* all frames have been NVENC-encoded */
    cudaEvent_t event;      /* signals RS completion */
} PipeSlot;

/* ─── Pipeline state ──────────────────────────────────────────────────── */

struct GpuEncodePipeline {
    GpuBackend       *backend;
    GpuNvencEncoder  *nvenc;

    /* Double-buffered RS slots */
    PipeSlot          slot[2];
    int               active_slot;       /* slot currently being RS-encoded */

    /* Persistent GPU resources */
    uint8_t  *d_dict;          /* [k * 256 * n_k] RS parity dictionary */
    uint8_t  *d_dict_T;        /* [k * n_k * 256] transposed dict */
    uint8_t  *d_frame_bits;    /* [pay_bits] per-frame bit buffer */
    uint8_t  *d_frame;         /* BGR24 frame buffer [W*H*3] */

    /* CUDA streams */
    cudaStream_t stream;

    /* Configuration */
    GpuPipelineConfig cfg;
    int chunk_max_cw;          /* max codewords per chunk (memory budget) */

    /* Persistent frame pattern data (avoids per-frame allocations) */
    int cal_bottom;
    int pay_y_start;
    int pay_x_start;

    /* Statistics */
    GpuPipelineStats stats;
    double last_rs_time;
    double last_fg_time;
    double last_nvenc_time;

    /* Remaining data tracking */
    bool  input_done;
    int64_t remaining_bytes;   /* bytes left in final partial chunk */
};

/* ─── Helper: compute chunk size based on available GPU memory ────────── */

static int compute_chunk_codewords(int k, int n_k) {
    /* Target: handle entire file in one submit for simplicity.
     * For 100MB file with k=223: ceil(100MB/223) = 470,212 codewords.
     * GPU memory per slot:
     *   raw:     N_cw * k         = 470212 * 223 = ~100 MB
     *   enc:     N_cw * 255       = 470212 * 255 = ~114 MB
     *   bits:    N_cw * 255 * 8   = ~914 MB (bit-expanded!)
     * Double-buffered: ~2.2 GB — fits on RTX 5070 (12 GB).
     * Set limit to 500,000 codewords (~106 MB raw for RS32, ~2.3 GB GPU). */
    (void)n_k;
    return 500000;
}

/* ─── Allocate a pipeline slot's GPU buffers ─────────────────────────── */

static bool slot_alloc(PipeSlot *s, int num_cw, int k, int n_k,
                       char *error_out, int error_size) {
    size_t raw_sz = (size_t)num_cw * (size_t)k;
    size_t enc_sz = (size_t)num_cw * 255UL;
    size_t bits_sz = enc_sz * 8;

    s->d_raw  = NULL;
    s->d_enc  = NULL;
    s->d_bits = NULL;
    s->h_bits = NULL;

    if (cudaMalloc((void **)&s->d_raw, raw_sz) != cudaSuccess) goto fail;
    if (cudaMalloc((void **)&s->d_enc, enc_sz) != cudaSuccess) goto fail;
    if (cudaMalloc((void **)&s->d_bits, bits_sz) != cudaSuccess) goto fail;
    if (cudaHostAlloc((void **)&s->h_bits, bits_sz,
                       cudaHostAllocDefault) != cudaSuccess) goto fail;

    s->num_cw = num_cw;
    (void)n_k;
    return true;

fail:
    cudaFree(s->d_raw);  s->d_raw = NULL;
    cudaFree(s->d_enc);  s->d_enc = NULL;
    cudaFree(s->d_bits); s->d_bits = NULL;
    cudaFreeHost(s->h_bits); s->h_bits = NULL;
    if (error_out) snprintf(error_out, (size_t)error_size,
                            "GPU OOM allocating pipeline slot (%zu MB)",
                            (raw_sz + enc_sz + bits_sz) / (1024*1024));
    return false;
}

static void slot_free(PipeSlot *s) {
    cudaFree(s->d_raw);
    cudaFree(s->d_enc);
    cudaFree(s->d_bits);
    cudaFreeHost(s->h_bits);
    if (s->event) { cudaEventDestroy(s->event); s->event = NULL; }
    memset(s, 0, sizeof(PipeSlot));
}

/* ─── Create / Destroy ──────────────────────────────────────────────────── */

GpuEncodePipeline *gpu_pipeline_create(
    GpuBackend *backend,
    GpuNvencEncoder *nvenc,
    const GpuPipelineConfig *config,
    char *error_out, int error_size)
{
#ifdef USE_CUDA
    GpuEncodePipeline *p = (GpuEncodePipeline *)calloc(1, sizeof(GpuEncodePipeline));
    if (!p) {
        if (error_out) snprintf(error_out, (size_t)error_size, "OOM");
        return NULL;
    }

    p->backend = backend;
    p->nvenc   = nvenc;
    p->cfg     = *config;

    int k   = config->rs_k > 0 ? config->rs_k : config->pay_bits / 8;
    int n_k = config->rs_n_k;

    p->chunk_max_cw = compute_chunk_codewords(k, n_k);

    /* Allocate both pipeline slots */
    for (int i = 0; i < 2; ++i) {
        if (!slot_alloc(&p->slot[i], p->chunk_max_cw, k, n_k,
                        error_out, error_size)) {
            for (int j = 0; j < i; ++j) slot_free(&p->slot[j]);
            free(p);
            return NULL;
        }
        if (cudaEventCreate(&p->slot[i].event) != cudaSuccess) {
            for (int j = 0; j < 2; ++j) slot_free(&p->slot[j]);
            free(p);
            if (error_out) snprintf(error_out, (size_t)error_size,
                                    "cudaEventCreate failed");
            return NULL;
        }
    }

    /* Upload RS dictionary to GPU */
    if (config->rs_dict && config->rs_dict_size > 0) {
        if (cudaMalloc((void **)&p->d_dict,
                        config->rs_dict_size) != cudaSuccess) {
            for (int j = 0; j < 2; ++j) slot_free(&p->slot[j]);
            free(p);
            if (error_out) snprintf(error_out, (size_t)error_size,
                                    "GPU OOM (dict)");
            return NULL;
        }
        cudaMemcpy(p->d_dict, config->rs_dict, config->rs_dict_size,
                   cudaMemcpyHostToDevice);

        /* Build and upload transposed dictionary */
        size_t dict_T_size = (size_t)k * (size_t)n_k * 256UL;
        uint8_t *dict_T_host = (uint8_t *)malloc(dict_T_size);
        if (dict_T_host) {
            rs_build_transposed_dict(config->rs_dict, k, n_k, dict_T_host);
            cudaMalloc((void **)&p->d_dict_T, dict_T_size);
            cudaMemcpy(p->d_dict_T, dict_T_host, dict_T_size,
                       cudaMemcpyHostToDevice);
            free(dict_T_host);
        } else {
            p->d_dict_T = NULL; /* non-fatal: tiled kernel falls back */
        }
    }

    /* Allocate per-frame buffers */
    size_t frame_bits_sz = (size_t)config->pay_bits;
    size_t frame_sz = (size_t)config->frame_width
                    * (size_t)config->frame_height * 3; /* BGR24 */

    if (cudaMalloc((void **)&p->d_frame_bits, frame_bits_sz) != cudaSuccess ||
        cudaMalloc((void **)&p->d_frame, frame_sz) != cudaSuccess) {
        for (int j = 0; j < 2; ++j) slot_free(&p->slot[j]);
        cudaFree(p->d_dict); cudaFree(p->d_dict_T);
        free(p);
        if (error_out) snprintf(error_out, (size_t)error_size, "GPU OOM (frame)");
        return NULL;
    }

    /* Create CUDA stream */
    if (cudaStreamCreate(&p->stream) != cudaSuccess) {
        for (int j = 0; j < 2; ++j) slot_free(&p->slot[j]);
        cudaFree(p->d_dict); cudaFree(p->d_dict_T);
        cudaFree(p->d_frame_bits); cudaFree(p->d_frame);
        free(p);
        if (error_out) snprintf(error_out, (size_t)error_size, "stream failed");
        return NULL;
    }

    /* Upload LUT64 to device constant memory */
    gpu_rs_upload_lut64();

    /* Pre-compute payload region geometry */
    p->cal_bottom  = (int)((float)config->frame_height * 0.06f);
    p->pay_y_start = p->cal_bottom + config->margin_y + config->block_size;
    p->pay_x_start = config->margin_x;

    return p;
#else
    (void)backend; (void)nvenc; (void)config;
    if (error_out) snprintf(error_out, (size_t)error_size, "CUDA not compiled");
    return NULL;
#endif
}

void gpu_pipeline_destroy(GpuEncodePipeline *p) {
#ifdef USE_CUDA
    if (!p) return;
    for (int i = 0; i < 2; ++i) slot_free(&p->slot[i]);
    cudaFree(p->d_dict);
    cudaFree(p->d_dict_T);
    cudaFree(p->d_frame_bits);
    cudaFree(p->d_frame);
    if (p->stream) cudaStreamDestroy(p->stream);
    free(p);
#else
    (void)p;
#endif
}

/* ─── Submit a chunk of raw data for GPU RS encoding ──────────────────── */

bool gpu_pipeline_submit(
    GpuEncodePipeline *p,
    const uint8_t *data, int64_t data_size, bool is_last,
    char *error_out, int error_size)
{
#ifdef USE_CUDA
    if (!p || !data || data_size <= 0) {
        if (is_last) { p->input_done = true; return true; }
        if (error_out) snprintf(error_out, (size_t)error_size, "Invalid submit");
        return false;
    }

    int k   = p->cfg.rs_k > 0 ? p->cfg.rs_k : (p->cfg.pay_bits / 8);
    int n_k = p->cfg.rs_n_k;
    int n   = p->cfg.rs_n;

    int idx = p->active_slot;
    PipeSlot *s = &p->slot[idx];

    /* Wait for previous work on this slot to complete */
    if (s->active) {
        cudaError_t e = cudaEventSynchronize(s->event);
        if (e != cudaSuccess) {
            if (error_out) snprintf(error_out, (size_t)error_size,
                                    "cudaEventSync: %s", cudaGetErrorString(e));
            return false;
        }
    }

    /* Determine how many codewords in this chunk */
    int num_cw = (int)((data_size + k - 1) / k);
    if (num_cw > p->chunk_max_cw) num_cw = p->chunk_max_cw;
    int actual_bytes = (int)data_size;
    if (actual_bytes > num_cw * k) actual_bytes = num_cw * k;

    /* Copy raw data to GPU (zero-padded to codeword boundary) */
    size_t raw_sz = (size_t)num_cw * (size_t)k;
    cudaMemsetAsync(s->d_raw, 0, raw_sz, p->stream);
    cudaMemcpyAsync(s->d_raw, data, (size_t)actual_bytes,
                    cudaMemcpyHostToDevice, p->stream);

    if (p->cfg.rs_k > 0 && p->d_dict) {
        /* ── RS encode + bit expand on GPU ──────────────────────────── */
        double t0 = (double)clock() / CLOCKS_PER_SEC;

        rs_encode_tiled_kernel(s->d_raw, p->d_dict, p->d_dict_T,
                                s->d_enc, k, n_k, num_cw, p->stream);

        int enc_bytes = num_cw * 255;
        rs_bit_expand_kernel(s->d_enc, s->d_bits, enc_bytes, p->stream);

        s->bit_len = (int64_t)enc_bytes * 8;
        s->num_cw  = num_cw;

        /* Download bit-expanded data to pinned host memory.
         * This lets the frame gen + NVENC loop run on CPU with
         * host-side data, avoiding GPU-side frame scheduling complexity. */
        cudaMemcpyAsync(s->h_bits, s->d_bits, (size_t)s->bit_len,
                        cudaMemcpyDeviceToHost, p->stream);

        double t1 = (double)clock() / CLOCKS_PER_SEC;
        p->last_rs_time = t1 - t0;
        p->stats.rs_encode_ms += p->last_rs_time * 1000.0;
    } else {
        /* ── No RS: LUT bit expansion on CPU ────────────────────────── */
        int byte_sz = actual_bytes;
        s->bit_len = (int64_t)byte_sz * 8;
        s->num_cw  = 0;

        /* LUT expand on host, then upload to pinned memory */
        uint8_t *temp = (uint8_t *)malloc((size_t)s->bit_len);
        if (!temp) {
            if (error_out) snprintf(error_out, (size_t)error_size, "OOM");
            return false;
        }
        for (int i = 0; i < byte_sz; ++i)
            ((uint64_t *)temp)[i] = LUT64[data[i]];
        memcpy(s->h_bits, temp, (size_t)s->bit_len);
        free(temp);
        cudaEventRecord(s->event, p->stream); /* no-op event for sync */
    }

    /* Record RS completion event */
    cudaEventRecord(s->event, p->stream);

    /* ceil division to include the partial final frame */
    s->num_frames  = (int)((s->bit_len + p->cfg.pay_bits - 1) / p->cfg.pay_bits);
    if (s->num_frames == 0 && s->bit_len > 0) s->num_frames = 1;
    s->frame_idx   = 0;
    s->active      = true;
    s->rs_done     = true;
    s->drained     = false;

    p->active_slot = (idx + 1) % 2;
    p->input_done = is_last;
    p->stats.total_bytes_submitted += actual_bytes;
    p->stats.total_codewords_encoded += num_cw;

    return true;
#else
    (void)p; (void)data; (void)data_size; (void)is_last;
    if (error_out) snprintf(error_out, (size_t)error_size, "CUDA not compiled");
    return false;
#endif
}

/* ─── Drain completed frames to H.264 writer ──────────────────────────── */

bool gpu_pipeline_drain(
    GpuEncodePipeline *p,
    H264Writer *writer,
    int *frames_output,
    char *error_out, int error_size)
{
#ifdef USE_CUDA
    if (frames_output) *frames_output = 0;
    if (!p || !writer) return true;

    /* Process completed slots: slot 1 if slot 0 is active, and vice versa.
     * Actually: drain any slot that has completed RS and has frames left. */
    for (int slot_idx = 0; slot_idx < 2; ++slot_idx) {
        PipeSlot *s = &p->slot[slot_idx];
        if (!s->active || !s->rs_done || s->drained) continue;

        /* Wait for RS to complete if needed */
        cudaError_t e = cudaEventSynchronize(s->event);
        if (e != cudaSuccess) {
            if (error_out) snprintf(error_out, (size_t)error_size,
                                    "sync: %s", cudaGetErrorString(e));
            return false;
        }

        /* Encode frames from this slot */
        while (s->frame_idx < s->num_frames) {
            int64_t offset = (int64_t)s->frame_idx * (int64_t)p->cfg.pay_bits;
            int remaining = (int)(s->bit_len - offset);
            int this_frame_bits = remaining < p->cfg.pay_bits
                                  ? remaining : p->cfg.pay_bits;

            /* Upload frame bits to GPU */
            cudaMemcpyAsync(p->d_frame_bits, s->h_bits + offset,
                            (size_t)this_frame_bits,
                            cudaMemcpyHostToDevice, p->stream);
            if (this_frame_bits < p->cfg.pay_bits) {
                cudaMemsetAsync(p->d_frame_bits + this_frame_bits, 0,
                                (size_t)(p->cfg.pay_bits - this_frame_bits),
                                p->stream);
            }

            /* Generate BGR24 frame on GPU */
            double t_fg0 = (double)clock() / CLOCKS_PER_SEC;
            uint8_t *d_frame_out = gpu_backend_generate_frame_dbits_bgr24(
                p->backend,
                p->d_frame_bits, p->cfg.pay_bits,
                p->cfg.frame_width, p->cfg.frame_height,
                p->cfg.grid_cols, p->cfg.pay_rows,
                p->cfg.block_size,
                p->cfg.margin_x, p->cfg.margin_y,
                &p->cfg.cal_params);
            if (!d_frame_out) {
                if (error_out) snprintf(error_out, (size_t)error_size,
                                        "GPU frame gen failed");
                return false;
            }

            /* Sync frame gen before NVENC reads it */
            cudaStreamSynchronize(p->stream);
            double t_fg1 = (double)clock() / CLOCKS_PER_SEC;
            p->last_fg_time = t_fg1 - t_fg0;
            p->stats.frame_gen_ms += p->last_fg_time * 1000.0;

            /* NVENC encode (zero-copy ARGB path) */
            double t_nv0 = (double)clock() / CLOCKS_PER_SEC;
            const uint8_t *pkt = NULL;
            int pkt_size = 0;
            int ret = gpu_nvenc_encode_frame_zerocopy(p->nvenc,
                d_frame_out,
                p->cfg.frame_width * 3, /* BGR24 stride */
                &pkt, &pkt_size);
            if (ret < 0) {
                if (error_out) snprintf(error_out, (size_t)error_size,
                                        "NVENC encode failed");
                return false;
            }
            double t_nv1 = (double)clock() / CLOCKS_PER_SEC;
            p->last_nvenc_time = t_nv1 - t_nv0;
            p->stats.nvenc_ms += p->last_nvenc_time * 1000.0;

            /* Write encoded packet */
            if (pkt && pkt_size > 0) {
                if (!h264_writer_write_packet(writer, pkt, pkt_size)) {
                    if (error_out) snprintf(error_out, (size_t)error_size,
                                            "H.264 writer failed");
                    return false;
                }
            }

            /* Drain queued NVENC packets */
            const uint8_t *extra_pkt;
            int extra_sz;
            while ((extra_sz = gpu_nvenc_get_packet(p->nvenc, &extra_pkt)) > 0) {
                if (!h264_writer_write_packet(writer, extra_pkt, extra_sz)) {
                    if (error_out) snprintf(error_out, (size_t)error_size,
                                            "H.264 drain write failed");
                    return false;
                }
            }

            p->stats.total_frames_encoded++;
            if (frames_output) (*frames_output)++;
            s->frame_idx++;
        }
        s->drained = true;
        s->active  = false;
    }
    return true;
#else
    (void)p; (void)writer; (void)frames_output;
    if (error_out) snprintf(error_out, (size_t)error_size, "CUDA not compiled");
    return false;
#endif
}

/* ─── Flush remaining frames ──────────────────────────────────────────── */

int gpu_pipeline_flush(
    GpuEncodePipeline *p,
    H264Writer *writer,
    char *error_out, int error_size)
{
    int total = 0;

    /* Process partial final frame from any remaining slot data */
    for (int i = 0; i < 2; ++i) {
        PipeSlot *s = &p->slot[i];
        if (!s->active || !s->rs_done) continue;

        /* Process all remaining frames */
        while (s->frame_idx < s->num_frames) {
            int flushed = 0;
            if (!gpu_pipeline_drain(p, writer, &flushed, error_out, error_size))
                return -1;
            if (flushed == 0) break; /* prevent infinite loop */
            total += flushed;
        }

        /* Handle partial frame (remaining bits) */
        if (s->frame_idx >= s->num_frames) {
            int64_t used = (int64_t)s->frame_idx * (int64_t)p->cfg.pay_bits;
            int64_t rem  = s->bit_len - used;
            if (rem > 0 && rem < p->cfg.pay_bits) {
                /* Generate final padded frame */
                cudaMemsetAsync(p->d_frame_bits, 0,
                                (size_t)p->cfg.pay_bits, p->stream);
                cudaMemcpyAsync(p->d_frame_bits, s->h_bits + used,
                                (size_t)rem, cudaMemcpyHostToDevice, p->stream);
                cudaStreamSynchronize(p->stream);

                uint8_t *d_frame_out = gpu_backend_generate_frame_dbits_bgr24(
                    p->backend,
                    p->d_frame_bits, p->cfg.pay_bits,
                    p->cfg.frame_width, p->cfg.frame_height,
                    p->cfg.grid_cols, p->cfg.pay_rows,
                    p->cfg.block_size,
                    p->cfg.margin_x, p->cfg.margin_y,
                    &p->cfg.cal_params);
                if (d_frame_out) {
                    cudaStreamSynchronize(p->stream);
                    const uint8_t *pkt = NULL;
                    int pkt_size = 0;
                    if (gpu_nvenc_encode_frame_zerocopy(p->nvenc,
                            d_frame_out, p->cfg.frame_width * 3,
                            &pkt, &pkt_size) >= 0 && pkt && pkt_size > 0) {
                        h264_writer_write_packet(writer, pkt, pkt_size);
                        total++;
                        p->stats.total_frames_encoded++;
                    }
                    /* Drain any queued packets */
                    const uint8_t *extra_pkt;
                    int extra_sz;
                    while ((extra_sz = gpu_nvenc_get_packet(p->nvenc,
                             &extra_pkt)) > 0) {
                        h264_writer_write_packet(writer, extra_pkt, extra_sz);
                    }
                }
            }
        }
        s->active = false;
    }

    /* Final NVENC drain */
    gpu_nvenc_flush(p->nvenc);
    const uint8_t *pkt;
    int sz;
    while ((sz = gpu_nvenc_get_packet(p->nvenc, &pkt)) > 0) {
        h264_writer_write_packet(writer, pkt, sz);
    }

    return total;
}

/* ─── Statistics ───────────────────────────────────────────────────────── */

void gpu_pipeline_get_stats(const GpuEncodePipeline *p,
                             GpuPipelineStats *stats) {
    if (p && stats) memcpy(stats, &p->stats, sizeof(GpuPipelineStats));
}

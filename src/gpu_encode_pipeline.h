#ifndef VIDCRYPT_GPU_ENCODE_PIPELINE_H
#define VIDCRYPT_GPU_ENCODE_PIPELINE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "calibration.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * Streaming GPU encode pipeline orchestrator
 *
 * Manages a triple-buffered pipeline across three CUDA streams:
 *   Stream RS:   RS encode + bit expansion (GPU compute)
 *   Stream FG:   Frame generation (GPU compute → BGR24 output)
 *   Stream NVENC: NVENC hardware encode (ASIC)
 *
 * Each pipeline slot holds the data for one batch of frames.
 * Slots rotate through stages as work completes, synchronized by
 * CUDA events. This allows all three hardware units (CUDA SMs, CUDA
 * copy engines, NVENC ASIC) to operate concurrently.
 *
 * Architecture:
 *   ┌─────────────────────────────────────────────────────────┐
 *   │                  Pipeline Controller                     │
 *   │  ┌──────────┐   ┌──────────┐   ┌──────────┐           │
 *   │  │  Slot 0  │   │  Slot 1  │   │  Slot 2  │           │
 *   │  │ d_raw    │   │ d_raw    │   │ d_raw    │           │
 *   │  │ d_enc    │   │ d_enc    │   │ d_enc    │           │
 *   │  │ d_bits   │   │ d_bits   │   │ d_bits   │           │
 *   │  │ d_frame  │   │ d_frame  │   │ d_frame  │           │
 *   │  └────┬─────┘   └────┬─────┘   └────┬─────┘           │
 *   │       │              │              │                  │
 *   │       ▼              ▼              ▼                  │
 *   │  stream_rs ────▶ stream_fg ────▶ stream_nvenc         │
 *   │  (compute)      (compute)       (HW encode)           │
 *   └─────────────────────────────────────────────────────────┘
 *
 * The pipeline supports two modes:
 *   1. RS mode:    file → RS encode → bit expand → frame gen → NVENC
 *   2. No-RS mode: file → LUT expand → frame gen → NVENC
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Forward declarations */
struct GpuBackend;
typedef struct GpuBackend GpuBackend;
struct GpuNvencEncoder;
typedef struct GpuNvencEncoder GpuNvencEncoder;
struct H264Writer;
typedef struct H264Writer H264Writer;
struct GpuEncodePipeline;
typedef struct GpuEncodePipeline GpuEncodePipeline;

/* ─── Pipeline configuration ──────────────────────────────────────────── */

typedef struct {
    int frame_width;
    int frame_height;
    int grid_cols;
    int pay_rows;          /* payload rows per frame */
    int pay_bits;          /* payload bits per frame */
    int block_size;
    int margin_x;
    int margin_y;

    /* RS codec parameters (0 = no RS) */
    int rs_k;              /* message length */
    int rs_n_k;            /* parity symbols */
    int rs_n;              /* block length (255) */

    /* Batch size: number of codewords per pipeline slot.
     * Each codeword = 255 RS bytes = 255*8 bits.
     * Multiple codewords may be needed per frame. */
    int batch_codewords;   /* codewords per batch */
    int batch_frames;      /* frames per batch */

    /* NVENC parameters */
    const uint8_t *sps_pps;
    int sps_pps_size;

    /* Calibration parameters (for template render) */
    CalParams cal_params;

    /* Precomputed RS dictionary (CPU memory, owned by RSCodec cache) */
    const uint8_t *rs_dict;
    size_t rs_dict_size;
} GpuPipelineConfig;

/* ─── Pipeline lifecycle ──────────────────────────────────────────────── */

/* Create the pipeline. Allocates all GPU buffers for triple-buffered slots.
 * 'backend': initialized GPU backend.
 * 'nvenc': initialized NVENC encoder.
 * 'config': pipeline configuration.
 * 'error_out', 'error_size': error message buffer.
 * Returns NULL on failure. */
GpuEncodePipeline *gpu_pipeline_create(
    GpuBackend *backend,
    GpuNvencEncoder *nvenc,
    const GpuPipelineConfig *config,
    char *error_out, int error_size);

/* Destroy pipeline and free all GPU resources. */
void gpu_pipeline_destroy(GpuEncodePipeline *pipeline);

/* ─── Streaming data submission ────────────────────────────────────────── */

/* Submit raw file data to the pipeline for encoding.
 *
 * 'data': pointer to raw file bytes on CPU.
 * 'data_size': number of bytes in this chunk.
 * 'is_last': true if this is the final chunk.
 *
 * The pipeline copies data to pinned memory and uploads to the next
 * available GPU slot. RS encode + bit expansion + frame gen + NVENC
 * proceed asynchronously on GPU streams.
 *
 * Returns true on success. Caller should call gpu_pipeline_drain()
 * after each submission to retrieve encoded packets. */
bool gpu_pipeline_submit(
    GpuEncodePipeline *pipeline,
    const uint8_t *data, int64_t data_size, bool is_last,
    char *error_out, int error_size);

/* ─── Output retrieval ──────────────────────────────────────────────────── */

/* Drain completed encoded packets from the pipeline.
 *
 * 'writer': H264Writer to receive encoded AVCC packets.
 * 'frames_output': receives number of frames written this drain.
 *
 * Non-blocking: returns immediately with whatever frames are ready.
 * Returns false on write error. */
bool gpu_pipeline_drain(
    GpuEncodePipeline *pipeline,
    H264Writer *writer,
    int *frames_output,
    char *error_out, int error_size);

/* Flush all remaining frames through the pipeline.
 * Blocks until all in-flight work is complete.
 * Returns total frames flushed. */
int gpu_pipeline_flush(
    GpuEncodePipeline *pipeline,
    H264Writer *writer,
    char *error_out, int error_size);

/* ─── Statistics ────────────────────────────────────────────────────────── */

typedef struct {
    int64_t total_bytes_submitted;
    int     total_frames_encoded;
    int     total_codewords_encoded;
    double  rs_encode_ms;
    double  frame_gen_ms;
    double  nvenc_ms;
} GpuPipelineStats;

void gpu_pipeline_get_stats(const GpuEncodePipeline *pipeline,
                             GpuPipelineStats *stats);

#ifdef __cplusplus
}
#endif

#endif /* VIDCRYPT_GPU_ENCODE_PIPELINE_H */

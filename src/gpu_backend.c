#include "gpu_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include "gpu_kernels.h"
#include "gpu_presets.h"
#include "ffmpeg_hwdecoder.h"
#include "gpu_nvenc.h"
#include "calibration.h"

/* ─── Internal backend state ──────────────────────────────────────── */
struct GpuBackend {
    int             device_id;
    struct cudaDeviceProp prop;
    GpuPreset       preset;

    /* CUDA streams */
    cudaStream_t    stream_decode;
    cudaStream_t    stream_encode;
    cudaStream_t    stream_transfer;

    /* Memory pool — GPU device memory */
    uint8_t        *d_gray;           /* grayscale buffer: width * height */
    uint8_t        *d_bits;           /* bits buffer: max_bits bytes */
    uint8_t        *d_output_frame;   /* output BGR frame: width * height * 3 */
    uint8_t        *d_template;       /* pre-rendered encoder template frame */
    uint32_t       *d_cal_data;       /* calibration data (24 bytes = 6 uint32) */
    size_t          gray_size;
    size_t          frame_size;

    /* Memory pool — pinned host memory */
    uint8_t        *h_bits;           /* pinned: bits for CPU RS decode */
    int             h_bits_capacity;

    /* FFmpeg HW decoder (lazy init) */
    FfmpegHwDecoder *hwdec;          /* FFmpeg CUDA hwaccel decoder instance */
    int             decode_width;
    int             decode_height;
    bool            hwdec_active;    /* true when HW decoder is actively decoding */

    /* GPU-side bit extraction buffers */
    uint8_t        *d_bits_out;      /* GPU: extracted payload bits */
    uint8_t        *h_bits_out;      /* pinned: extracted bits for CPU */
    int             bits_capacity;   /* allocated size of bits buffers */

    /* NVENC encoder (lazy init) */
    GpuNvencEncoder *nvenc;
    int             encode_width;
    int             encode_height;
    double          encode_fps;
    uint8_t        *d_uv_fill;        /* GPU: NV12 UV plane filled with 128 */

    /* Template rendering state — separate from NVENC dimensions */
    int             template_width;
    int             template_height;
    bool            template_rendered;

    /* RS encode acceleration */
    uint8_t        *d_rs_dict;        /* GPU: uploaded parity dictionary */
    size_t          rs_dict_size;
    int             rs_k, rs_n_k;
    uint8_t        *d_enc_buf;        /* GPU: RS-encoded output */
    size_t          enc_buf_size;
    uint64_t       *d_bit_buf;        /* GPU: bit-expanded output */
    size_t          bit_buf_size;
};

/* ─── Memory pool allocation ─────────────────────────────────────── */

static bool alloc_pool(GpuBackend *b, int width, int height) {
    b->gray_size  = (size_t)width * (size_t)height;
    b->frame_size = (size_t)width * (size_t)height * 3;
    int max_bits  = 65536; /* enough for any grid configuration */

    cudaError_t e;
    e = cudaMalloc((void **)&b->d_gray,  b->gray_size);         if (e) goto fail;
    e = cudaMalloc((void **)&b->d_bits,  (size_t)max_bits);     if (e) goto fail;
    e = cudaMalloc((void **)&b->d_output_frame, b->frame_size); if (e) goto fail;
    e = cudaMalloc((void **)&b->d_template, b->frame_size);     if (e) goto fail;

    e = cudaHostAlloc((void **)&b->h_bits, (size_t)max_bits, cudaHostAllocDefault);
    if (e) goto fail;
    b->h_bits_capacity = max_bits;

    return true;

fail:
    cudaFree(b->d_gray);         b->d_gray = NULL;
    cudaFree(b->d_bits);         b->d_bits = NULL;
    cudaFree(b->d_output_frame); b->d_output_frame = NULL;
    cudaFree(b->d_template);     b->d_template = NULL;
    cudaFreeHost(b->h_bits);     b->h_bits = NULL;
    return false;
}

/* ─── Init / Destroy ─────────────────────────────────────────────── */

GpuStatus gpu_backend_init(GpuBackend **backend, int device_id,
                            char *error_out, int error_size) {
    GpuBackend *b = (GpuBackend *)calloc(1, sizeof(GpuBackend));
    if (!b) { *backend = NULL; return GPU_ERR_MEMORY; }
    b->device_id = device_id;

    cudaError_t e;

    /* ── CUDA context: adopt FFmpeg's context or create new ──────────
     * If ffmpeg_hwdecoder has already created a CUDA context (via Driver
     * API av_hwdevice_ctx_create), it's current from the cuCtxSetCurrent
     * call in ffmpeg_hwdecoder_open(). We skip cudaSetDevice to avoid
     * creating a SEPARATE primary context — all Runtime API calls
     * (cudaMalloc, cudaStreamCreate) will use the existing context.
     *
     * If no existing context (CPU fallback), create via cudaSetDevice. */
    CUcontext existing_ctx = NULL;
    if (cuCtxGetCurrent(&existing_ctx) == CUDA_SUCCESS && existing_ctx != NULL) {
        /* FFmpeg's context is current. Skip cudaSetDevice to adopt it.
         * All subsequent cudaMalloc/cudaStreamCreate use this context. */
    } else {
        e = cudaSetDevice(device_id);
        if (e != cudaSuccess) {
            if (error_out) snprintf(error_out, (size_t)error_size, "cudaSetDevice: %s", cudaGetErrorString(e));
            free(b); return GPU_ERR_NO_DEVICE;
        }
    }

    e = cudaGetDeviceProperties(&b->prop, device_id);
    if (e != cudaSuccess) {
        if (error_out) snprintf(error_out, (size_t)error_size, "cudaGetDeviceProperties: %s", cudaGetErrorString(e));
        free(b); return GPU_ERR_DRIVER;
    }

    /* Select preset based on detected compute capability */
    const GpuPreset *p = gpu_preset_lookup(b->prop.major, b->prop.minor);
    if (!p) {
        /* No GPU preset found — fall back gracefully */
        if (error_out) snprintf(error_out, (size_t)error_size,
                                "No preset for CC %d.%d", b->prop.major, b->prop.minor);
        free(b); return GPU_ERR_NO_DEVICE;
    }
    b->preset = *p;

    /* Create CUDA streams */
    cudaStreamCreate(&b->stream_decode);
    cudaStreamCreate(&b->stream_encode);
    cudaStreamCreate(&b->stream_transfer);

    /* Allocate default pool (will be reallocated on first use) */
    if (!alloc_pool(b, 1920, 1080)) {
        if (error_out) snprintf(error_out, (size_t)error_size, "Out of GPU memory");
        gpu_backend_destroy(b);
        return GPU_ERR_MEMORY;
    }

    /* Allocate calibration data buffer (24 bytes) */
    if (cudaMalloc((void **)&b->d_cal_data, 24) != cudaSuccess) {
        if (error_out) snprintf(error_out, (size_t)error_size, "Out of GPU memory (cal data)");
        gpu_backend_destroy(b);
        return GPU_ERR_MEMORY;
    }

    *backend = b;
    return GPU_INIT_OK;
}

void gpu_backend_destroy(GpuBackend *b) {
    if (!b) return;

    if (b->hwdec) { ffmpeg_hwdecoder_close(b->hwdec); b->hwdec = NULL; }
    if (b->nvenc) { gpu_nvenc_destroy(b->nvenc); b->nvenc = NULL; }

    cudaStreamDestroy(b->stream_decode);
    cudaStreamDestroy(b->stream_encode);
    cudaStreamDestroy(b->stream_transfer);

    cudaFree(b->d_gray);
    cudaFree(b->d_bits);
    cudaFree(b->d_bits_out);
    cudaFreeHost(b->h_bits_out);
    cudaFreeHost(b->h_bits);
    cudaFree(b->d_output_frame);
    cudaFree(b->d_template);
    cudaFree(b->d_cal_data);
    cudaFree(b->d_uv_fill);

    memset(b, 0, sizeof(GpuBackend));
    free(b);
}

const GpuPreset* gpu_backend_preset(const GpuBackend *b) { return &b->preset; }
const char* gpu_backend_device_name(const GpuBackend *b) { return b->prop.name; }

bool gpu_backend_open_nvdec(GpuBackend *b, const char *path,
                            char *error_out, int error_size) {
    if (!b) return false;
    gpu_backend_close_nvdec(b);

    /* Allocate GPU-side bit extraction buffers based on 4K max resolution */
    int max_tiles = 256; /* 16×16 grid — enough for any configuration */
    if (!b->d_bits_out) {
        cudaMalloc((void **)&b->d_bits_out, (size_t)max_tiles);
    }
    if (!b->h_bits_out) {
        cudaHostAlloc((void **)&b->h_bits_out, (size_t)max_tiles,
                       cudaHostAllocDefault);
    }
    b->bits_capacity = max_tiles;

    /* Ensure CUDA context is current (FFmpeg may have changed it) */
    cudaSetDevice(b->device_id);
    cudaFree(0);

    b->hwdec = ffmpeg_hwdecoder_open(path, b->device_id,
                                      error_out, error_size);
    b->hwdec_active = (b->hwdec != NULL);
    return b->hwdec_active;
}

void gpu_backend_close_nvdec(GpuBackend *b) {
    if (!b || !b->hwdec) return;
    ffmpeg_hwdecoder_close(b->hwdec);
    b->hwdec = NULL;
    b->hwdec_active = false;
}

bool gpu_backend_open_nvenc(GpuBackend *b, int width, int height,
                            double fps, int bitrate, int gop_length,
                            char *error_out, int error_size) {
    if (!b) return false;
    gpu_backend_close_nvenc(b);
    b->nvenc = gpu_nvenc_create(width, height, fps, b->device_id,
                                bitrate, gop_length,
                                error_out, error_size);
    if (!b->nvenc) return false;
    b->encode_width = width;
    b->encode_height = height;
    b->encode_fps = fps;

    /* Allocate UV fill plane (all 128 = neutral chroma for NV12) */
    size_t uv_size = (size_t)width * (size_t)height / 2;
    if (cudaMalloc((void **)&b->d_uv_fill, uv_size) != cudaSuccess) {
        gpu_backend_close_nvenc(b);
        if (error_out) snprintf(error_out, (size_t)error_size,
                                "cudaMalloc UV fill failed");
        return false;
    }
    if (cudaMemset(b->d_uv_fill, 128, uv_size) != cudaSuccess) {
        gpu_backend_close_nvenc(b);
        if (error_out) snprintf(error_out, (size_t)error_size,
                                "cudaMemset UV fill failed");
        return false;
    }

    return true;
}

void gpu_backend_close_nvenc(GpuBackend *b) {
    if (!b) return;
    if (b->nvenc) {
        gpu_nvenc_destroy(b->nvenc);
        b->nvenc = NULL;
    }
    if (b->d_uv_fill) {
        cudaFree(b->d_uv_fill);
        b->d_uv_fill = NULL;
    }
}

int gpu_backend_get_nvenc_packet(GpuBackend *b, const uint8_t **packet_out) {
    if (!b || !b->nvenc) {
        if (packet_out) *packet_out = NULL;
        return 0;
    }
    return gpu_nvenc_get_packet(b->nvenc, packet_out);
}

int gpu_backend_flush_nvenc(GpuBackend *b) {
    if (!b || !b->nvenc) return -1;
    return gpu_nvenc_flush(b->nvenc);
}

uint32_t* gpu_backend_calibration_buffer(GpuBackend *b) {
    return b ? b->d_cal_data : NULL;
}

int gpu_backend_get_nvenc_sps_pps(GpuBackend *b, const uint8_t **data_out) {
    if (!b || !b->nvenc) {
        if (data_out) *data_out = NULL;
        return 0;
    }
    return gpu_nvenc_get_sps_pps(b->nvenc, data_out);
}

GpuNvencEncoder* gpu_backend_get_nvenc_encoder(GpuBackend *b) {
    return b ? b->nvenc : NULL;
}

int gpu_backend_write_frame_zerocopy(GpuBackend *b,
                                      const uint8_t *d_frame_bgr24,
                                      int stride,
                                      const uint8_t **packet_out,
                                      int *packet_size_out) {
    if (!b || !b->nvenc || !d_frame_bgr24) {
        if (packet_out) *packet_out = NULL;
        if (packet_size_out) *packet_size_out = 0;
        return -1;
    }
    return gpu_nvenc_encode_frame_zerocopy(b->nvenc, d_frame_bgr24, stride,
                                            packet_out, packet_size_out);
}

bool gpu_backend_write_frame_nv12(GpuBackend *b,
                                   const uint8_t *d_y, int stride) {
    fprintf(stderr, "  [TRACE] write_frame_nv12: b=%p nvenc=%p d_y=%p uv=%p\n",
            (void*)b, (void*)(b ? b->nvenc : NULL), (void*)d_y,
            (void*)(b ? b->d_uv_fill : NULL));
    if (!b || !b->nvenc || !d_y || !b->d_uv_fill) {
        fprintf(stderr, "  write_frame_nv12: NULL CHECK FAILED\n");
        return false;
    }
    (void)stride;
    int result = gpu_nvenc_encode_frame_nv12(b->nvenc, d_y, b->d_uv_fill,
                                              (void *)b->stream_encode);
    fprintf(stderr, "  [TRACE] write_frame_nv12: result=%d\n", result);
    return result >= 0;
}

bool gpu_backend_write_frame_submit(GpuBackend *b,
                                     const uint8_t *d_y, int stride) {
    if (!b || !b->nvenc || !d_y || !b->d_uv_fill) return false;
    (void)stride;
    int result = gpu_nvenc_encode_frame_nv12_submit(b->nvenc, d_y, b->d_uv_fill,
                                                     (void *)b->stream_encode);
    return result == 0;
}

int gpu_backend_drain_nvenc(GpuBackend *b) {
    if (!b || !b->nvenc) return -1;
    return gpu_nvenc_drain_output(b->nvenc);
}

/* ─── Decode: FFmpeg HW → nv12_to_gray → extract_bits → CPU ──────── */

bool gpu_backend_decode_frame(GpuBackend *b,
                               const uint8_t *encoded_packet, int packet_size,
                               uint8_t **y_plane_out, int *pitch_out,
                               int *width_out, int *height_out) {
    (void)encoded_packet; (void)packet_size;

    /* FFmpeg CUDA hwaccel: ffmpeg_hwdecoder_read_frame() returns the
     * next decoded frame as a CUdeviceptr to the NV12 Y-plane in GPU
     * memory. FFmpeg handles demuxing, codec detection, and NVDEC
     * initialization. The returned pointer is valid until the next call.
     *
     * The Y-plane pointer is in CUDA memory and can be used directly
     * by the nv12_to_gray kernel — zero-copy.
     *
     * Caller should:
     *   1. Call gpu_backend_nv12_to_gray() to convert NV12 → grayscale
     *   2. Call gpu_backend_extract_bits() to decode tiles on GPU
     *      (eliminates full-frame CPU transfer)
     *   3. Transfer compact bits to CPU for RS decode */

    if (!b->hwdec || !b->hwdec_active) {
        *y_plane_out = NULL;
        *pitch_out = 0;
        *width_out = 0;
        *height_out = 0;
        return false;
    }

    return ffmpeg_hwdecoder_read_frame(b->hwdec, y_plane_out, pitch_out,
                                        width_out, height_out);
}

void gpu_backend_unmap_decode_frame(GpuBackend *b) {
    /* No explicit unmap needed — FFmpeg hwdecoder manages frame lifetime.
     * The previous frame's CUDA memory ref is released on the next call
     * to ffmpeg_hwdecoder_read_frame(). */
    (void)b;
}

uint8_t* gpu_backend_bgr24_to_gray(GpuBackend *b,
                                    const uint8_t *d_bgr, int bgr_stride,
                                    int width, int height) {
    /* Resize gray buffer if needed */
    size_t needed = (size_t)width * (size_t)height;
    if (needed > b->gray_size) {
        cudaFree(b->d_gray);
        if (cudaMalloc((void **)&b->d_gray, needed) != cudaSuccess) return NULL;
        b->gray_size = needed;
    }

    /* Launch BGR24 → grayscale kernel (extracts G channel) */
    dim3 block = {(unsigned int)b->preset.decode_block_x,
                  (unsigned int)b->preset.decode_block_y, 1};
    dim3 grid = {(unsigned int)((width  + block.x - 1) / block.x),
                 (unsigned int)((height + block.y - 1) / block.y), 1};

    bgr24_to_gray_kernel(grid, block, d_bgr, bgr_stride,
                          b->d_gray, width, width, height,
                          b->stream_decode);

    return b->d_gray;
}

uint8_t* gpu_backend_nv12_to_gray(GpuBackend *b,
                                   const uint8_t *nv12_y, int pitch,
                                   int width, int height) {
    /* Resize gray buffer if needed */
    size_t needed = (size_t)width * (size_t)height;
    if (needed > b->gray_size) {
        cudaFree(b->d_gray);
        if (cudaMalloc((void **)&b->d_gray, needed) != cudaSuccess) return NULL;
        b->gray_size = needed;
    }

    /* Launch CUDA kernel */
    dim3 block = {(unsigned int)b->preset.decode_block_x,
                  (unsigned int)b->preset.decode_block_y, 1};
    dim3 grid = {(unsigned int)((width  + block.x - 1) / block.x),
                 (unsigned int)((height + block.y - 1) / block.y), 1};

    /* nv12_to_gray_kernel — declared in gpu_kernels.h */
    nv12_to_gray_kernel(grid, block, nv12_y, pitch,
                         b->d_gray, width, width, height,
                         b->stream_decode);

    return b->d_gray;
}

/* ─── GPU-side bit extraction ────────────────────────────────────── */

/* Extract payload bits from a grayscale GPU frame.
 * Returns device pointer to the bits array (GPU memory).
 * The bits can be transferred to CPU via cudaMemcpy DeviceToHost.
 *
 * 'd_gray': device pointer to planar grayscale (width * height bytes).
 * 'gray_stride': bytes per row (typically = width for packed grayscale).
 * 'width', 'height': frame dimensions.
 * 'grid_top_y', 'grid_left_x': starting position of grid.
 * 'block_size': size of each tile.
 * 'grid_cols', 'pay_rows': grid dimensions (sync rows excluded).
 * 'subsample': pixel subsampling factor.
 * 'sync_rows': number of sync rows (typically 1).
 * Returns device pointer to bits array, or NULL on failure. */
uint8_t* gpu_backend_extract_bits(GpuBackend *b,
                                   const uint8_t *d_gray, int gray_stride,
                                   int width, int height,
                                   int grid_top_y, int grid_left_x,
                                   int block_size,
                                   int grid_cols, int pay_rows,
                                   int subsample, int sync_rows) {
    int total_bits = grid_cols * pay_rows;
    if (total_bits > b->bits_capacity) {
        /* Reallocate if needed (shouldn't happen with 256 max) */
        cudaFree(b->d_bits_out);
        cudaFreeHost(b->h_bits_out);
        b->bits_capacity = total_bits;
        cudaMalloc((void **)&b->d_bits_out, (size_t)total_bits);
        cudaHostAlloc((void **)&b->h_bits_out, (size_t)total_bits,
                       cudaHostAllocDefault);
    }

    /* Zero-initialize the output */
    cudaMemsetAsync(b->d_bits_out, 0, (size_t)total_bits, b->stream_decode);

    /* Launch one thread per tile */
    dim3 grid = {(unsigned int)grid_cols, (unsigned int)pay_rows, 1};
    dim3 block = {1, 1, 1};

    extract_bits_kernel(grid, block,
                         d_gray, gray_stride,
                         grid_top_y, grid_left_x,
                         block_size,
                         grid_cols, pay_rows,
                         subsample, sync_rows,
                         b->d_bits_out,
                         b->stream_decode);

    return b->d_bits_out;
}

/* Copy GPU-extracted bits to pinned CPU buffer for RS decode.
 * Returns pointer to host memory with extracted bits.
 * Uses stream_decode (same stream as the extract kernel) so the copy is
 * ordered after the kernel without needing explicit events.
 * Must call gpu_backend_sync() before reading the result. */
uint8_t* gpu_backend_get_bits_cpu(GpuBackend *b, int total_bits) {
    if (!b || !b->d_bits_out || !b->h_bits_out) return NULL;
    cudaMemcpyAsync(b->h_bits_out, b->d_bits_out,
                    (size_t)total_bits,
                    cudaMemcpyDeviceToHost, b->stream_decode);
    return b->h_bits_out;
}

/* GPU-side calibration extraction.
 * Extracts 192 calibration bits from a grayscale GPU frame.
 * Result is written to a pinned host buffer.
 * Returns the host pointer (192 bytes) or NULL on failure.
 * Must sync before reading. */
uint8_t* gpu_backend_extract_calibration(GpuBackend *b,
                                          const uint8_t *d_gray,
                                          int gray_stride,
                                          int width, int height) {
    if (!b || !d_gray) return NULL;

    /* Ensure we have space in the bits buffer (192 bytes) */
    if (192 > b->bits_capacity) {
        cudaFree(b->d_bits_out);
        cudaFreeHost(b->h_bits_out);
        b->bits_capacity = 192;
        cudaMalloc((void **)&b->d_bits_out, 192);
        cudaHostAlloc((void **)&b->h_bits_out, 192, cudaHostAllocDefault);
    }

    cudaMemsetAsync(b->d_bits_out, 0, 192, b->stream_decode);

    extract_calibration_bits_kernel(d_gray, gray_stride,
                                     width, height,
                                     b->d_bits_out,
                                     b->stream_decode);

    cudaMemcpyAsync(b->h_bits_out, b->d_bits_out, 192,
                    cudaMemcpyDeviceToHost, b->stream_decode);

    return b->h_bits_out;
}

/* ─── Encode: bits → frame_generate → NVENC/FFmpeg ──────────────── */

/* ─── Render template (calibration + sync row) once ──────────────── */
static bool ensure_template_rendered(GpuBackend *b, int w, int h,
                                      int grid_cols, int block_size,
                                      int margin_x, int margin_y,
                                      int payload_rows) {
    /* Use dedicated template_rendered flag — NOT encode_width/height
     * which are set by gpu_backend_open_nvenc for NVENC. */
    if (b->template_rendered && b->template_width == w && b->template_height == h) {
        return true; /* already rendered at this resolution */
    }
    /* Resize template buffer if needed */
    size_t bgr_size = (size_t)w * (size_t)h * 3;
    cudaFree(b->d_template);
    b->d_template = NULL;
    if (cudaMalloc((void **)&b->d_template, bgr_size) != cudaSuccess) return false;

    /* Build calibration params */
    CalParams params;
    memset(&params, 0, sizeof(params));
    params.frame_width      = (uint16_t)w;
    params.frame_height     = (uint16_t)h;
    params.margin_x         = (uint16_t)margin_x;
    params.margin_y         = (uint16_t)margin_y;
    params.block_size_x     = (uint16_t)block_size;
    params.block_size_y     = (uint16_t)block_size;
    params.grid_cols        = (uint16_t)grid_cols;
    params.grid_rows        = (uint16_t)(payload_rows + 1);
    params.rs_ecc_symbols   = DEFAULT_RS_ECC_SYMBOLS;
    params.rs_data_bytes    = DEFAULT_RS_DATA_BYTES;
    params.header_version   = 3;
    params.calibration_rows = CAL_ROWS;
    params.sync_rows        = 1;
    /* Render template on CPU (proven calibration rendering),
     * then upload to GPU. This avoids the GPU render_template kernel
     * which can have subtle pixel-value issues under H.264. */
    size_t gray_size = (size_t)w * (size_t)h;
    uint8_t *h_template_gray = (uint8_t *)calloc(1, gray_size);
    uint8_t *h_template_bgr  = (uint8_t *)calloc(1, bgr_size);
    if (!h_template_gray || !h_template_bgr) {
        free(h_template_gray); free(h_template_bgr);
        cudaFree(b->d_template); b->d_template = NULL;
        return false;
    }

    /* Render calibration dots on CPU */
    write_calibration_dots(h_template_gray, w, h, &params);

    /* Render sync row (alternating black/white tiles) */
    int cal_bottom = (int)(h * (CAL_TOP_FRAC + CAL_HEIGHT_FRAC));
    int sync_y = cal_bottom + margin_y;
    for (int col = 0; col < grid_cols; ++col) {
        uint8_t val = (col % 2) ? 255 : 0;
        int x0 = margin_x + col * block_size;
        int y0 = sync_y;
        int x_end = x0 + block_size;
        int y_end = y0 + block_size;
        if (x_end > w) x_end = w;
        if (y_end > h) y_end = h;
        for (int yy = y0; yy < y_end; ++yy)
            memset(h_template_gray + yy * w + x0, val, (size_t)(x_end - x0));
    }

    /* Place corner markers (8x8 checkerboard) for decoder self-alignment */
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            uint8_t cv = ((x + y) & 1) ? 255 : 0;
            h_template_gray[y * w + x] = cv;                          /* top-left */
            h_template_gray[y * w + (w - 8 + x)] = cv;               /* top-right */
            h_template_gray[(h - 8 + y) * w + x] = cv;               /* bottom-left */
            h_template_gray[(h - 8 + y) * w + (w - 8 + x)] = cv;    /* bottom-right */
        }
    }

    /* Convert grayscale to BGR24 (all 3 channels = gray value) */
    for (size_t i = 0; i < gray_size; ++i) {
        uint8_t v = h_template_gray[i];
        h_template_bgr[i * 3 + 0] = v;  /* B */
        h_template_bgr[i * 3 + 1] = v;  /* G */
        h_template_bgr[i * 3 + 2] = v;  /* R */
    }

    /* Upload BGR24 template to GPU */
    cudaMemcpy(b->d_template, h_template_bgr, bgr_size, cudaMemcpyHostToDevice);

    free(h_template_gray);
    free(h_template_bgr);

    b->template_width = w;
    b->template_height = h;
    b->template_rendered = true;
    return true;
}

uint8_t* gpu_backend_generate_frame(GpuBackend *b,
                                     const uint8_t *bits, int nbits,
                                     int width, int height,
                                     int grid_cols, int payload_rows,
                                     int block_size,
                                     int margin_x, int margin_y,
                                     const CalParams *params) {
    (void)params;
    /* Resize output frame buffer if needed */
    size_t frame_needed = (size_t)width * (size_t)height * 3;
    if (frame_needed > b->frame_size) {
        cudaFree(b->d_output_frame);
        b->d_output_frame = NULL;
        if (cudaMalloc((void **)&b->d_output_frame, frame_needed) != cudaSuccess) return NULL;
        b->frame_size = frame_needed;
    }

    /* Ensure template is rendered */
    if (!ensure_template_rendered(b, width, height,
                                   grid_cols, block_size,
                                   margin_x, margin_y,
                                   payload_rows)) {
        return NULL;
    }

    /* Copy template → output frame (fast cudaMemcpyAsync, ~1µs) */
    cudaMemcpyAsync(b->d_output_frame, b->d_template, frame_needed,
                    cudaMemcpyDeviceToDevice, b->stream_encode);

    /* Copy bits to GPU */
    cudaMemcpyAsync(b->d_bits, bits, (size_t)nbits,
                    cudaMemcpyHostToDevice, b->stream_encode);

    /* Payload region */
    int cal_bottom = (int)(height * (CAL_TOP_FRAC + CAL_HEIGHT_FRAC));
    int pay_y = cal_bottom + margin_y + block_size;
    int pay_x = margin_x;

    /* Launch frame generation kernel (paints payload blocks) */
    dim3 block_px = {(unsigned int)block_size, (unsigned int)block_size, 1};
    dim3 grid_px = {(unsigned int)grid_cols, (unsigned int)payload_rows, 1};

    frame_generate_kernel(grid_px, block_px,
                           b->d_output_frame, width * 3,
                           b->d_bits, pay_y, pay_x,
                           block_size, grid_cols, payload_rows,
                           b->stream_encode);

    return b->d_output_frame;
}

bool gpu_backend_write_frame(GpuBackend *b,
                              const uint8_t *gpu_frame, int stride) {
    if (!b || !b->nvenc || !gpu_frame) return false;

    /* Submit frame to NVENC encoder.
     * gpu_nvenc_encode_frame internally converts BGR24 → BGRA32 via
     * a CUDA kernel, copies to the NVENC input buffer, and submits.
     * The encoded bitstream is queued internally. */
    int result = gpu_nvenc_encode_frame(b->nvenc, gpu_frame, stride);
    return result >= 0;
}

/* ─── CPU grayscale upload / download ──────────────────────────── */

uint8_t* gpu_backend_upload_gray(GpuBackend *b,
                                  const uint8_t *gray, int stride,
                                  int width, int height) {
    if (!b || !gray) return NULL;
    size_t needed = (size_t)width * (size_t)height;
    if (needed > b->gray_size) {
        cudaFree(b->d_gray);
        if (cudaMalloc((void **)&b->d_gray, needed) != cudaSuccess) return NULL;
        b->gray_size = needed;
    }
    cudaError_t e = cudaMemcpy2DAsync(b->d_gray, (size_t)width,
                      gray, (size_t)stride,
                      (size_t)width, (size_t)height,
                      cudaMemcpyHostToDevice, b->stream_decode);
    if (e != cudaSuccess) return NULL;
    return b->d_gray;
}

/* Download a GPU grayscale frame to CPU memory.
 * Wraps cudaMemcpy2DAsync so it's compiled by nvcc (decoder.c uses MSVC). */
bool gpu_backend_download_gray(GpuBackend *b,
                                const uint8_t *d_gray, int gray_stride,
                                int width, int height,
                                uint8_t *h_dst, int dst_stride) {
    if (!b || !d_gray || !h_dst) return false;
    cudaError_t e = cudaMemcpy2DAsync(h_dst, (size_t)dst_stride,
                      d_gray, (size_t)gray_stride,
                      (size_t)width, (size_t)height,
                      cudaMemcpyDeviceToHost, b->stream_decode);
    return e == cudaSuccess;
}

/* ─── Calibration read (GPU accelerated) ──────────────────────────── */

bool gpu_backend_read_calibration(GpuBackend *b,
                                   const uint8_t *d_gray, int gray_stride,
                                   int width, int height,
                                   uint32_t *d_cal_data_out) {
    if (!b || !d_gray || !d_cal_data_out) return false;

    /* Zero-init the output buffer (6 uint32 = 24 bytes) */
    cudaMemsetAsync(d_cal_data_out, 0, 24, b->stream_decode);

    /* Launch read calibration dots kernel — 1 block, 192 threads */
    dim3 grid1 = {1, 1, 1};
    dim3 block192 = {192, 1, 1};
    read_calibration_dots_kernel(grid1, block192,
                                  d_gray, gray_stride,
                                  width, height,
                                  d_cal_data_out,
                                  b->stream_decode);

    return true;
}

void gpu_backend_sync(GpuBackend *b) {
    cudaStreamSynchronize(b->stream_decode);
    cudaStreamSynchronize(b->stream_encode);
    cudaStreamSynchronize(b->stream_transfer);
}

void gpu_backend_sync_encode(GpuBackend *b) {
    cudaStreamSynchronize(b->stream_encode);
}

void gpu_backend_sync_decode(GpuBackend *b) {
    cudaStreamSynchronize(b->stream_decode);
}

void* gpu_backend_get_decode_stream(GpuBackend *b) {
    return b ? (void*)b->stream_decode : NULL;
}

/* ─── GPU RS encode + device-bits frame gen ─────────────────────────── */

bool gpu_backend_upload_rs_dict(GpuBackend *b,
                                 const uint8_t *dict, size_t dict_size,
                                 int k, int n_k,
                                 char *error_out, int error_size) {
    if (!b || !dict || dict_size == 0) {
        if (error_out) snprintf(error_out, (size_t)error_size, "Invalid dict args");
        return false;
    }
    if (b->d_rs_dict && b->rs_dict_size != dict_size) {
        cudaFree(b->d_rs_dict); b->d_rs_dict = NULL; b->rs_dict_size = 0;
    }
    if (!b->d_rs_dict) {
        if (cudaMalloc((void **)&b->d_rs_dict, dict_size) != cudaSuccess) {
            if (error_out) snprintf(error_out, (size_t)error_size, "cudaMalloc dict failed");
            return false;
        }
    }
    if (cudaMemcpy(b->d_rs_dict, dict, dict_size, cudaMemcpyHostToDevice) != cudaSuccess) {
        if (error_out) snprintf(error_out, (size_t)error_size, "cudaMemcpy dict failed");
        return false;
    }
    b->rs_dict_size = dict_size; b->rs_k = k; b->rs_n_k = n_k;
    return true;
}

bool gpu_backend_rs_encode(GpuBackend *b,
                            const uint8_t *d_raw, int64_t raw_bytes,
                            int k, int n_k, int n,
                            uint8_t **d_enc_out, int *enc_bytes_out,
                            uint64_t **d_bits_out, int *bits_bytes_out,
                            char *error_out, int error_size) {
    if (!b || !d_raw || raw_bytes <= 0) return false;
    if (!b->d_rs_dict || b->rs_k != k || b->rs_n_k != n_k) {
        if (error_out) snprintf(error_out, (size_t)error_size, "RS dict not uploaded");
        return false;
    }
    int N = (int)((raw_bytes + k - 1) / k);
    int64_t enc_bytes = (int64_t)N * (int64_t)n;
    int64_t bits_bytes = enc_bytes * 8;
    if (enc_bytes > INT_MAX || bits_bytes > INT_MAX) return false;

    size_t enc_needed = (size_t)enc_bytes;
    if (!b->d_enc_buf || b->enc_buf_size < enc_needed) {
        if (b->d_enc_buf) cudaFree(b->d_enc_buf);
        if (cudaMalloc((void **)&b->d_enc_buf, enc_needed) != cudaSuccess) return false;
        b->enc_buf_size = enc_needed;
    }
    size_t bits_needed = (size_t)bits_bytes;
    if (!b->d_bit_buf || b->bit_buf_size < bits_needed) {
        if (b->d_bit_buf) cudaFree(b->d_bit_buf);
        if (cudaMalloc((void **)&b->d_bit_buf, bits_needed) != cudaSuccess) return false;
        b->bit_buf_size = bits_needed;
    }
    rs_encode_dict_kernel(d_raw, b->d_rs_dict, b->d_enc_buf, k, n_k, N, b->stream_encode);
    bit_expand_kernel(b->d_enc_buf, b->d_bit_buf, (int)enc_bytes, b->stream_encode);
    if (cudaStreamSynchronize(b->stream_encode) != cudaSuccess) return false;

    if (d_enc_out) *d_enc_out = b->d_enc_buf;
    if (enc_bytes_out) *enc_bytes_out = (int)enc_bytes;
    if (d_bits_out) *d_bits_out = b->d_bit_buf;
    if (bits_bytes_out) *bits_bytes_out = (int)bits_bytes;
    return true;
}

uint8_t* gpu_backend_generate_frame_dbits(GpuBackend *b,
                                           const uint8_t *d_bits, int nbits,
                                           int width, int height,
                                           int grid_cols, int payload_rows,
                                           int block_size,
                                           int margin_x, int margin_y,
                                           const CalParams *params) {
    size_t frame_needed = (size_t)width * (size_t)height;
    if (frame_needed > b->frame_size) {
        cudaFree(b->d_output_frame);
        if (cudaMalloc((void **)&b->d_output_frame, frame_needed) != cudaSuccess) return NULL;
        b->frame_size = frame_needed;
    }
    if (!ensure_template_rendered(b, width, height, grid_cols, block_size, margin_x, margin_y, payload_rows))
        return NULL;
    cudaMemcpyAsync(b->d_output_frame, b->d_template, frame_needed, cudaMemcpyDeviceToDevice, b->stream_encode);
    int cal_bottom = (int)(height * 0.06f);
    int pay_y = cal_bottom + margin_y + block_size;
    int pay_x = margin_x;
    dim3 block_px = {(unsigned int)block_size, (unsigned int)block_size, 1};
    dim3 grid_px = {(unsigned int)grid_cols, (unsigned int)payload_rows, 1};
    frame_generate_gray_kernel(grid_px, block_px, b->d_output_frame, width,
                                d_bits, pay_y, pay_x, block_size, grid_cols, payload_rows,
                                b->stream_encode);
    return b->d_output_frame;
}

/* BGR24 variant for NVENC ARGB pipeline (device-resident bits, no H2D copy). */
uint8_t* gpu_backend_generate_frame_dbits_bgr24(GpuBackend *b,
                                                 const uint8_t *d_bits, int nbits,
                                                 int width, int height,
                                                 int grid_cols, int payload_rows,
                                                 int block_size,
                                                 int margin_x, int margin_y,
                                                 const CalParams *params) {
    (void)params;
    size_t frame_needed = (size_t)width * (size_t)height * 3;
    if (frame_needed > b->frame_size) {
        cudaFree(b->d_output_frame);
        if (cudaMalloc((void **)&b->d_output_frame, frame_needed) != cudaSuccess)
            return NULL;
        b->frame_size = frame_needed;
    }
    if (!ensure_template_rendered(b, width, height, grid_cols, block_size,
                                   margin_x, margin_y, payload_rows))
        return NULL;
    /* Copy full BGR24 template */
    cudaMemcpyAsync(b->d_output_frame, b->d_template, frame_needed,
                    cudaMemcpyDeviceToDevice, b->stream_encode);
    int cal_bottom = (int)(height * 0.06f);
    int pay_y = cal_bottom + margin_y + block_size;
    int pay_x = margin_x;
    dim3 block_px = {(unsigned int)block_size, (unsigned int)block_size, 1};
    dim3 grid_px = {(unsigned int)grid_cols, (unsigned int)payload_rows, 1};
    /* Use BGR24 payload kernel (not grayscale) for NVENC ARGB pipeline.
     * CRITICAL: use the d_bits parameter (device-resident bit data passed
     * by caller), NOT b->d_bits (internal buffer that may have stale data). */
    frame_generate_kernel(grid_px, block_px, b->d_output_frame, width * 3,
                           d_bits, pay_y, pay_x, block_size, grid_cols, payload_rows,
                           b->stream_encode);
    (void)nbits;
    return b->d_output_frame;
}

#else /* !USE_CUDA — stubs */

GpuStatus gpu_backend_init(GpuBackend **backend, int device_id,
                            char *error_out, int error_size) {
    (void)device_id;
    *backend = NULL;
    if (error_out) snprintf(error_out, (size_t)error_size, "CUDA not compiled");
    return GPU_ERR_NO_DEVICE;
}

void gpu_backend_destroy(GpuBackend *backend) { (void)backend; }

const GpuPreset* gpu_backend_preset(const GpuBackend *b) { (void)b; return NULL; }
const char* gpu_backend_device_name(const GpuBackend *b) { (void)b; return "No GPU"; }

bool gpu_backend_open_nvdec(GpuBackend *b, const char *path,
                            char *error_out, int error_size) {
    (void)b; (void)path;
    if (error_out) snprintf(error_out, (size_t)error_size, "CUDA not compiled");
    return false;
}
void gpu_backend_close_nvdec(GpuBackend *b) { (void)b; }

bool gpu_backend_open_nvenc(GpuBackend *b, int width, int height,
                            double fps, int bitrate, int gop_length,
                            char *error_out, int error_size) {
    (void)b; (void)width; (void)height; (void)fps; (void)bitrate; (void)gop_length;
    if (error_out) snprintf(error_out, (size_t)error_size, "CUDA not compiled");
    return false;
}
void gpu_backend_close_nvenc(GpuBackend *b) { (void)b; }
int gpu_backend_get_nvenc_packet(GpuBackend *b, const uint8_t **packet_out) {
    (void)b; if (packet_out) *packet_out = NULL; return 0;
}
int gpu_backend_flush_nvenc(GpuBackend *b) { (void)b; return -1; }
int gpu_backend_get_nvenc_sps_pps(GpuBackend *b, const uint8_t **data_out) {
    (void)b; if (data_out) *data_out = NULL; return 0;
}
GpuNvencEncoder* gpu_backend_get_nvenc_encoder(GpuBackend *b) { (void)b; return NULL; }
int gpu_backend_write_frame_zerocopy(GpuBackend *b, const uint8_t *d, int s,
                                      const uint8_t **p, int *ps) {
    (void)b; (void)d; (void)s; if (p) *p = NULL; if (ps) *ps = 0; return -1;
}
bool gpu_backend_write_frame_nv12(GpuBackend *b, const uint8_t *d_y, int stride) {
    fprintf(stderr, "  [STUB] write_frame_nv12 called!\n");
    (void)b; (void)d_y; (void)stride; return false;
}
bool gpu_backend_write_frame_submit(GpuBackend *b, const uint8_t *d_y, int stride) {
    (void)b; (void)d_y; (void)stride; return false;
}
int gpu_backend_drain_nvenc(GpuBackend *b) { (void)b; return -1; }
uint32_t* gpu_backend_calibration_buffer(GpuBackend *b) { (void)b; return NULL; }

bool gpu_backend_decode_frame(GpuBackend *b, const uint8_t *p, int s,
                               uint8_t **y, int *pi, int *w, int *h) {
    (void)b; (void)p; (void)s; (void)y; (void)pi; (void)w; (void)h;
    return false;
}

void gpu_backend_unmap_decode_frame(GpuBackend *b) { (void)b; }

uint8_t* gpu_backend_bgr24_to_gray(GpuBackend *b,
                                    const uint8_t *d_bgr, int bgr_stride,
                                    int width, int height) {
    (void)b; (void)d_bgr; (void)bgr_stride; (void)width; (void)height;
    return NULL;
}

uint8_t* gpu_backend_nv12_to_gray(GpuBackend *b, const uint8_t *y,
                                   int p, int w, int h) {
    (void)b; (void)y; (void)p; (void)w; (void)h;
    return NULL;
}

uint8_t* gpu_backend_extract_bits(GpuBackend *b,
                                   const uint8_t *d_gray, int gray_stride,
                                   int width, int height,
                                   int grid_top_y, int grid_left_x,
                                   int block_size,
                                   int grid_cols, int pay_rows,
                                   int subsample, int sync_rows) {
    (void)b; (void)d_gray; (void)gray_stride; (void)width; (void)height;
    (void)grid_top_y; (void)grid_left_x; (void)block_size;
    (void)grid_cols; (void)pay_rows; (void)subsample; (void)sync_rows;
    return NULL;
}

uint8_t* gpu_backend_get_bits_cpu(GpuBackend *b, int total_bits) {
    (void)b; (void)total_bits; return NULL;
}

uint8_t* gpu_backend_extract_calibration(GpuBackend *b,
                                          const uint8_t *d_gray,
                                          int gray_stride,
                                          int width, int height) {
    (void)b; (void)d_gray; (void)gray_stride; (void)width; (void)height;
    return NULL;
}

uint8_t* gpu_backend_generate_frame(GpuBackend *b, const uint8_t *bits,
                                     int nbits, int w, int h,
                                     int gc, int pr, int bs, int mx, int my) {
    (void)b; (void)bits; (void)nbits; (void)w; (void)h;
    (void)gc; (void)pr; (void)bs; (void)mx; (void)my;
    return NULL;
}

bool gpu_backend_write_frame(GpuBackend *b, const uint8_t *f, int s) {
    (void)b; (void)f; (void)s; return false;
}

bool gpu_backend_read_calibration(GpuBackend *b,
                                   const uint8_t *d_gray, int gray_stride,
                                   int width, int height,
                                   uint32_t *d_cal_data_out) {
    (void)b; (void)d_gray; (void)gray_stride;
    (void)width; (void)height; (void)d_cal_data_out;
    return false;
}

uint8_t* gpu_backend_upload_gray(GpuBackend *b, const uint8_t *gray,
                                    int stride, int width, int height) {
    (void)b; (void)gray; (void)stride; (void)width; (void)height;
    return NULL;
}

bool gpu_backend_download_gray(GpuBackend *b,
                                const uint8_t *d_gray, int gray_stride,
                                int width, int height,
                                uint8_t *h_dst, int dst_stride) {
    (void)b; (void)d_gray; (void)gray_stride;
    (void)width; (void)height; (void)h_dst; (void)dst_stride;
    return false;
}

void gpu_backend_sync(GpuBackend *b) { (void)b; }

void* gpu_backend_get_decode_stream(GpuBackend *b) { (void)b; return NULL; }

#endif /* USE_CUDA */

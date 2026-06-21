#include "gpu_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_CUDA

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

    e = cudaSetDevice(device_id);
    if (e != cudaSuccess) {
        if (error_out) snprintf(error_out, (size_t)error_size, "cudaSetDevice: %s", cudaGetErrorString(e));
        free(b); return GPU_ERR_NO_DEVICE;
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
    return true;
}

void gpu_backend_close_nvenc(GpuBackend *b) {
    if (!b || !b->nvenc) return;
    gpu_nvenc_destroy(b->nvenc);
    b->nvenc = NULL;
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
    if (b->d_template && b->encode_width == w && b->encode_height == h) {
        return true; /* already rendered at this resolution */
    }
    /* Resize template buffer if needed */
    size_t needed = (size_t)w * (size_t)h * 3;
    cudaFree(b->d_template);
    b->d_template = NULL;
    if (cudaMalloc((void **)&b->d_template, needed) != cudaSuccess) return false;

    /* Allocate/upload calibration data */
    if (!b->d_cal_data) {
        if (cudaMalloc((void **)&b->d_cal_data, 24) != cudaSuccess) {
            cudaFree(b->d_template);
            b->d_template = NULL;
            return false;
        }
    }
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

    uint8_t cal_bytes[24];
    build_calibration_bytes(&params, cal_bytes);
    cudaMemcpy(b->d_cal_data, cal_bytes, 24, cudaMemcpyHostToDevice);

    /* Launch template render kernel (grid ignored, recomputed from dims) */
    dim3 grid1 = {1, 1, 1};
    dim3 block32_16 = {32, 16, 1};
    render_template_kernel(grid1, block32_16, b->d_template, w * 3,
                            w, h, grid_cols, block_size,
                            margin_x, margin_y,
                            b->d_cal_data,
                            b->stream_encode);
    b->encode_width = w;
    b->encode_height = h;
    return true;
}

uint8_t* gpu_backend_generate_frame(GpuBackend *b,
                                     const uint8_t *bits, int nbits,
                                     int width, int height,
                                     int grid_cols, int payload_rows,
                                     int block_size,
                                     int margin_x, int margin_y) {
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

void gpu_backend_sync(GpuBackend *b) { (void)b; }

#endif /* USE_CUDA */

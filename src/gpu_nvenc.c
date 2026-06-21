#include "gpu_nvenc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#ifdef USE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvEncodeAPI.h>
#include "gpu_kernels.h"

/* Platform-specific dynamic library loading */
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

/* ─── Internal state ──────────────────────────────────────────────── */
struct GpuNvencEncoder {
    /* CUDA state */
    int             cuda_device;
    CUcontext       cuda_ctx;

    /* NVENC API function table */
    NV_ENCODE_API_FUNCTION_LIST nvenc_api;

    /* Encoder session */
    void           *encoder;         /* NV_ENC_SESSION_HANDLE (void*) */

    /* Video parameters */
    int             width;
    int             height;
    double          fps;
    int             bitrate;         /* kbps */
    int             gop_length;

    /* NVENC input buffer */
    NV_ENC_INPUT_PTR input_buffer;   /* Input surface for raw frames */
    int             input_buffer_pitch;  /* bytes per row */

    /* NVENC output bitstream buffer */
    NV_ENC_OUTPUT_PTR output_buffer;

    /* Convert buffer: BGR24 → BGRA32 conversion surface */
    CUdeviceptr     d_bgra;              /* device memory: BGRA32 frame */
    size_t          bgra_size;
    cudaStream_t    stream;

    /* Packet ring buffer — stores encoded packets for retrieval */
    uint8_t       **packet_data;
    int            *packet_sizes;
    int            *packet_capacities;
    int             packet_count;
    int             packet_capacity;

    /* Statistics */
    int64_t         total_encoded_bytes;
    int             total_encoded_frames;

    /* Error buffer */
    char            error_buf[256];
};

/* ─── Error reporting ─────────────────────────────────────────────── */
#define SET_ERR(enc, fmt, ...) do { \
    snprintf((enc)->error_buf, sizeof((enc)->error_buf), fmt, ##__VA_ARGS__); \
} while(0)

/* ─── GUID definitions (from nvEncodeAPI.h) ───────────────────────── */
/* These GUIDs are defined in the NVENC SDK header. We define them here
 * as a fallback for compilation without the full SDK headers path. */

/* ─── Packet ring buffer management ───────────────────────────────── */

static bool add_packet(GpuNvencEncoder *enc, const uint8_t *data, int size) {
    if (enc->packet_count >= enc->packet_capacity) {
        int new_cap = enc->packet_capacity == 0 ? 64 : enc->packet_capacity * 2;
        uint8_t **new_data = (uint8_t **)realloc(enc->packet_data,
                                                   (size_t)new_cap * sizeof(uint8_t *));
        int *new_sizes = (int *)realloc(enc->packet_sizes,
                                         (size_t)new_cap * sizeof(int));
        int *new_caps = (int *)realloc(enc->packet_capacities,
                                        (size_t)new_cap * sizeof(int));
        if (!new_data || !new_sizes || !new_caps) return false;
        enc->packet_data = new_data;
        enc->packet_sizes = new_sizes;
        enc->packet_capacities = new_caps;
        /* Zero-initialize new entries */
        memset(enc->packet_data + enc->packet_capacity, 0,
               (size_t)(new_cap - enc->packet_capacity) * sizeof(uint8_t *));
        memset(enc->packet_sizes + enc->packet_capacity, 0,
               (size_t)(new_cap - enc->packet_capacity) * sizeof(int));
        memset(enc->packet_capacities + enc->packet_capacity, 0,
               (size_t)(new_cap - enc->packet_capacity) * sizeof(int));
        enc->packet_capacity = new_cap;
    }

    int idx = enc->packet_count;
    if (size > enc->packet_capacities[idx]) {
        free(enc->packet_data[idx]);
        enc->packet_data[idx] = (uint8_t *)malloc((size_t)size);
        if (!enc->packet_data[idx]) return false;
        enc->packet_capacities[idx] = size;
    }
    memcpy(enc->packet_data[idx], data, (size_t)size);
    enc->packet_sizes[idx] = size;
    enc->packet_count++;
    enc->total_encoded_bytes += size;
    return true;
}

/* ─── BGR24 → BGRA32 CUDA kernel ──────────────────────────────────── */
/* The kernel is defined in gpu_kernels.cu and declared in gpu_kernels.h.
 * It is called via bgr24_to_bgra32_kernel() from this file. */

/* ─── NVENC API loading ───────────────────────────────────────────── */

/* Load the NVENC API via NvEncodeAPICreateInstance.
 * This function is exported from nvEncodeAPI.dll / libnvcuvid.so. */
typedef NVENCSTATUS (NVENCAPI *NvEncodeAPICreateInstanceFunc)(NV_ENCODE_API_FUNCTION_LIST *);
typedef CUresult (CUDAAPI *CuCtxGetCurrentFunc)(CUcontext *);

static NvEncodeAPICreateInstanceFunc    load_nvenc_api(void) {
#ifdef _WIN32
    HMODULE mod = LoadLibraryA("nvEncodeAPI64.dll");
    if (!mod) mod = LoadLibraryA("nvEncodeAPI.dll");
    if (!mod) return NULL;
    return (NvEncodeAPICreateInstanceFunc)GetProcAddress(mod, "NvEncodeAPICreateInstance");
#else
    void *handle = dlopen("libnvidia-encode.so.1", RTLD_LAZY);
    if (!handle) handle = dlopen("libnvidia-encode.so", RTLD_LAZY);
    if (!handle) return NULL;
    return (NvEncodeAPICreateInstanceFunc)dlsym(handle, "NvEncodeAPICreateInstance");
#endif
}

static CuCtxGetCurrentFunc load_cu_ctx_get_current(void) {
#ifdef _WIN32
    HMODULE mod = LoadLibraryA("nvcuda.dll");
    if (!mod) return NULL;
    return (CuCtxGetCurrentFunc)GetProcAddress(mod, "cuCtxGetCurrent");
#else
    void *handle = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!handle) handle = dlopen("libcuda.so", RTLD_LAZY);
    if (!handle) return NULL;
    return (CuCtxGetCurrentFunc)dlsym(handle, "cuCtxGetCurrent");
#endif
}

/* ─── NVENC error string ──────────────────────────────────────────── */
static const char* nvenc_error_string(NVENCSTATUS status) {
    switch (status) {
        case NV_ENC_SUCCESS: return "NV_ENC_SUCCESS";
        case NV_ENC_ERR_NO_ENCODE_DEVICE: return "NV_ENC_ERR_NO_ENCODE_DEVICE";
        case NV_ENC_ERR_UNSUPPORTED_DEVICE: return "NV_ENC_ERR_UNSUPPORTED_DEVICE";
        case NV_ENC_ERR_INVALID_ENCODERDEVICE: return "NV_ENC_ERR_INVALID_ENCODERDEVICE";
        case NV_ENC_ERR_INVALID_DEVICE: return "NV_ENC_ERR_INVALID_DEVICE";
        case NV_ENC_ERR_DEVICE_NOT_EXIST: return "NV_ENC_ERR_DEVICE_NOT_EXIST";
        case NV_ENC_ERR_UNSUPPORTED_PARAM: return "NV_ENC_ERR_UNSUPPORTED_PARAM";
        case NV_ENC_ERR_OUT_OF_MEMORY: return "NV_ENC_ERR_OUT_OF_MEMORY";
        case NV_ENC_ERR_INVALID_PTR: return "NV_ENC_ERR_INVALID_PTR";
        case NV_ENC_ERR_INVALID_PARAM: return "NV_ENC_ERR_INVALID_PARAM";
        case NV_ENC_ERR_INVALID_VERSION: return "NV_ENC_ERR_INVALID_VERSION";
        case NV_ENC_ERR_ENCODER_NOT_INITIALIZED: return "NV_ENC_ERR_ENCODER_NOT_INITIALIZED";
        case NV_ENC_ERR_GENERIC: return "NV_ENC_ERR_GENERIC";
        default: return "Unknown NVENC error";
    }
}

/* ─── Public API ──────────────────────────────────────────────────── */

GpuNvencEncoder* gpu_nvenc_create(int width, int height, double fps,
                                   int cuda_device, int bitrate,
                                   int gop_length,
                                   char *error_out, int error_size) {
    if (width <= 0 || height <= 0 || fps <= 0.0) {
        if (error_out) snprintf(error_out, (size_t)error_size, "Invalid parameters");
        return NULL;
    }

    GpuNvencEncoder *enc = (GpuNvencEncoder *)calloc(1, sizeof(GpuNvencEncoder));
    if (!enc) {
        if (error_out) snprintf(error_out, (size_t)error_size, "Out of memory");
        return NULL;
    }

    enc->width = width;
    enc->height = height;
    enc->fps = fps;
    enc->cuda_device = cuda_device;
    enc->bitrate = bitrate > 0 ? bitrate : 8000; /* default: 8 Mbps for 1080p */
    enc->gop_length = gop_length > 0 ? gop_length : (int)(fps * 2); /* ~2 sec GOP */

    /* ── 1. Set CUDA context ──────────────────────────────────────── */
    cudaError_t rt_err = cudaSetDevice(cuda_device);
    if (rt_err != cudaSuccess) {
        if (error_out) snprintf(error_out, (size_t)error_size,
                                "cudaSetDevice failed: %s", cudaGetErrorString(rt_err));
        free(enc);
        return NULL;
    }

    rt_err = cudaFree(0);
    if (rt_err != cudaSuccess) {
        free(enc);
        if (error_out) snprintf(error_out, (size_t)error_size,
                                "CUDA context init failed: %s", cudaGetErrorString(rt_err));
        return NULL;
    }

    CuCtxGetCurrentFunc cu_ctx_get_current = load_cu_ctx_get_current();
    if (!cu_ctx_get_current ||
        cu_ctx_get_current(&enc->cuda_ctx) != CUDA_SUCCESS ||
        !enc->cuda_ctx) {
        free(enc);
        if (error_out) snprintf(error_out, (size_t)error_size,
                                "Cannot get current CUDA context from nvcuda.dll");
        return NULL;
    }

    /* ── 2. Load NVENC API ────────────────────────────────────────── */
    NvEncodeAPICreateInstanceFunc create_instance = load_nvenc_api();
    if (!create_instance) {
        free(enc);
        if (error_out) snprintf(error_out, (size_t)error_size,
                                "Cannot load nvEncodeAPI64.dll");
        return NULL;
    }

    memset(&enc->nvenc_api, 0, sizeof(enc->nvenc_api));
    enc->nvenc_api.version = NV_ENCODE_API_FUNCTION_LIST_VER;

    NVENCSTATUS nv_status = create_instance(&enc->nvenc_api);
    if (nv_status != NV_ENC_SUCCESS) {
        free(enc);
        if (error_out) snprintf(error_out, (size_t)error_size,
                                "NvEncodeAPICreateInstance: %s",
                                nvenc_error_string(nv_status));
        return NULL;
    }

    /* ── 3. Open encode session ────────────────────────────────────── */
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS session_params;
    memset(&session_params, 0, sizeof(session_params));
    session_params.version    = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
    session_params.device     = enc->cuda_ctx;
    session_params.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
    session_params.apiVersion = NVENCAPI_VERSION;

    nv_status = enc->nvenc_api.nvEncOpenEncodeSessionEx(&session_params, &enc->encoder);
    if (nv_status != NV_ENC_SUCCESS) {
        free(enc);
        if (error_out) snprintf(error_out, (size_t)error_size,
                                "nvEncOpenEncodeSessionEx: %s",
                                nvenc_error_string(nv_status));
        return NULL;
    }

    /* ── 4. Initialize encoder ─────────────────────────────────────── */
    /* First query capabilities to determine if H.264 is supported */
    NV_ENC_CAPS_PARAM caps_param;
    memset(&caps_param, 0, sizeof(caps_param));
    caps_param.version    = NV_ENC_CAPS_PARAM_VER;
    caps_param.capsToQuery = NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES;

    int caps_val = 0;
    nv_status = enc->nvenc_api.nvEncGetEncodeCaps(enc->encoder,
                                                   NV_ENC_CODEC_H264_GUID,
                                                   &caps_param, &caps_val);

    NV_ENC_TUNING_INFO tuning_info = NV_ENC_TUNING_INFO_LOW_LATENCY;

    NV_ENC_PRESET_CONFIG preset_config;
    memset(&preset_config, 0, sizeof(preset_config));
    preset_config.version = NV_ENC_PRESET_CONFIG_VER;
    preset_config.presetCfg.version = NV_ENC_CONFIG_VER;

    nv_status = enc->nvenc_api.nvEncGetEncodePresetConfigEx(
        enc->encoder,
        NV_ENC_CODEC_H264_GUID,
        NV_ENC_PRESET_P1_GUID,
        tuning_info,
        &preset_config);
    if (nv_status != NV_ENC_SUCCESS) {
        enc->nvenc_api.nvEncDestroyEncoder(enc->encoder);
        free(enc);
        if (error_out) snprintf(error_out, (size_t)error_size,
                                "nvEncGetEncodePresetConfigEx: %s",
                                nvenc_error_string(nv_status));
        return NULL;
    }

    NV_ENC_CONFIG encode_config = preset_config.presetCfg;
    encode_config.profileGUID = NV_ENC_H264_PROFILE_HIGH_GUID;

    /* Rate control: CBR (constant bitrate) */
    encode_config.encodeCodecConfig.h264Config.entropyCodingMode =
        NV_ENC_H264_ENTROPY_CODING_MODE_CABAC;
    encode_config.encodeCodecConfig.h264Config.outputAUD = 1; /* emit access unit delimiters */
    /* Note: NVENC outputs in AVCC format (4-byte length-prefixed NALUs).
     * The encoder caller is responsible for converting to Annex B
     * (00 00 00 01 start codes) before writing raw .h264 output. */
    encode_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
    encode_config.rcParams.averageBitRate  = enc->bitrate * 1000; /* bps */
    encode_config.rcParams.maxBitRate      = enc->bitrate * 1000;
    encode_config.rcParams.vbvBufferSize   = enc->bitrate * 1000 * 2; /* 2 sec buffer */
    encode_config.rcParams.vbvInitialDelay = enc->bitrate * 1000 * 1; /* 1 sec delay */

    /* Frame rate */
    encode_config.frameIntervalP = 1;
    encode_config.gopLength = enc->gop_length;

    /* Initialize params */
    NV_ENC_INITIALIZE_PARAMS init_params;
    memset(&init_params, 0, sizeof(init_params));
    init_params.version          = NV_ENC_INITIALIZE_PARAMS_VER;
    init_params.encodeGUID       = NV_ENC_CODEC_H264_GUID;
    init_params.presetGUID       = NV_ENC_PRESET_P1_GUID; /* fastest */
    init_params.encodeWidth      = width;
    init_params.encodeHeight     = height;
    init_params.darWidth         = width;
    init_params.darHeight        = height;
    init_params.maxEncodeWidth   = width;
    init_params.maxEncodeHeight  = height;
    init_params.frameRateNum     = (uint32_t)(fps * 1000);
    init_params.frameRateDen     = 1000;
    init_params.enableEncodeAsync = 0;
    init_params.enablePTD        = 1;
    init_params.tuningInfo       = tuning_info;
    init_params.encodeConfig     = &encode_config;

    nv_status = enc->nvenc_api.nvEncInitializeEncoder(enc->encoder, &init_params);
    if (nv_status != NV_ENC_SUCCESS) {
        enc->nvenc_api.nvEncDestroyEncoder(enc->encoder);
        free(enc);
        if (error_out) snprintf(error_out, (size_t)error_size,
                                "nvEncInitializeEncoder: %s",
                                nvenc_error_string(nv_status));
        return NULL;
    }

    /* ── 5. Create input buffer ────────────────────────────────────── */
    /* Use ARGB format (4 bytes per pixel). Our frames are BGR24, which
     * we convert to BGRA32 (A=0xFF) via a CUDA kernel before submission. */
    NV_ENC_BUFFER_FORMAT buffer_fmt = NV_ENC_BUFFER_FORMAT_ARGB;

    NV_ENC_CREATE_INPUT_BUFFER alloc_input;
    memset(&alloc_input, 0, sizeof(alloc_input));
    alloc_input.version      = NV_ENC_CREATE_INPUT_BUFFER_VER;
    alloc_input.width        = width;
    alloc_input.height       = height;
    alloc_input.memoryHeap   = NV_ENC_MEMORY_HEAP_AUTOSELECT;
    alloc_input.bufferFmt    = buffer_fmt;

    nv_status = enc->nvenc_api.nvEncCreateInputBuffer(enc->encoder, &alloc_input);
    if (nv_status != NV_ENC_SUCCESS) {
        enc->nvenc_api.nvEncDestroyEncoder(enc->encoder);
        free(enc);
        if (error_out) snprintf(error_out, (size_t)error_size,
                                "nvEncCreateInputBuffer: %s",
                                nvenc_error_string(nv_status));
        return NULL;
    }
    enc->input_buffer = alloc_input.inputBuffer;
    enc->input_buffer_pitch = width * 4; /* ARGB = 4 bytes/pixel */

    /* ── 6. Create output bitstream buffer ─────────────────────────── */
    NV_ENC_CREATE_BITSTREAM_BUFFER alloc_output;
    memset(&alloc_output, 0, sizeof(alloc_output));
    alloc_output.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;

    nv_status = enc->nvenc_api.nvEncCreateBitstreamBuffer(enc->encoder, &alloc_output);
    if (nv_status != NV_ENC_SUCCESS) {
        enc->nvenc_api.nvEncDestroyInputBuffer(enc->encoder, enc->input_buffer);
        enc->nvenc_api.nvEncDestroyEncoder(enc->encoder);
        free(enc);
        if (error_out) snprintf(error_out, (size_t)error_size,
                                "nvEncCreateBitstreamBuffer: %s",
                                nvenc_error_string(nv_status));
        return NULL;
    }
    enc->output_buffer = alloc_output.bitstreamBuffer;

    /* ── 7. Allocate BGRA32 conversion buffer ──────────────────────── */
    enc->bgra_size = (size_t)width * (size_t)height * 4;
    rt_err = cudaMalloc((void **)&enc->d_bgra, enc->bgra_size);
    if (rt_err != cudaSuccess) {
        enc->nvenc_api.nvEncDestroyBitstreamBuffer(enc->encoder, enc->output_buffer);
        enc->nvenc_api.nvEncDestroyInputBuffer(enc->encoder, enc->input_buffer);
        enc->nvenc_api.nvEncDestroyEncoder(enc->encoder);
        free(enc);
        if (error_out) snprintf(error_out, (size_t)error_size,
                                "cudaMalloc BGRA32: %s", cudaGetErrorString(rt_err));
        return NULL;
    }

    if (cudaStreamCreate(&enc->stream) != cudaSuccess) {
        cudaFree((void *)enc->d_bgra);
        enc->nvenc_api.nvEncDestroyBitstreamBuffer(enc->encoder, enc->output_buffer);
        enc->nvenc_api.nvEncDestroyInputBuffer(enc->encoder, enc->input_buffer);
        enc->nvenc_api.nvEncDestroyEncoder(enc->encoder);
        free(enc);
        if (error_out) snprintf(error_out, (size_t)error_size,
                                "cudaStreamCreate NVENC conversion stream failed");
        return NULL;
    }

    /* ── Initialize packet ring buffer ─────────────────────────────── */
    enc->packet_capacity = 64;
    enc->packet_data = (uint8_t **)calloc((size_t)enc->packet_capacity, sizeof(uint8_t *));
    enc->packet_sizes = (int *)calloc((size_t)enc->packet_capacity, sizeof(int));
    enc->packet_capacities = (int *)calloc((size_t)enc->packet_capacity, sizeof(int));
    if (!enc->packet_data || !enc->packet_sizes || !enc->packet_capacities) {
        free(enc->packet_data); free(enc->packet_sizes); free(enc->packet_capacities);
        cudaStreamDestroy(enc->stream);
        cudaFree((void *)enc->d_bgra);
        enc->nvenc_api.nvEncDestroyBitstreamBuffer(enc->encoder, enc->output_buffer);
        enc->nvenc_api.nvEncDestroyInputBuffer(enc->encoder, enc->input_buffer);
        enc->nvenc_api.nvEncDestroyEncoder(enc->encoder);
        free(enc);
        if (error_out) snprintf(error_out, (size_t)error_size, "Out of memory");
        return NULL;
    }

    if (error_out) snprintf(error_out, (size_t)error_size, "");
    return enc;
}

void gpu_nvenc_destroy(GpuNvencEncoder *enc) {
    if (!enc) return;

    /* Destroy NVENC resources */
    if (enc->output_buffer && enc->nvenc_api.nvEncDestroyBitstreamBuffer) {
        enc->nvenc_api.nvEncDestroyBitstreamBuffer(enc->encoder, enc->output_buffer);
    }
    if (enc->input_buffer && enc->nvenc_api.nvEncDestroyInputBuffer) {
        enc->nvenc_api.nvEncDestroyInputBuffer(enc->encoder, enc->input_buffer);
    }
    if (enc->encoder && enc->nvenc_api.nvEncDestroyEncoder) {
        enc->nvenc_api.nvEncDestroyEncoder(enc->encoder);
    }

    /* Free CUDA resources */
    if (enc->stream) cudaStreamDestroy(enc->stream);
    if (enc->d_bgra) cudaFree((void *)enc->d_bgra);

    /* Free packet buffers */
    if (enc->packet_data) {
        for (int i = 0; i < enc->packet_capacity; ++i) {
            free(enc->packet_data[i]);
        }
        free(enc->packet_data);
    }
    free(enc->packet_sizes);
    free(enc->packet_capacities);

    memset(enc, 0, sizeof(GpuNvencEncoder));
    free(enc);
}

int gpu_nvenc_encode_frame(GpuNvencEncoder *enc,
                            const uint8_t *gpu_frame_bgr24, int stride) {
    if (!enc || !gpu_frame_bgr24) return -1;

    /* ── 1. Convert BGR24 → BGRA32 on GPU ─────────────────────────── */
    dim3 block = {32, 16};
    dim3 grid = {(unsigned int)((enc->width  + 32 - 1) / 32),
                 (unsigned int)((enc->height + 16 - 1) / 16)};

    bgr24_to_bgra32_kernel(grid, block,
                            gpu_frame_bgr24, stride,
                            (uint8_t *)enc->d_bgra, enc->input_buffer_pitch,
                            enc->width, enc->height, enc->stream);

    /* ── 2. Copy BGRA32 frame to NVENC input buffer ───────────────── */
    /* NVENC input buffers are registered CUDA memory. We use
     * nvEncLockInputBuffer / nvEncUnlockInputBuffer to get a CPU pointer,
     * then copy from GPU to that pinned buffer.
     *
     * Actually, the newer NVENC API supports direct CUDAarray input.
     * For portability, use the lock/copy/unlock approach. */

    void *input_surface = NULL;
    uint32_t input_pitch = (uint32_t)enc->input_buffer_pitch;
    NVENCSTATUS nv_status;

    {
        NV_ENC_LOCK_INPUT_BUFFER lock_input;
        memset(&lock_input, 0, sizeof(lock_input));
        lock_input.version     = NV_ENC_LOCK_INPUT_BUFFER_VER;
        lock_input.inputBuffer = enc->input_buffer;
        nv_status = enc->nvenc_api.nvEncLockInputBuffer(enc->encoder, &lock_input);
        if (nv_status != NV_ENC_SUCCESS) return -1;
        input_surface = lock_input.bufferDataPtr;
        input_pitch = lock_input.pitch;
    }

    /* Copy BGRA32 from device to the locked input surface */
    cudaError_t ce = cudaMemcpy2DAsync(input_surface, (size_t)input_pitch,
                      (void *)enc->d_bgra, (size_t)enc->input_buffer_pitch,
                      (size_t)enc->width * 4, (size_t)enc->height,
                      cudaMemcpyDeviceToHost, enc->stream);
    if (ce != cudaSuccess) {
        enc->nvenc_api.nvEncUnlockInputBuffer(enc->encoder, enc->input_buffer);
        return -1;
    }
    ce = cudaStreamSynchronize(enc->stream);
    if (ce != cudaSuccess) {
        /* Kernel or copy failed — surface may be corrupt */
        enc->nvenc_api.nvEncUnlockInputBuffer(enc->encoder, enc->input_buffer);
        return -1;
    }

    /* Unlock input buffer */
    enc->nvenc_api.nvEncUnlockInputBuffer(enc->encoder, enc->input_buffer);

    /* ── 3. Submit frame to NVENC encoder ──────────────────────────── */
    NV_ENC_PIC_PARAMS pic_params;
    memset(&pic_params, 0, sizeof(pic_params));
    pic_params.version       = NV_ENC_PIC_PARAMS_VER;
    pic_params.inputBuffer   = enc->input_buffer;
    pic_params.bufferFmt     = NV_ENC_BUFFER_FORMAT_ARGB;
    pic_params.inputWidth    = enc->width;
    pic_params.inputHeight   = enc->height;
    pic_params.inputPitch    = input_pitch;
    pic_params.outputBitstream = enc->output_buffer;
    pic_params.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;

    nv_status = enc->nvenc_api.nvEncEncodePicture(enc->encoder, &pic_params);
    if (nv_status != NV_ENC_SUCCESS) {
        return -1;
    }

    enc->total_encoded_frames++;

    /* ── 4. Retrieve encoded bitstream ─────────────────────────────── */
    NV_ENC_LOCK_BITSTREAM lock;
    memset(&lock, 0, sizeof(lock));
    lock.version          = NV_ENC_LOCK_BITSTREAM_VER;
    lock.outputBitstream  = enc->output_buffer;
    lock.doNotWait        = 1;

    nv_status = enc->nvenc_api.nvEncLockBitstream(enc->encoder, &lock);
    if (nv_status != NV_ENC_SUCCESS) return 0; /* no packet ready yet — will be collected later */

    /* Copy encoded data to our internal buffer */
    int packet_size = (int)lock.bitstreamSizeInBytes;
    if (packet_size > 0) {
        const uint8_t *data = (const uint8_t *)lock.bitstreamBufferPtr;

        /* For the first call, we return the packet synchronously.
         * For subsequent calls, packets are queued for retrieval. */
        if (!add_packet(enc, data, packet_size)) {
            enc->nvenc_api.nvEncUnlockBitstream(enc->encoder, enc->output_buffer);
            return -1;
        }
    }

    enc->nvenc_api.nvEncUnlockBitstream(enc->encoder, enc->output_buffer);

    /* Return the size of the first queued packet, or 0 if delayed */
    return enc->packet_count > 0 ? enc->packet_sizes[0] : 0;
}

int gpu_nvenc_get_packet(GpuNvencEncoder *enc, const uint8_t **packet_out) {
    if (!enc || !packet_out || enc->packet_count <= 0) {
        if (packet_out) *packet_out = NULL;
        return 0;
    }

    *packet_out = enc->packet_data[0];
    int size = enc->packet_sizes[0];

    /* Shift queue (swap pointers instead of copying data) */
    uint8_t *tmp_data = enc->packet_data[0];
    int tmp_cap = enc->packet_capacities[0];
    for (int i = 1; i < enc->packet_count; ++i) {
        enc->packet_data[i - 1] = enc->packet_data[i];
        enc->packet_sizes[i - 1] = enc->packet_sizes[i];
        enc->packet_capacities[i - 1] = enc->packet_capacities[i];
    }
    enc->packet_data[enc->packet_count - 1] = tmp_data;
    enc->packet_capacities[enc->packet_count - 1] = tmp_cap;
    enc->packet_sizes[enc->packet_count - 1] = 0;
    enc->packet_count--;

    return size;
}

int gpu_nvenc_flush(GpuNvencEncoder *enc) {
    if (!enc || !enc->encoder) return -1;

    /* Send end-of-stream signal to NVENC to flush all buffered frames.
     * Without this, the encoder may hold back several frame periods'
     * worth of encoded data (look-ahead / B-frame reorder buffer). */
    NV_ENC_PIC_PARAMS eos_params;
    memset(&eos_params, 0, sizeof(eos_params));
    eos_params.version         = NV_ENC_PIC_PARAMS_VER;
    eos_params.encodePicFlags  = NV_ENC_PIC_FLAG_EOS;
    eos_params.inputBuffer     = NULL;
    eos_params.outputBitstream = enc->output_buffer;

    NVENCSTATUS nv_status = enc->nvenc_api.nvEncEncodePicture(enc->encoder, &eos_params);
    if (nv_status != NV_ENC_SUCCESS) {
        return -1;
    }

    /* After EOS, retrieve the last pending packet(s) from the bitstream buffer.
     * First call with doNotWait=0 (wait for encoder to finish all pending work).
     * Subsequent calls with doNotWait=1 to drain any remaining packets. */
    {
        NV_ENC_LOCK_BITSTREAM lock;
        memset(&lock, 0, sizeof(lock));
        lock.version         = NV_ENC_LOCK_BITSTREAM_VER;
        lock.outputBitstream = enc->output_buffer;
        lock.doNotWait       = 0;  /* Wait for encoder to finish flushing */

        NVENCSTATUS ns = enc->nvenc_api.nvEncLockBitstream(enc->encoder, &lock);
        if (ns == NV_ENC_SUCCESS) {
            int packet_size = (int)lock.bitstreamSizeInBytes;
            if (packet_size > 0) {
                const uint8_t *data = (const uint8_t *)lock.bitstreamBufferPtr;
                if (!add_packet(enc, data, packet_size)) {
                    enc->nvenc_api.nvEncUnlockBitstream(enc->encoder, enc->output_buffer);
                    return -1;
                }
            }
            enc->nvenc_api.nvEncUnlockBitstream(enc->encoder, enc->output_buffer);
        }
    }

    /* Try to get any additional packets (non-blocking) */
    while (1) {
        NV_ENC_LOCK_BITSTREAM lock;
        memset(&lock, 0, sizeof(lock));
        lock.version         = NV_ENC_LOCK_BITSTREAM_VER;
        lock.outputBitstream = enc->output_buffer;
        lock.doNotWait       = 1;

        NVENCSTATUS ns = enc->nvenc_api.nvEncLockBitstream(enc->encoder, &lock);
        if (ns != NV_ENC_SUCCESS) break;

        int packet_size = (int)lock.bitstreamSizeInBytes;
        if (packet_size <= 0) {
            enc->nvenc_api.nvEncUnlockBitstream(enc->encoder, enc->output_buffer);
            break;
        }

        const uint8_t *data = (const uint8_t *)lock.bitstreamBufferPtr;
        if (!add_packet(enc, data, packet_size)) {
            enc->nvenc_api.nvEncUnlockBitstream(enc->encoder, enc->output_buffer);
            return -1;
        }
        enc->nvenc_api.nvEncUnlockBitstream(enc->encoder, enc->output_buffer);
    }

    return 0;
}

int64_t gpu_nvenc_encoded_bytes(GpuNvencEncoder *enc) {
    return enc ? enc->total_encoded_bytes : 0;
}

int gpu_nvenc_encoded_frames(GpuNvencEncoder *enc) {
    return enc ? enc->total_encoded_frames : 0;
}

#else /* !USE_CUDA — stubs */

GpuNvencEncoder* gpu_nvenc_create(int width, int height, double fps,
                                   int cuda_device, int bitrate,
                                   int gop_length,
                                   char *error_out, int error_size) {
    (void)width; (void)height; (void)fps; (void)cuda_device;
    (void)bitrate; (void)gop_length;
    if (error_out) snprintf(error_out, (size_t)error_size, "CUDA not compiled");
    return NULL;
}

void gpu_nvenc_destroy(GpuNvencEncoder *enc) { (void)enc; }

int gpu_nvenc_encode_frame(GpuNvencEncoder *enc,
                            const uint8_t *frame, int stride) {
    (void)enc; (void)frame; (void)stride; return -1;
}

int gpu_nvenc_get_packet(GpuNvencEncoder *enc, const uint8_t **p) {
    (void)enc; if (p) *p = NULL; return 0;
}

int gpu_nvenc_flush(GpuNvencEncoder *enc) { (void)enc; return -1; }

int64_t gpu_nvenc_encoded_bytes(GpuNvencEncoder *enc) { (void)enc; return 0; }
int gpu_nvenc_encoded_frames(GpuNvencEncoder *enc) { (void)enc; return 0; }

#endif /* USE_CUDA */

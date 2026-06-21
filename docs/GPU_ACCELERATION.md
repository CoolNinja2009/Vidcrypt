# Vidcrypt GPU Acceleration — Production Architecture

<!--
Copyright (C) 2024-2026 Vidcrypt Contributors

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
-->

> **Author:** Senior CUDA/FFmpeg/HPC Systems Engineer  
> **Version:** 3.0 (Multi-GPU Production Design)  
> **Target Stack:** CUDA 12.x+, CMake 3.18+, C17, C++17 (for CUDA kernels)

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Supported GPU Matrix](#2-supported-gpu-matrix)
3. [Auto-Detect & Preset System](#3-auto-detect--preset-system)
4. [CUDA Kernel Design](#4-cuda-kernel-design)
5. [Memory Flow Design](#5-memory-flow-design)
6. [Queue & Thread Model](#6-queue--thread-model)
7. [NVDEC/NVENC Integration](#7-nvdecnvenc-integration)
8. [Build System: Multi-Arch Fat Binary](#8-build-system-multi-arch-fat-binary)
9. [Profiling Plan](#9-profiling-plan)
10. [Implementation Roadmap](#10-implementation-roadmap)
11. [CPU-Only Mode Guarantee](#11-cpu-only-mode-guarantee)
12. [Testing & Validation](#12-testing--validation)

---

## 1. Architecture Overview

VidCrypt implements a **heterogeneous CPU+GPU pipeline** where massively parallel work (frame thresholding, block extraction, frame generation) runs on CUDA, while control-flow-heavy work (Reed-Solomon, file I/O, SHA-256) stays on CPU.

### Design Principles

| Principle | Rationale |
|-----------|-----------|
| **CPU-only always works** | `USE_CUDA=OFF` builds with zero CUDA dependencies. GPU is an acceleration layer. |
| **PCIe minimization** | Keep full frames in VRAM. Transfer only compact bitstreams (~3KB/frame, not ~3MB). |
| **Runtime GPU detection** | `--backend auto` probes GPU at startup, selects optimal preset, falls back to CPU. |
| **Architecture presets** | Each GPU generation gets tuned block sizes, shared memory, and kernel variants. |
| **Fat binary + PTX** | Compile SASS for major architectures + PTX for forward compatibility. |

### Pipeline Overview

```
DECODE (HYBRID_CPU_GPU):
  Video file → NVDEC (hardware) → GPU frame surfaces
                                  → CUDA: nv12_to_gray
                                  → CUDA: block_extract (per-block threshold)
                                  → CUDA: sync_validate (warp-level)
                                  → PCIe: bits (~3KB)
                                  → CPU: Reed-Solomon decode (thread pool)
                                  → CPU: file reconstruction → disk

ENCODE (HYBRID_CPU_GPU):
  File → CPU: Reed-Solomon encode (thread pool)
       → PCIe: bits (~3KB)
       → CUDA: frame_generate (expand bits → block grid)
       → CUDA: render_template (calibration + sync row, one-time)
       → NVENC (hardware) or CUDA → pinned buffer → FFmpeg pipe
       → Video file
```

---

## 2. Supported GPU Matrix

### Compute Capability Coverage

| Architecture | sm_XX | Example GPUs | Tensor Cores | Min CUDA | Status |
|-------------|-------|-------------|:------------:|:--------:|:------:|
| **Maxwell** | sm_50, sm_52 | GTX 750 Ti, GTX 960, GTX 980 | ❌ | 6.5+ | Deprecated — PTX-only JIT |
| **Pascal** | sm_60, sm_61, sm_62 | GTX 1060, GTX 1080, Titan Xp, P100 | ❌ | 8.0+ | PTX-only JIT (CUDA 12.x) |
| **Volta** | sm_70, sm_72 | V100, Titan V | ✅ Gen-1 | 9.0+ | SASS included |
| **Turing** | sm_75 | RTX 2060, RTX 2080, T4 | ✅ Gen-2 | 10.0+ | SASS included |
| **Ampere** | sm_80, sm_86, sm_87 | A100, RTX 3060, RTX 3080, RTX 3090 | ✅ Gen-3 | 11.0+ | SASS included |
| **Ada Lovelace** | sm_89 | RTX 4060, RTX 4080, RTX 4090, L40, L40S | ✅ Gen-4 | 11.8+ | SASS included |
| **Hopper** | sm_90, sm_90a | H100, H200, GH200 | ✅ Gen-5 | 12.0+ | SASS included |
| **Blackwell DC** | sm_100, sm_103 | B100, B200, B300 | ✅ Gen-6 | 12.8+ | SASS included |
| **Blackwell Consumer** | **sm_120** | **RTX 5060, RTX 5070, RTX 5080, RTX 5090** | ✅ Gen-6 | 12.8+ | SASS included |
| **Blackwell Embedded** | sm_110 | Thor | ✅ Gen-6 | 12.8+ | PTX JIT |

> **RTX 5070** has compute capability **sm_120** (Blackwell consumer family).

**Key architectural differences that affect kernel design:**

| Feature | sm_50-62 (Maxwell/Pascal) | sm_70-75 (Volta/Turing) | sm_80-89 (Ampere/Ada) | sm_90+ (Hopper/Blackwell) |
|---------|:-------------------------:|:-----------------------:|:---------------------:|:-------------------------:|
| Max threads / block | 1024 | 1024 | 1024 | 1024 |
| Warp size | 32 | 32 | 32 | 32 |
| Shared mem / block | 48 KB | 48-64 KB | 48-164 KB | 48-228 KB |
| Shared mem / SM | 96 KB | 96-128 KB | 128-256 KB | 256 KB |
| Registers / SM | 64K | 64K | 65K | 65K-128K |
| Async copy | ❌ | ❌ | ✅ (sm_80+) | ✅ |
| Tensor cores | ❌ | ✅ Gen-1 | ✅ Gen-3 | ✅ Gen-5/6 |
| L2 cache | 512K-2M | 4-6M | 6-40M | 50-80M |
| DPX (dynamic prog) | ❌ | ❌ | ❌ | ✅ (sm_90+) |

---

## 3. Auto-Detect & Preset System

### GPU Detection Flow

```
Application start
       │
       ▼
┌──────────────────┐
│ --backend flag   │
│  ├─ cpu    → USE CPU ONLY
│  ├─ gpu    → REQUIRE GPU; fail if unavailable
│  └─ auto   → Probe GPU; fall back to CPU
└──────────────────┘
       │ (if backend != CPU)
       ▼
┌─────────────────────────────────────┐
│ cudaGetDeviceCount(&count)          │
│ if (count == 0) → fall back to CPU │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ cudaSetDevice(0)                    │
│ cudaGetDeviceProperties(&prop, 0)  │
│  → major, minor (e.g., 12, 0)      │
│  → multiProcessorCount              │
│  → sharedMemPerBlock                │
│  → name (e.g., "NVIDIA GeForce      │
│            RTX 5070")               │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ gpu_preset_lookup(major, minor)    │
│  → returns GpuPreset*              │
│  → adjusts for specific device     │
│    characteristics                  │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ cudaOccupancyMaxActiveBlocksPerMult │
│ ocessor(...) → verify/refine preset│
│ Apply preset to all kernel launches│
└─────────────────────────────────────┘
```

### Preset Definition

```c
// gpu_presets.h — GPU architecture presets

typedef struct {
    int   compute_capability_major;    // e.g., 12 for Blackwell
    int   compute_capability_minor;    // e.g., 0
    const char *architecture_name;     // e.g., "Blackwell (RTX 50)"

    // Decode kernel parameters
    int   decode_block_size_x;         // threads per block (x-dim) for block_extract
    int   decode_block_size_y;         // threads per block (y-dim) for block_extract
    int   decode_shared_mem_bytes;     // dynamic shared memory per block
    int   decode_subsample;            // block subsampling factor (2 typical)

    // Encode kernel parameters
    int   encode_block_size_x;         // threads per block (x-dim) for frame_generate
    int   encode_block_size_y;         // threads per block (y-dim) for frame_generate

    // Pipeline tuning
    int   num_cuda_streams;            // CUDA streams for pipelining
    int   num_nvdec_surfaces;          // NVDEC surface count (4-16)
    int   batch_frames_per_kernel;     // frames per batch for encoder

    // Feature flags
    bool  has_tensor_cores;
    bool  has_async_copy;              // cudaMemcpyAsync with SM engine
    bool  has_dp_instructions;         // DPX support (sm_90+)
    bool  prefer_nvenc;                // use NVENC over FFmpeg pipe

    // NVENC specific
    int   nvenc_max_width;
    int   nvenc_max_height;
    int   nvenc_supported_codecs;      // bitmask

    // NVDEC specific
    int   nvdec_max_width;
    int   nvdec_max_height;
    int   nvdec_max_surfaces;
} GpuPreset;

// ─── Built-in presets for every major architecture ───────────────────

const GpuPreset PRESET_MAXWELL = {
    .compute_capability_major = 5,
    .compute_capability_minor = 0,
    .architecture_name = "Maxwell",

    .decode_block_size_x    = 32,
    .decode_block_size_y    = 4,     // 128 threads/block — small shared mem
    .decode_shared_mem_bytes = 8192, // 8 KB (limited to 48 KB total on Maxwell)
    .decode_subsample       = 2,

    .encode_block_size_x    = 16,
    .encode_block_size_y    = 16,    // 256 threads/block

    .num_cuda_streams       = 2,
    .num_nvdec_surfaces     = 4,
    .batch_frames_per_kernel = 4,

    .has_tensor_cores       = false,
    .has_async_copy         = false,
    .has_dp_instructions    = false,
    .prefer_nvenc           = false, // FFmpeg pipe fallback — NVENC old gen
    .nvenc_max_width        = 4096,
    .nvenc_max_height       = 4096,
    .nvenc_supported_codecs = 0,
    .nvdec_max_width        = 4096,
    .nvdec_max_height       = 4096,
    .nvdec_max_surfaces     = 4,
};

const GpuPreset PRESET_PASCAL = {
    .compute_capability_major = 6,
    .compute_capability_minor = 0,
    .architecture_name = "Pascal",

    .decode_block_size_x    = 32,
    .decode_block_size_y    = 4,
    .decode_shared_mem_bytes = 12288, // 12 KB (48 KB total)
    .decode_subsample       = 2,

    .encode_block_size_x    = 16,
    .encode_block_size_y    = 16,

    .num_cuda_streams       = 2,
    .num_nvdec_surfaces     = 6,
    .batch_frames_per_kernel = 8,

    .has_tensor_cores       = false,
    .has_async_copy         = false,
    .has_dp_instructions    = false,
    .prefer_nvenc           = false, // NVENC gen-6 (good but FFmpeg is fine)
    .nvenc_max_width        = 4096,
    .nvenc_max_height       = 4096,
    .nvenc_supported_codecs = 0,
    .nvdec_max_width        = 4096,
    .nvdec_max_height       = 4096,
    .nvdec_max_surfaces     = 6,
};

const GpuPreset PRESET_VOLTA = {
    .compute_capability_major = 7,
    .compute_capability_minor = 0,
    .architecture_name = "Volta",

    .decode_block_size_x    = 32,
    .decode_block_size_y    = 8,     // 256 threads/block — more threads for tensor
    .decode_shared_mem_bytes = 16384,// 16 KB (48 KB total, but can opt-in to 96 KB)
    .decode_subsample       = 2,

    .encode_block_size_x    = 16,
    .encode_block_size_y    = 16,

    .num_cuda_streams       = 3,
    .num_nvdec_surfaces     = 8,
    .batch_frames_per_kernel = 12,

    .has_tensor_cores       = true,
    .has_async_copy         = false,
    .has_dp_instructions    = false,
    .prefer_nvenc           = true,  // NVENC gen-7 (efficient)
    .nvenc_max_width        = 4096,
    .nvenc_max_height       = 4096,
    .nvenc_supported_codecs = 0,
    .nvdec_max_width        = 4096,
    .nvdec_max_height       = 4096,
    .nvdec_max_surfaces     = 8,
};

const GpuPreset PRESET_TURING = {
    .compute_capability_major = 7,
    .compute_capability_minor = 5,
    .architecture_name = "Turing",

    .decode_block_size_x    = 32,
    .decode_block_size_y    = 8,
    .decode_shared_mem_bytes = 16384,
    .decode_subsample       = 2,

    .encode_block_size_x    = 16,
    .encode_block_size_y    = 16,

    .num_cuda_streams       = 4,
    .num_nvdec_surfaces     = 8,
    .batch_frames_per_kernel = 16,

    .has_tensor_cores       = true,
    .has_async_copy         = false,
    .has_dp_instructions    = false,
    .prefer_nvenc           = true,
    .nvenc_max_width        = 7680,
    .nvenc_max_height       = 4320,
    .nvenc_supported_codecs = 0,
    .nvdec_max_width        = 7680,
    .nvdec_max_height       = 4320,
    .nvdec_max_surfaces     = 8,
};

const GpuPreset PRESET_AMPERE = {
    .compute_capability_major = 8,
    .compute_capability_minor = 0,
    .architecture_name = "Ampere",

    .decode_block_size_x    = 32,
    .decode_block_size_y    = 16,    // 512 threads/block — high occupancy
    .decode_shared_mem_bytes = 24576,// 24 KB (can go up to 164 KB opt-in)
    .decode_subsample       = 2,

    .encode_block_size_x    = 16,
    .encode_block_size_y    = 16,

    .num_cuda_streams       = 4,
    .num_nvdec_surfaces     = 10,
    .batch_frames_per_kernel = 24,

    .has_tensor_cores       = true,
    .has_async_copy         = true,  // sm_80+ has async copy engine
    .has_dp_instructions    = false,
    .prefer_nvenc           = true,
    .nvenc_max_width        = 7680,
    .nvenc_max_height       = 4320,
    .nvenc_supported_codecs = 0,
    .nvdec_max_width        = 7680,
    .nvdec_max_height       = 4320,
    .nvdec_max_surfaces     = 10,

    .nvenc_codec_h264       = true,
    .nvenc_codec_hevc       = true,
    .nvenc_codec_av1        = true,
};

const GpuPreset PRESET_ADA = {
    .compute_capability_major = 8,
    .compute_capability_minor = 9,
    .architecture_name = "Ada Lovelace",

    .decode_block_size_x    = 32,
    .decode_block_size_y    = 16,
    .decode_shared_mem_bytes = 32768,// 32 KB (can opt-in to much more)
    .decode_subsample       = 2,

    .encode_block_size_x    = 16,
    .encode_block_size_y    = 16,

    .num_cuda_streams       = 6,     // more streams for better overlap
    .num_nvdec_surfaces     = 12,
    .batch_frames_per_kernel = 32,

    .has_tensor_cores       = true,
    .has_async_copy         = true,
    .has_dp_instructions    = false,
    .prefer_nvenc           = true,
    .nvenc_max_width        = 7680,
    .nvenc_max_height       = 4320,
    .nvenc_supported_codecs = 0,
    .nvdec_max_width        = 7680,
    .nvdec_max_height       = 4320,
    .nvdec_max_surfaces     = 12,
};

const GpuPreset PRESET_HOPPER = {
    .compute_capability_major = 9,
    .compute_capability_minor = 0,
    .architecture_name = "Hopper",

    .decode_block_size_x    = 32,
    .decode_block_size_y    = 32,    // 1024 threads/block — maximum occupancy
    .decode_shared_mem_bytes = 49152,// 48 KB (shared mem up to 228 KB)
    .decode_subsample       = 2,

    .encode_block_size_x    = 32,    // larger blocks for Hopper
    .encode_block_size_y    = 32,

    .num_cuda_streams       = 8,     // many streams for Hopper's MIG
    .num_nvdec_surfaces     = 16,
    .batch_frames_per_kernel = 48,

    .has_tensor_cores       = true,
    .has_async_copy         = true,
    .has_dp_instructions    = true,  // DPX (dynamic programming) instructions
    .prefer_nvenc           = true,
    .nvenc_max_width        = 8192,
    .nvenc_max_height       = 8192,
    .nvenc_supported_codecs = 0,
    .nvdec_max_width        = 8192,
    .nvdec_max_height       = 8192,
    .nvdec_max_surfaces     = 16,
};

const GpuPreset PRESET_BLACKWELL_DC = {
    .compute_capability_major = 10,
    .compute_capability_minor = 0,
    .architecture_name = "Blackwell (DC)",

    .decode_block_size_x    = 32,
    .decode_block_size_y    = 32,
    .decode_shared_mem_bytes = 65536,// 64 KB — Blackwell has abundant shared mem
    .decode_subsample       = 2,

    .encode_block_size_x    = 32,
    .encode_block_size_y    = 32,

    .num_cuda_streams       = 8,
    .num_nvdec_surfaces     = 16,
    .batch_frames_per_kernel = 64,

    .has_tensor_cores       = true,
    .has_async_copy         = true,
    .has_dp_instructions    = true,
    .prefer_nvenc           = true,
    .nvenc_max_width        = 8192,
    .nvenc_max_height       = 8192,
    .nvdec_max_width        = 8192,
    .nvdec_max_height       = 8192,
    .nvdec_max_surfaces     = 16,
};

const GpuPreset PRESET_BLACKWELL_CONSUMER = {
    .compute_capability_major = 12,
    .compute_capability_minor = 0,
    .architecture_name = "Blackwell (RTX 50)",

    .decode_block_size_x    = 32,
    .decode_block_size_y    = 32,    // 1024 threads/block for RTX 5070+
    .decode_shared_mem_bytes = 49152,// 48 KB (can scale dynamically)
    .decode_subsample       = 2,     // can try subsample=1 for quality at perf cost

    .encode_block_size_x    = 32,
    .encode_block_size_y    = 32,    // 1024 threads/block

    .num_cuda_streams       = 8,     // max overlap
    .num_nvdec_surfaces     = 16,
    .batch_frames_per_kernel = 64,

    .has_tensor_cores       = true,
    .has_async_copy         = true,
    .has_dp_instructions    = true,
    .prefer_nvenc           = true,  // RTX 5070 has Blackwell NVENC (AV1, H.265, H.264)
    .nvenc_max_width        = 8192,
    .nvenc_max_height       = 8192,
    .nvdec_max_width        = 8192,
    .nvdec_max_height       = 8192,
    .nvdec_max_surfaces     = 16,

    .nvenc_codec_av1        = true,  // RTX 50-series has AV1 encoding
    .nvenc_codec_hevc       = true,
    .nvenc_codec_h264       = true,
};

// ─── Preset lookup ───────────────────────────────────────────────────

typedef struct {
    int             count;
    const GpuPreset *presets[20];  // ordered by preference
    GpuPreset       runtime_adjusted; // copy of final adjusted preset
} ActivePresets;

// Lookup the best preset for a given compute capability
const GpuPreset* gpu_preset_lookup(int major, int minor);

// Refine preset based on specific device properties (occupancy tuning)
GpuPreset gpu_preset_refine(const GpuPreset *base, const cudaDeviceProp *prop);

// Get human-readable architecture name
const char* gpu_arch_name(int major, int minor);

// Get description string for preset
void gpu_preset_describe(const GpuPreset *preset, char *buf, int buf_size);
```

### Preset Lookup Logic

```c
const GpuPreset* gpu_preset_lookup(int major, int minor) {
    int cc = major * 100 + minor;

    // Ordered from newest to oldest — first match wins
    if (cc >= 1200) return &PRESET_BLACKWELL_CONSUMER; // sm_12x (RTX 50)
    if (cc >= 1000) return &PRESET_BLACKWELL_DC;       // sm_10x (B100/B200)
    if (cc >= 900)  return &PRESET_HOPPER;             // sm_9x (H100)
    if (cc >= 890)  return &PRESET_ADA;                // sm_89 (RTX 40)
    if (cc >= 800)  return &PRESET_AMPERE;             // sm_8x (RTX 30/A100)
    if (cc >= 750)  return &PRESET_TURING;             // sm_75 (RTX 20)
    if (cc >= 700)  return &PRESET_VOLTA;              // sm_7x (V100)
    if (cc >= 600)  return &PRESET_PASCAL;             // sm_6x (GTX 10)
    if (cc >= 500)  return &PRESET_MAXWELL;            // sm_5x (GTX 9)
    return NULL; // unsupported — fall back to CPU
}
```

### Runtime Occupancy Tuning

After selecting a base preset, the GPU backend refines it at startup using `cudaOccupancyMaxActiveBlocksPerMultiprocessor`:

```c
GpuPreset gpu_preset_refine(const GpuPreset *base, const cudaDeviceProp *prop) {
    GpuPreset refined = *base;

    // Adjust shared memory if target GPU has more than the preset assumes
    if (prop->sharedMemPerBlock > refined.decode_shared_mem_bytes) {
        refined.decode_shared_mem_bytes = (int)prop->sharedMemPerBlock / 4;
        if (refined.decode_shared_mem_bytes > 65536)
            refined.decode_shared_mem_bytes = 65536;
    }

    // Verify decode block size yields good occupancy
    int min_grid_size;
    int block_size = refined.decode_block_size_x * refined.decode_block_size_y;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                       block_extract_kernel,
                                       (size_t)refined.decode_shared_mem_bytes,
                                       (int)prop->multiProcessorCount);

    // Adjust batch sizes based on SM count
    int sm_count = (int)prop->multiProcessorCount;
    refined.batch_frames_per_kernel =
        (base->batch_frames_per_kernel * sm_count) / 80; // normalize to base (80 SM ≈ RTX 4090)
    if (refined.batch_frames_per_kernel < 4)
        refined.batch_frames_per_kernel = 4;
    if (refined.batch_frames_per_kernel > 128)
        refined.batch_frames_per_kernel = 128;

    return refined;
}
```

---

## 4. CUDA Kernel Design

### A. `nv12_to_gray` — Frame Conversion

```cuda
// Input:  NV12 frame in GPU memory (from NVDEC decode output)
// Output: Planar grayscale uint8_t buffer
//
// NV12 layout: Y plane (width × height) followed by interleaved UV (width/2 × height/2)
// For grayscale, we only need the Y plane — this kernel is essentially a
// pitched-to-linear copy with potential stride fixup.

__global__ void nv12_to_gray_kernel(
    const uint8_t* __restrict__ nv12_y_plane,  // NV12 Y plane (device ptr)
    int pitch_y,                                // pitch of Y plane in bytes
    uint8_t* __restrict__ gray_out,             // planar gray output (width × height)
    int width,
    int height,
    int gray_pitch)                             // pitch of gray output
{
    // 2D block: 32×8 = 256 threads
    // Grid covers the entire frame in tiles
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Coalesced read from NV12 Y plane, coalesced write to gray
    // Using __ldg() for read-only cache on Kepler+
    gray_out[y * gray_pitch + x] = __ldg(&nv12_y_plane[y * pitch_y + x]);
}

// ─── Launch configuration ───────────────────────────────────────────
// For a preset with 32×16 threads/block:
dim3 blockDim(32, 16);
dim3 gridDim(
    (width  + blockDim.x - 1) / blockDim.x,
    (height + blockDim.y - 1) / blockDim.y
);
```

### B. `block_extract` — Payload Block Bit Recovery

This is the **most performance-critical kernel** — it replaces the CPU's `scalar_count_white` loop which accounts for ~40-50% of decode runtime.

```cuda
// Key optimization: Use shared memory to cache the tile region loaded
// by a block, then each thread reads its subsampled portion from shared mem.
// This turns scattered global reads into coalesced shared memory loads.

// Supported launch configurations (per preset):
//   Maxwell/Pascal:  32×4  = 128 threads, 8 KB shared mem
//   Volta/Turing:    32×8  = 256 threads, 16 KB shared mem
//   Ampere/Ada:      32×16 = 512 threads, 24 KB shared mem
//   Hopper/BWell:    32×32 = 1024 threads, 48+ KB shared mem

__global__ void block_extract_kernel_tiled(
    const uint8_t* __restrict__ gray,
    int frame_width,
    int frame_height,
    int grid_top_y,
    int grid_left_x,
    int block_size,
    int grid_cols,
    int grid_rows,
    int sync_rows,
    int subsample,
    uint8_t* __restrict__ bits_out)
{
    // One thread per payload block
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_blocks = (grid_rows - sync_rows) * grid_cols;
    if (block_idx >= total_blocks) return;

    int row = block_idx / grid_cols + sync_rows;
    int col = block_idx % grid_cols;

    int y0 = grid_top_y + row * block_size;
    int x0 = grid_left_x + col * block_size;

    int white_count = 0;
    int sample_count = 0;

    // Subsampled threshold read — using vectorized uchar4 where possible
    if (subsample == 2) {
        // Fast path: stride-2 subsampling with __ldg
        for (int dy = 0; dy < block_size; dy += 2) {
            for (int dx = 0; dx < block_size; dx += 2) {
                white_count += (__ldg(&gray[(y0 + dy) * frame_width + (x0 + dx)]) >= 128);
                sample_count++;
            }
        }
    } else {
        // Generic path
        for (int dy = 0; dy < block_size; dy += subsample) {
            for (int dx = 0; dx < block_size; dx += subsample) {
                white_count += (__ldg(&gray[(y0 + dy) * frame_width + (x0 + dx)]) >= 128);
                sample_count++;
            }
        }
    }

    bits_out[block_idx] = (white_count > sample_count / 2) ? 1 : 0;
}
```

> **Why not use shared memory tiling?** Each block extract accesses non-contiguous
> memory (strided by block_size), so a full shared memory tile would require loading
> an entire block region that no thread reads entirely. With subsample=2 and
> __ldg cache, global reads from neighboring blocks hit L2 — the simpler
> one-thread-per-block approach actually wins on modern GPUs with large L2 caches.

### C. `sync_validate` — Sync Row Detection

```cuda
// A single warp (32 threads) validates the entire sync row.
// Each lane checks one grid column. Uses warp-level ballot + popcount.

__global__ void sync_validate_kernel(
    const uint8_t* __restrict__ gray,
    int frame_width,
    int sync_y,
    int sync_x_start,
    int block_size,
    int grid_cols,
    int subsample,
    float* confidence_out,
    bool* pass_out)
{
    int col = threadIdx.x;
    if (col >= grid_cols) return;

    int expected = col % 2;  // alternating pattern: 0,1,0,1...

    int x0 = sync_x_start + col * block_size;

    int white_count = 0;
    int sample_count = 0;
    for (int dy = 0; dy < block_size; dy += subsample)
        for (int dx = 0; dx < block_size; dx += subsample) {
            white_count += (__ldg(&gray[(sync_y + dy) * frame_width + (x0 + dx)]) >= 128);
            sample_count++;
        }

    int bit = (white_count > sample_count / 2);
    int match = (bit == expected) ? 1 : 0;

    // Warp-level aggregation (single-instruction on sm_30+)
    int warp_match_count = __popc(__ballot_sync(0xFFFFFFFF, match));

    if (threadIdx.x == 0) {
        *pass_out = (warp_match_count == grid_cols);
        *confidence_out = (float)warp_match_count / (float)grid_cols;
    }
}
```

### D. `frame_generate` — Encoder Frame Generation

```cuda
// 2D grid: one thread per pixel in the payload region.
// Output frame buffer pre-filled with calibration + sync template.

// Launch dims: (grid_cols × pay_rows) blocks, (block_size × block_size) threads
// Each block fills one payload tile.

__global__ void frame_generate_kernel(
    uint8_t* __restrict__ frame,             // output BGR frame (pre-filled template)
    int frame_stride,                         // bytes per row (= width * 3)
    const uint8_t* __restrict__ bits,         // packed bits, one byte per block
    int pay_y_start,
    int pay_x_start,
    int block_size,
    int grid_cols,
    int pay_rows)
{
    int col = blockIdx.x;
    int row = blockIdx.y;

    if (row >= pay_rows || col >= grid_cols) return;

    int bit_idx = row * grid_cols + col;
    uint8_t val = bits[bit_idx] ? 255 : 0;

    int x0 = pay_x_start + col * block_size;
    int y0 = pay_y_start + row * block_size;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    if (tx >= block_size || ty >= block_size) return;

    int px = x0 + tx;
    int py = y0 + ty;

    int off = py * frame_stride + px * 3;
    frame[off]     = val;  // B
    frame[off + 1] = val;  // G
    frame[off + 2] = val;  // R
}
```

### E. Calibration Detection (First Frame Only)

```cuda
// Runs once on the first frame — not performance-critical.
// 2D grid: CAL_ROWS × CAL_COLS threads (4 × 48 = 192 threads, trivial)

__global__ void calibration_detect_kernel(
    const uint8_t* __restrict__ gray,
    int frame_width,
    int frame_height,
    uint8_t* __restrict__ cal_bits_out)
{
    int row = blockIdx.y;  // CAL_ROWS (4)
    int col = blockIdx.x;  // CAL_COLS (48)
    if (row >= CAL_ROWS || col >= CAL_COLS) return;

    int cal_top    = (int)(frame_height * CAL_TOP_FRAC);
    int cal_height = (int)(frame_height * CAL_HEIGHT_FRAC);
    int cal_left   = (int)(frame_width  * CAL_LEFT_FRAC);
    int cal_width  = (int)(frame_width  * CAL_WIDTH_FRAC);

    int cell_w = cal_width  / CAL_COLS;
    int cell_h = cal_height / CAL_ROWS;
    int sample_w = (cell_w * 60) / 100; if (sample_w < 2) sample_w = 2;
    int sample_h = (cell_h * 60) / 100; if (sample_h < 2) sample_h = 2;

    int cx = cal_left + col * cell_w + cell_w / 2;
    int cy = cal_top  + row * cell_h + cell_h / 2;
    int sx = cx - sample_w / 2;
    int sy = cy - sample_h / 2;

    int sum = 0;
    int samples = 0;
    for (int yy = sy; yy < sy + sample_h && yy < frame_height; ++yy)
        for (int xx = sx; xx < sx + sample_w && xx < frame_width; ++xx) {
            sum += __ldg(&gray[yy * frame_width + xx]);
            samples++;
        }

    cal_bits_out[row * CAL_COLS + col] = (sum / samples >= THRESHOLD) ? 1 : 0;
}
```

---

## 5. Memory Flow Design

### Strategy: Minimize PCIe Transfers

```
Bad (naive GPU approach):              Good (our approach):
  GPU: NV12 frame (3.1 MB)              GPU: NV12 frame (3.1 MB)
  → cudaMemcpy to CPU (3.1 MB)          → GPU: nv12_to_gray (remains in VRAM)
  → CPU: BGR→grayscale (2.0 MB)         → GPU: block_extract (remains in VRAM)
  → CPU: threshold (2.0 MB)             → GPU: sync_validate (remains in VRAM)
  → CPU: bits (~3 KB)                   → cudaMemcpyAsync bits (~3 KB)
  ─────────────────────────               ─────────────────────────
  3.1 MB × N frames per PCIe            ~3 KB × N frames per PCIe
  = ~10 GB/s bottleneck                  = ~10 MB/s — 1000× reduction
```

### Memory Pool

```c
// gpu_backend.h

typedef struct {
    // ── NVDEC surfaces (ring buffer, allocated by NVDEC decoder) ──
    CUdeviceptr  *nv12_surfaces;     // array of CUdeviceptr
    int           nv12_pitch;        // pitch of each NV12 surface
    int           num_surfaces;      // count in ring buffer
    int           current_surface;   // current write index

    // ── Working buffers (cudaMalloc) ──────────────────────────────
    uint8_t      *d_gray;            // planar grayscale: width × height
    int           gray_pitch;        // pitch of gray buffer
    uint8_t      *d_bits;            // extracted bits: pay_rows * grid_cols
    float        *d_confidence;      // sync confidence (single float)
    uint8_t      *d_cal_bits;        // calibration bits: 192 bytes

    // ── Encoder buffers ───────────────────────────────────────────
    uint8_t      *d_template;        // pre-rendered template (cal + sync)
    uint8_t      *d_payload_bits;    // encoder bits for current frame
    uint8_t      *d_output_frame;    // output BGR frame (width × height × 3)

    // ── Pinned host buffers (cudaHostAlloc) ───────────────────────
    uint8_t      *h_bits;            // pinned: bits stage for CPU
    uint8_t      *h_frame_payload;   // pinned: encoder output for FFmpeg pipe
    uint8_t      *h_cal_bits;        // pinned: calibration bits

    // ── CUDA streams ──────────────────────────────────────────────
    cudaStream_t  stream_decode;     // decode kernel stream
    cudaStream_t  stream_transfer;   // PCIe transfer stream
    cudaStream_t  stream_encode;     // encode kernel stream

    // ── CUDA events (for sync + timing) ───────────────────────────
    cudaEvent_t   timer_start;
    cudaEvent_t   timer_stop;
    cudaEvent_t   stream_sync_event;

    // ── Active preset ─────────────────────────────────────────────
    GpuPreset     active_preset;

    // ── Device properties ─────────────────────────────────────────
    cudaDeviceProp device_prop;
} GpuBackend;

// ─── Lifecycle ──────────────────────────────────────────────────────
bool gpu_backend_init(GpuBackend **backend, int device_id);
void gpu_backend_destroy(GpuBackend *backend);
```

### Memory Budget (1080p)

| Buffer | Size per instance |
|--------|-------------------|
| NV12 surfaces (×16 for RTX 5070) | 16 × 3.1 MB = 49.6 MB |
| Grayscale working buffer | 1920 × 1080 = 2.0 MB |
| Bits output | ~3 KB |
| Template frame | 1920 × 1080 × 3 = 5.9 MB |
| Output frame | 5.9 MB |
| Pinned host bits | 64 KB |
| Pinned host frame payload | 5.9 MB |
| **Total** | **~69 MB** — fits easily in any GPU's VRAM (≥ 2 GB) |

---

## 6. Queue & Thread Model

### Thread Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         MAIN THREAD                                 │
│  • CLI parsing                                                      │
│  • Backend selection (CPU/GPU/AUTO)                                 │
│  • Pipeline orchestration                                           │
│  • Profiling report generation                                      │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────────┐ ┌──────────┐ ┌──────────────┐
│ DECODE THREAD│ │ENCODE TH│ │ RS POOL      │
│ (CUDA stream)│ │(CUDA str)│ │ (N CPU thrds)│
│              │ │          │ │              │
│ 1. NVDEC dec │ │1. PCIe   │ │ SPSC ← bits  │
│ 2. GPU kernls│ │  bits in │ │ SPSC → decod │
│ 3. PCIe bits │ │2. Frame  │ │              │
│    → SPSC Q  │ │  gen k.rn│ │              │
│              │ │3. NVENC  │ │              │
│              │ │  /write  │ │              │
└──────────────┘ └──────────┘ └──────────────┘
```

### Lock-Free Queue Topology

```
DECODER:
  CUDA Stream → SPSC Queue[bits + sync_ok] → RS Decode Pool
                                               → Ordered Queue (frame reorder)
                                               → File Writer (single thread)

ENCODER:
  File Reader (single thread) → SPSC Queue[raw data]
                                  → RS Encode Pool (CPU threads)
                                  → SPSC Queue[encoded bits]
                                  → CUDA Frame Gen Stream
                                  → SPSC Queue[frames]
                                  → NVENC / FFmpeg Writer
```

### CUDA Stream Pipelining

```c
// Double-buffered overlap pattern:
// Stream 0: NVDEC decode frame N+1  →  kernel frame N+1  →  PCIe bits N+1
// Stream 1:   (wait)                →  kernel frame N    →  PCIe bits N
//
// Meanwhile CPU processes bits[N-1] through RS decode

for (int i = 0; i < total_frames; i++) {
    int stream_idx = i % preset.num_cuda_streams;
    cudaStream_t stream = backend->streams[stream_idx];

    // Launch NVDEC decode (async, returns immediately)
    cuvidDecodePicture(decoder, &pic_params);

    // Launch post-decode kernels on the same stream
    nv12_to_gray_kernel<<<grid, block, 0, stream>>>(...);
    block_extract_kernel_tiled<<<grid_b, block_b, shmem, stream>>>(...);
    sync_validate_kernel<<<1, 32, 0, stream>>>(...);

    // Async transfer of bits to pinned host
    cudaMemcpyAsync(backend->h_bits[stream_idx], backend->d_bits,
                    nbits, cudaMemcpyDeviceToHost, stream);

    // Record event for CPU-side sync
    cudaEventRecord(backend->events[stream_idx], stream);

    // CPU-side: check if previous frame's bits are ready
    if (i >= preset.num_cuda_streams) {
        int prev_stream = (i - preset.num_cuda_streams) % preset.num_cuda_streams;
        cudaEventSynchronize(backend->events[prev_stream]);
        // Process bits[prev_stream] through RS decode (on CPU)
        threadpool_submit(rs_pool, &rs_item(backend->h_bits[prev_stream]));
    }
}

// Drain remaining in-flight frames
for (int i = 0; i < preset.num_cuda_streams; i++) {
    cudaEventSynchronize(backend->events[i]);
    threadpool_submit(rs_pool, &rs_item(backend->h_bits[i]));
}
threadpool_wait(rs_pool);
```

---

## 7. NVDEC/NVENC Integration

### NVDEC Decoder

```c
typedef struct {
    CUvideoparser     parser;
    CUvideodecoder    decoder;
    CUdeviceptr      *surfaces;     // decoded frame surfaces (GPU memory)
    int               surface_count;
    int               current_surface;
    int               width, height;
    cudaVideoCodec    codec;
    cudaVideoChromaFormat chroma_format;
    int               bit_depth;
} GpuNvdecDecoder;

GpuNvdecDecoder* gpu_nvdec_create(const char *path, const GpuPreset *preset);
bool gpu_nvdec_decode_frame(GpuNvdecDecoder *nvd,
                             CUdeviceptr *frame_out, int *pitch_out);
void gpu_nvdec_destroy(GpuNvdecDecoder *nvd);

// Fallback: if NVDEC fails or CUDA Video SDK unavailable,
// use existing ffmpeg_pipe.c mechanism and transfer to GPU.
// The CPU → GPU transfer path is much slower but keeps compatibility.
```

### NVENC Encoder

```c
typedef struct {
    NVENCSTRUCT enc;
    CUdeviceptr *output_bitstream;
    int          output_size;
    int          width, height;
    GUID         codec_guid;       // AV1, HEVC, or H.264
} GpuNvencEncoder;

GpuNvencEncoder* gpu_nvenc_create(int width, int height, double fps,
                                   const GpuPreset *preset);
bool gpu_nvenc_encode_frame(GpuNvencEncoder *nve,
                             CUdeviceptr frame, int pitch);
bool gpu_nvenc_end_frame(GpuNvencEncoder *nve, uint8_t **packet, int *packet_size);
void gpu_nvenc_destroy(GpuNvencEncoder *nve);

// Fallback: write frames to pinned host buffer, feed to FFmpeg pipe
// using video_writer_write_frame() from the existing ffmpeg_pipe.c.

// Codec selection per GPU:
//   Blackwell (sm_120): AV1 (best), HEVC (good), H.264 (fallback)
//   Ada/Ampere (sm_80-89): AV1, HEVC, H.264
//   Turing (sm_75): HEVC, H.264 (no AV1 encode)
//   Volta (sm_70): HEVC, H.264
//   Pascal/Maxwell (sm_50-62): H.264 only (or FFmpeg pipe fallback)
```

### NVDEC/NVENC Capability Table

| Architecture | NVDEC | NVENC | AV1 Decode | AV1 Encode | Max Resolution |
|-------------|-------|-------|:----------:|:----------:|:-------------:|
| Maxwell (sm_50) | ✅ Gen-1 | ✅ Gen-6 | ❌ | ❌ | 4096×4096 |
| Pascal (sm_60) | ✅ Gen-2 | ✅ Gen-6 | ❌ | ❌ | 4096×4096 |
| Volta (sm_70) | ✅ Gen-3 | ✅ Gen-7 | ❌ | ❌ | 7680×4320 |
| Turing (sm_75) | ✅ Gen-4 | ✅ Gen-7 | ✅ | ❌ | 7680×4320 |
| Ampere (sm_80) | ✅ Gen-5 | ✅ Gen-7 | ✅ | ❌ | 7680×4320 |
| Ampere (sm_86) | ✅ Gen-5 | ✅ Gen-7 | ✅ | ❌ | 7680×4320 |
| Ada (sm_89) | ✅ Gen-5 | ✅ Gen-8 | ✅ | ✅ | 7680×4320 |
| Hopper (sm_90) | ✅ Gen-5 | ✅ Gen-8 | ✅ | ✅ | 8192×8192 |
| Blackwell (sm_100) | ✅ Gen-6 | ✅ Gen-9 | ✅ | ✅ | 8192×8192 |
| Blackwell (sm_120) | ✅ Gen-6 | ✅ Gen-9 | ✅ | ✅ | 8192×8192 |

---

## 8. Build System: Multi-Arch Fat Binary

### CMake Configuration

```cmake
# CMakeLists.txt additions

# ── CUDA Language Support ─────────────────────────────────────────────
option(USE_CUDA "Enable CUDA-based hybrid backend" OFF)

if(USE_CUDA)
    enable_language(CUDA)

    # ── Target architectures: production fat binary ──────────────────
    # SASS for current-gen GPUs + PTX for future compatibility
    #
    # Strategy:
    #   - Include SASS for the 5 most common current architectures
    #   - Include virtual PTX for sm_50 (the minimum we support via JIT)
    #   - PTX JIT-compiles on older GPUs at first launch
    #
    # This produces ONE .exe that runs on GTX 960 through RTX 5090+
    # without needing separate builds.

    set(VIDCRYPT_CUDA_ARCHS
        # PTX virtual architectures (forward + backward compatibility)
        "50-virtual"    # Maxwell+ via JIT

        # Real architectures (native SASS — no JIT overhead)
        "75"            # Turing (RTX 20, T4)
        "80"            # Ampere DC (A100)
        "86"            # Ampere Consumer (RTX 30)
        "89"            # Ada Lovelace (RTX 40, L40)
        "90"            # Hopper (H100)
        "90-virtual"    # Hopper PTX (for Blackwell DC forward compat)
        "100"           # Blackwell DC (B100/B200)
        "120"           # Blackwell Consumer (RTX 50 — this is the RTX 5070!)
    )

    set(CMAKE_CUDA_ARCHITECTURES "${VIDCRYPT_CUDA_ARCHS}")

    # ── CUDA source files ─────────────────────────────────────────────
    set(CUDA_SOURCES
        src/gpu_backend.c
        src/gpu_kernels.cu
        src/gpu_presets.c
        src/gpu_nvdec.c
        src/gpu_nvenc.c
        src/gpu_profiling.c
    )

    # ── CUDA compile options ──────────────────────────────────────────
    # Use fast-math since our kernels are threshold-based (no precision req)
    target_compile_options(vidcrypt PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>           # line info for profiling
    )

    target_compile_definitions(vidcrypt PRIVATE USE_CUDA=1)
    target_link_libraries(vidcrypt PRIVATE
        cuda
        nvcuvid          # NVDEC (CUDA Video SDK)
        nvencodeapi      # NVENC (CUDA Video SDK)
    )

    # ── Room for GPU-specific tests ───────────────────────────────────
    add_executable(test-cuda-gpu tests/test_cuda_gpu.cu)
    target_link_libraries(test-cuda-gpu PRIVATE vidcrypt)
endif()
```

### Fat Binary Contents

After compilation, `vidcrypt.exe` (or `libvidcrypt.a`) contains:

| Section | Architecture | Type | Size (approx) |
|---------|-------------|------|:-------------:|
| `.nv_fatbin` | sm_50 | PTX | ~50 KB |
| `.nv_fatbin` | sm_75 | SASS | ~50 KB |
| `.nv_fatbin` | sm_80 | SASS | ~50 KB |
| `.nv_fatbin` | sm_86 | SASS | ~50 KB |
| `.nv_fatbin` | sm_89 | SASS | ~50 KB |
| `.nv_fatbin` | sm_90 | SASS | ~50 KB |
| `.nv_fatbin` | sm_90 | PTX | ~50 KB |
| `.nv_fatbin` | sm_100 | SASS | ~50 KB |
| `.nv_fatbin` | sm_120 | SASS | ~50 KB |
| **Total** | | | **~450 KB** |

The driver selects the optimal section at load time — no manual dispatch needed.

### Build Variants

| Build Type | Command | Use Case |
|-----------|---------|----------|
| CPU-only | `cmake -DUSE_CUDA=OFF ..` | CI/CD, VMs, AMD/Intel GPUs, fallback |
| GPU production | `cmake -DUSE_CUDA=ON ..` | End-user builds, all NVIDIA GPUs |
| GPU debug | `cmake -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Debug ..` | Development, kernel debugging |
| GPU profiling | `cmake -DUSE_CUDA=ON -DENABLE_PROFILING=ON ..` | Performance analysis |

---

## 9. Profiling Plan

### Architecture

```c
// gpu_profiling.h

typedef struct {
    // ── Per-stage timestamps (CUDA events for GPU, clock() for CPU) ──

    // GPU decode pipeline
    double nvdec_decode_ms;          // NVDEC decode time (per batch)
    double nv12_to_gray_ms;          // CUDA kernel time
    double block_extract_ms;         // CUDA kernel time
    double sync_validate_ms;         // CUDA kernel time
    double calibration_detect_ms;    // CUDA kernel time (one-time)
    double gpu_pcie_transfer_ms;     // cudaMemcpyAsync time

    // GPU encode pipeline
    double frame_generate_ms;        // CUDA kernel time
    double template_render_ms;       // CUDA kernel time (one-time)
    double nvenc_encode_ms;          // NVENC encode time

    // CPU decode pipeline
    double cpu_frame_normalize_ms;   // frame resize (CPU fallback only)
    double cpu_rs_decode_ms;         // Reed-Solomon decode (thread pool)
    double cpu_file_write_ms;        // file I/O

    // CPU encode pipeline
    double cpu_file_read_ms;         // file I/O
    double cpu_rs_encode_ms;         // Reed-Solomon encode (thread pool)
    double cpu_sha256_ms;            // SHA-256 checksum
    double cpu_ffmpeg_pipe_ms;       // FFmpeg pipe I/O

    // Counters
    int    frames_processed;
    int    rs_failures;
    int    nvdec_surfaces_used;
    int    pcie_transfer_count;
    size_t pcie_total_bytes;
    int    occupancy_theoretical;    // theoretical occupancy %

    // Aggregate
    double total_elapsed_ms;
    double average_fps;
    double effective_fps;            // includes I/O, not just decode
} GpuProfilingReport;

void gpu_profiling_reset(GpuProfilingReport *report);
void gpu_profiling_print(const GpuProfilingReport *report);
bool gpu_profiling_save_csv(const GpuProfilingReport *report, const char *path);
```

### Profiling Output Example

```
╔══════════════════════════════════════════════════════════════════╗
║              VIDCRYPT GPU PROFILING REPORT                       ║
║              System: RTX 5070 | sm_120 | 6400 SMs               ║
║              Preset: PRESET_BLACKWELL_CONSUMER                  ║
║              Mode:   DECODE (HYBRID_CPU_GPU)                    ║
╠══════════════════════════════════════════════════════════════════╣
║ Stage                          Time(ms)    %total    FPS        ║
║ ──────────────────────────────────────────────────────────────── ║
║ NVDEC decode                   1.234      37.0%     810.4       ║
║ nv12_to_gray                   0.045       1.4%                 ║
║ block_extract                  0.289       8.7%                 ║
║ sync_validate                  0.003       0.1%                 ║
║ PCIe transfer (bits)           0.015       0.5%                 ║
║ ─── GPU Total ───              1.586      47.6%                 ║
║                                                                  ║
║ CPU: RS decode                 1.234      37.0%                 ║
║ CPU: File I/O                  0.456      13.7%                 ║
║ CPU: SHA-256                   0.056       1.7%                 ║
║ ─── CPU Total ───              1.746      52.4%                 ║
║ ──────────────────────────────────────────────────────────────── ║
║ TOTAL per frame                3.332     100.0%     300.1 FPS   ║
║                                                                  ║
║ Bottleneck: CPU RS decode (37.0%)                                ║
║ Recommended optimization: Optimize RS with GF(256) lookup tables ║
╚══════════════════════════════════════════════════════════════════╝
```

### Bottleneck Priority Table (Expected)

| Stage | Expected % | Movable to GPU? | Priority |
|-------|:---------:|:--------------:|:--------:|
| NVDEC decode | 30-40% | Already GPU (NVDEC) | ✅ No change needed |
| block_extract | 8-12% | ✅ GPU kernel | ✅ Done |
| RS decode | 25-35% | ❌ CPU (branch-heavy) | ⚡ Optimize lookup tables |
| File I/O | 10-15% | ❌ CPU (sequential) | ⚡ Async I/O |
| Frame gen (encoder) | 25-35% | ✅ GPU kernel | ✅ Done |
| Pipeline overhead | 3-5% | ❌ Latency | ⚡ Tune batch sizes |

---

## 10. Implementation Roadmap

### Phase 0: Baseline Profiling (Week 1)

- [ ] Add `ENABLE_PROFILING` instrumentation to existing CPU code
- [ ] Profile decode and encode on target RTX 5070 with representative files
- [ ] Publish `profiling_baseline.md` with real bottleneck data
- [ ] Validate that the 10% rule holds before GPU implementation

### Phase 1: GPU Backend Infrastructure (Week 2)

- [ ] Create `gpu_presets.h` / `gpu_presets.c` — all GPU presets + lookup + refine
- [ ] Create `gpu_backend.h` / `gpu_backend.c` — lifecycle, memory pool, stream management
- [ ] Update `backend.c` to use real GPU dispatch instead of stubs
- [ ] Update CMakeLists.txt with multi-arch configuration
- [ ] **Deliverable:** `--backend gpu` detects GPU, loads preset, reports capabilities

### Phase 2: Decoder GPU Kernels (Week 3-4)

- [ ] Implement `nv12_to_gray_kernel`
- [ ] Implement `block_extract_kernel_tiled` with per-preset launch config
- [ ] Implement `sync_validate_kernel` with warp-level ballot
- [ ] Implement `calibration_detect_kernel`
- [ ] Wire NVDEC → GPU kernels → bits → CPU RS pipeline
- [ ] **Deliverable:** GPU decode pipeline functional, bit-identical to CPU

### Phase 3: Encoder GPU Kernels (Week 4-5)

- [ ] Implement `frame_generate_kernel` with per-preset config
- [ ] Implement template pre-render (calibration + sync, one-time upload to GPU)
- [ ] Implement NVENC encoder wrapper
- [ ] Fallback: pinned memory → FFmpeg pipe for older GPUs without NVENC AV1
- [ ] **Deliverable:** GPU encode pipeline functional, bit-identical to CPU

### Phase 4: NVDEC Integration (Week 5-6)

- [ ] Implement `gpu_nvdec_create` / `gpu_nvdec_decode_frame` / `gpu_nvdec_destroy`
- [ ] Handle all supported codecs: H.264, HEVC, AV1
- [ ] Handle resolution changes, codec detection
- [ ] Fallback path: ffmpeg pipe → CUDA memory transfer
- [ ] **Deliverable:** NVDEC feeds frames directly into GPU memory

### Phase 5: Profiling & Tuning (Week 6-7)

- [ ] Implement `GpuProfilingReport` with CUDA events + CPU timers
- [ ] Add profile-guided occupancy tuning at startup
- [ ] Tune kernel launch parameters per preset
- [ ] Optimize RS decode bottleneck if needed (maybe SIMD RS?)
- [ ] **Deliverable:** ≥ 500 FPS decode, ≥ 500 FPS encode at bit-perfect parity

### Phase 6: Validation & Release (Week 7-8)

- [ ] Run full test suite on all available GPU generations
- [ ] Test CPU-only builds on systems without NVIDIA GPUs
- [ ] Test `--backend auto` fallback
- [ ] Add CI/CD jobs for CPU-only and CUDA builds
- [ ] Publish documentation and build guide
- [ ] **Deliverable:** v3.0 release with production-grade GPU acceleration

---

## 11. CPU-Only Mode Guarantee

### Hard Requirements

```c
// ── In CMakeLists.txt: ───────────────────────────────────────────────
// USE_CUDA defaults to OFF. CPU-only is the default build.
option(USE_CUDA "Enable CUDA-based hybrid backend" OFF)

// ── In every GPU source file: ────────────────────────────────────────
#ifdef USE_CUDA
// ... GPU code ...
#endif

// ── In backend.c dispatch: ───────────────────────────────────────────
bool decoder_decode_file_gpu(...) {
#ifndef USE_CUDA
    return false;  // Stub: GPU not available at compile time
#else
    // ... actual GPU implementation ...
#endif
}
```

### Validation Matrix

| Scenario | Expected Behavior |
|----------|------------------|
| No NVIDIA GPU, `USE_CUDA=OFF` | Compiles and runs CPU-only |
| No NVIDIA GPU, `USE_CUDA=ON` | Compiles (no CUDA deps at runtime), detects 0 GPUs → CPU fallback |
| NVIDIA GPU, `--backend cpu` | Uses CPU pipeline (bypasses CUDA entirely) |
| NVIDIA GPU, `--backend auto` | Uses GPU pipeline |
| NVIDIA GPU, `--backend gpu` | Uses GPU pipeline; fails if CUDA errors occur |
| AMD/Intel GPU, any build | CPU pipeline only |
| VM without GPU passthrough | CPU pipeline only |
| CI/CD environment | CPU pipeline only (USE_CUDA=OFF in CI) |

### Bit-Exactness Verification

```bash
# 1. Encode with CPU (reference)
./vidcrypt-encoder --backend cpu -i test.bin -o cpu_ref.avi

# 2. Decode with CPU (reference output)
./vidcrypt-decoder --backend cpu -i cpu_ref.avi
sha256sum test.bin        # original file checksum
sha256sum original.bin    # decoded file checksum

# 3. Decode CPU-encoded video with GPU
./vidcrypt-decoder --backend gpu -i cpu_ref.avi
sha256sum output.bin      # MUST match step 2

# 4. Encode with GPU
./vidcrypt-encoder --backend gpu -i test.bin -o gpu_test.avi

# 5. Decode GPU-encoded video with CPU
./vidcrypt-decoder --backend cpu -i gpu_test.avi
sha256sum output.bin      # MUST match step 2

# 6. Full round-trip: GPU encode → GPU decode
./vidcrypt-encoder --backend gpu -i test.bin -o gpu_only.avi
./vidcrypt-decoder --backend gpu -i gpu_only.avi
sha256sum output.bin      # MUST match step 2

# 7. Auto mode on system with GPU
./vidcrypt-decoder --backend auto -i cpu_ref.avi
sha256sum output.bin      # MUST match step 2

# 8. All checksums identical → GPU acceleration is bit-transparent
```

### Build Instructions

```bash
# CPU-only (any system)
mkdir build-cpu && cd build-cpu
cmake .. -DUSE_CUDA=OFF
cmake --build .

# GPU accelerated (requires NVIDIA CUDA toolkit + Video SDK)
mkdir build-gpu && cd build-gpu
cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build .

# Portable: build with GPU support, but it still runs on non-NVIDIA systems
# (falls back to CPU at runtime)
```

---

## 12. Testing & Validation

### Unit Tests (CUDA)

```cuda
// tests/test_cuda_kernels.cu

// 1. block_extract: verify bit-correct against CPU reference
//    - Generate known test pattern frame on GPU
//    - Run block_extract_kernel
//    - Compare bits against CPU-computed reference

// 2. sync_validate: verify correct detection
//    - Frame with perfect sync pattern → confidence = 1.0, pass = true
//    - Frame with corrupted sync → confidence < 1.0, pass = false

// 3. nv12_to_gray: verify pixel-accurate conversion
//    - Generate NV12 frame, convert, compare against CPU reference

// 4. frame_generate: verify encoder produces correct BGR output
//    - Known bits → generate frame → compare pixel values

// 5. Calibration detection: verify correct CalParams extraction
//    - Known calibration → extract → compare fields

// 6. Multi-GPU: verify same output across different compute capabilities
//    - Run on sm_50, sm_75, sm_86, sm_120 → output identical
```

### Integration Tests

```bash
# Test all backend combinations
for mode in cpu gpu auto; do
    echo "=== Testing DECODE with --backend $mode ==="
    ./vidcrypt-decoder --backend $mode -i test_vectors/reference.avi
    sha256sum output.bin >> results_$mode.txt
done

# Verify all checksums match reference
diff results_cpu.txt results_gpu.txt   # must be identical
diff results_cpu.txt results_auto.txt  # must be identical

# Test all encode → decode round-trips
for enc_mode in cpu gpu; do
    for dec_mode in cpu gpu; do
        echo "=== Encode($enc_mode) → Decode($dec_mode) ==="
        ./vidcrypt-encoder --backend $enc_mode -i test.bin -o test_${enc_mode}.avi
        ./vidcrypt-decoder --backend $dec_mode -i test_${enc_mode}.avi
        sha256sum test.bin output.bin
    done
done
```

### GPU Stress Tests

```bash
# Long-running decode (1000+ frames)
./vidcrypt-decoder --backend gpu -i long_video.mkv
# Verify: no memory leaks, no CUDA errors, stable FPS throughout

# Large file encode (>1 GB)
./vidcrypt-encoder --backend gpu -i large_file.bin -o large.avi
# Verify: memory stays bounded, no OOM

# Multiple concurrent encodes/decodes (if applicable for pipeline)
# Verify: no CUDA context conflicts
```

---

---

## 13. Backlog & Open Questions (Pre-Implementation Validation)

The following backlog items must be completed before any CUDA kernel implementation begins.
All GPU acceleration decisions must be backed by measured profiling data, not assumptions.

> **Golden rule:** Profile first. Optimize second. No stage consuming < 10% of runtime
> should receive GPU acceleration without clear justification.

### Priority 0 — Establish Real Bottlenecks

Current GPU design assumes `block_extract` and frame processing are major bottlenecks.
This has **NOT** yet been proven. Measure runtime spent in:

| Stage | Expected % | GPU Candidate? |
|-------|:---------:|:--------------:|
| FFmpeg video decode | ? | 🟡 NVDEC candidate |
| Frame acquisition | ? | ❌ I/O bound |
| Calibration extraction | ? | ❌ One-time cost |
| Header parsing | ? | ❌ One-time cost |
| Grid analysis | ? | ❌ One-time cost |
| Bit extraction (tile decode) | ? | ✅ Primary CUDA candidate |
| Reed-Solomon decode | ? | ❌ Branch-heavy, keep CPU |
| Reed-Solomon encode | ? | ❌ Branch-heavy, keep CPU |
| SHA-256 verification | ? | ❌ < 1% expected |
| File reconstruction | ? | ❌ I/O bound |
| Disk I/O | ? | ❌ I/O bound |

**Required output:**
```
Decode Profile
  Video Decode:       X ms  (X.X%)
  Frame Analysis:     X ms  (X.X%)
  Bit Extraction:     X ms  (X.X%)
  RS Decode:          X ms  (X.X%)
  SHA256:             X ms  (X.X%)
  File Write:         X ms  (X.X%)
  ─────────────────────────────────
  Total:              X ms  (100%)
```

The profiling instrumentation in `src/profiling.h` / `src/profiling.c` (compile with
`-DENABLE_PROFILING=ON`) measures all of these stages automatically.

### Priority 1 — Verify NVDEC Value

**Question:** Is FFmpeg video decode the dominant bottleneck? Does NVDEC help?

**Benchmark:** Compare CPU decode vs NVDEC decode on identical files.

If video decode is already fast (< 15% of runtime), NVDEC integration is lower priority.

### Priority 2 — Verify Block Extraction Cost

**Question:** Is tile thresholding (white-pixel counting, grid traversal) actually 40-50%?

The profiling instrumentation measures `PROF_DECODE_BIT_EXTRACTION` which captures
all `decode_payload_tiles` calls (including per-frame worker threads via atomics).

**Decision rule:**
- If < 10%: Do not GPU-accelerate
- If 10-25%: Consider GPU with low implementation cost
- If > 25%: Strong GPU candidate

### Priority 3 — Reed-Solomon Analysis

RS is expected to stay on CPU, but it must be measured.

If RS exceeds 25-30% of total runtime, investigate:
- SIMD acceleration (AVX2, AVX512)
- Lookup-table optimizations
- Multi-threaded chunk processing

**before** considering GPU-based RS (which is unlikely to help due to branch complexity).

### Priority 4 — Verify RTX 5070 Architecture

**Current assumption:** RTX 5070 = sm_120 (Blackwell consumer).

**Must verify using:**
```bash
nvidia-smi
# Or programmatically:
cudaGetDeviceProperties(&prop, 0);
printf("%d.%d\n", prop.major, prop.minor);
```

All presets are selected dynamically at runtime — never hard-coded.

### Priority 5 — NVENC Lossless Codec Validation

VidCrypt stores binary payloads as block patterns. Lossy codecs may introduce:
- Threshold errors (pixel values shifted by compression)
- Sync row corruption (bit flips in sync pattern)
- RS correction overhead (more ECC consumed by compression artifacts)
- Checksum failures

**Required test matrix:**

| Codec | Mode | Encode Speed | Decode Speed | Checksum Success |
|-------|------|:-----------:|:-----------:|:----------------:|
| FFV1 | lossless | baseline | baseline | ✅ reference |
| H.264 | lossless | TBD | TBD | TBD |
| HEVC | lossless | TBD | TBD | TBD |
| AV1 | lossless | TBD | TBD | TBD |

FFV1 remains the reference format until lossless NVENC codecs are proven equivalent.

### Priority 6 — GPU Transfer Validation

**Question:** How much time is saved by transferring bits (~3KB) vs full frames (~3MB)?

**Test:** Time `cudaMemcpy` for both sizes.

**Expected result:** Bits-only transfer is ~1000× faster. Verify this holds in practice.

### Priority 7 — Maintain CPU Reference Path

**Non-negotiable requirements:**
- CPU mode must remain fully functional (`USE_CUDA=OFF`)
- GPU mode must not alter file format, metadata, or checksum behavior
- CPU and GPU outputs must be **bit-identical**
- CPU implementation remains the canonical reference for correctness testing

### Priority 8 — Success Criteria

GPU implementation is considered successful **only if**:
1. Existing tests pass
2. Output files are identical (bit-for-bit) between CPU and GPU modes
3. SHA-256 verification succeeds on both paths
4. CPU mode remains functional with `--backend cpu`
5. Throughput improvement exceeds **50%**

A 5-10% improvement is insufficient to justify major CUDA complexity.

### Priority 9 — Stretch Goals (Post-Core-Implementation)

Only after all previous items are complete:
- Direct NVDEC → CUDA pipeline (zero-copy)
- GPUDirect Storage (bypass CPU for file I/O)
- Multi-stream frame processing
- Batch frame analysis

These are optimization phases, not initial implementation requirements.

---

## Appendix: Quick Reference

### Decode Pipeline — What Runs Where

| Step | Location | GPU? | Notes |
|------|----------|:----:|-------|
| Video container demux | CPU (FFmpeg) | ❌ | ffmpeg subprocess |
| Video decode | NVDEC | ✅ | Hardware decoder |
| NV12 → grayscale | CUDA kernel | ✅ | nv12_to_gray |
| Block extraction | CUDA kernel | ✅ | block_extract |
| Sync validation | CUDA kernel | ✅ | sync_validate |
| Calibration detect | CUDA kernel | ✅ | calibration_detect (once) |
| Bit transfer to CPU | PCIe (cudaMemcpyAsync) | ✅ | ~3KB |
| Reed-Solomon decode | CPU thread pool | ❌ | Branch-heavy |
| File reconstruction | CPU (sequential) | ❌ | Sequential I/O |
| SHA-256 verify | CPU | ❌ | <1% runtime |

### Encode Pipeline — What Runs Where

| Step | Location | GPU? | Notes |
|------|----------|:----:|-------|
| File read | CPU (sequential) | ❌ | Buffered I/O |
| Reed-Solomon encode | CPU thread pool | ❌ | Branch-heavy |
| Bit transfer to GPU | PCIe (cudaMemcpyAsync) | ✅ | ~3KB |
| Frame generation | CUDA kernel | ✅ | frame_generate |
| Calibration overlay | CUDA kernel | ✅ | One-time |
| Video encode | NVENC / FFmpeg | ✅/❌ | NVENC preferred |
| Container mux | CPU (FFmpeg) | ❌ | ffmpeg subprocess |

### CLI Flags

```
Backend selection:
  --backend cpu     Force CPU-only pipeline
  --backend gpu     Force GPU pipeline (fail if unavailable)
  --backend auto    Auto-detect GPU, fall back to CPU (default)

GPU info:
  --gpu-info        List detected NVIDIA GPUs and capabilities

Profiling:
  --profile         Print profiling report on completion
  --profile-csv     Export profiling data to CSV
```

---

> **Document version:** 3.0  
> **Last updated:** 2026-06-17  
> **Target GPU:** NVIDIA RTX 5070 (sm_120) and all major NVIDIA GPUs from Maxwell (sm_50) to Blackwell (sm_120)

#include "gpu_presets.h"
#include <string.h>

/* ─── Per-architecture presets ────────────────────────────────────── */

static const GpuPreset PRESET_MAXWELL = {
    .compute_capability_major = 5, .compute_capability_minor = 0,
    .architecture_name = "Maxwell",
    .decode_block_x = 32, .decode_block_y = 4,
    .decode_smem_bytes = 8192,
    .num_cuda_streams = 2, .num_nvdec_surfaces = 4,
    .batch_frames_per_launch = 4,
    .has_nvdec = true, .has_nvenc = false,
    .has_av1_encode = false, .has_async_copy = false,
    .max_width = 4096, .max_height = 4096,
};

static const GpuPreset PRESET_PASCAL = {
    .compute_capability_major = 6, .compute_capability_minor = 0,
    .architecture_name = "Pascal",
    .decode_block_x = 32, .decode_block_y = 8,
    .decode_smem_bytes = 12288,
    .num_cuda_streams = 2, .num_nvdec_surfaces = 6,
    .batch_frames_per_launch = 8,
    .has_nvdec = true, .has_nvenc = true,
    .has_av1_encode = false, .has_async_copy = false,
    .max_width = 4096, .max_height = 4096,
};

static const GpuPreset PRESET_VOLTA = {
    .compute_capability_major = 7, .compute_capability_minor = 0,
    .architecture_name = "Volta",
    .decode_block_x = 32, .decode_block_y = 8,
    .decode_smem_bytes = 16384,
    .num_cuda_streams = 3, .num_nvdec_surfaces = 8,
    .batch_frames_per_launch = 12,
    .has_nvdec = true, .has_nvenc = true,
    .has_av1_encode = false, .has_async_copy = false,
    .max_width = 7680, .max_height = 4320,
};

static const GpuPreset PRESET_TURING = {
    .compute_capability_major = 7, .compute_capability_minor = 5,
    .architecture_name = "Turing",
    .decode_block_x = 32, .decode_block_y = 16,
    .decode_smem_bytes = 16384,
    .num_cuda_streams = 4, .num_nvdec_surfaces = 10,
    .batch_frames_per_launch = 16,
    .has_nvdec = true, .has_nvenc = true,
    .has_av1_encode = false, .has_async_copy = false,
    .max_width = 7680, .max_height = 4320,
};

static const GpuPreset PRESET_AMPERE = {
    .compute_capability_major = 8, .compute_capability_minor = 0,
    .architecture_name = "Ampere",
    .decode_block_x = 32, .decode_block_y = 16,
    .decode_smem_bytes = 24576,
    .num_cuda_streams = 4, .num_nvdec_surfaces = 12,
    .batch_frames_per_launch = 24,
    .has_nvdec = true, .has_nvenc = true,
    .has_av1_encode = false, .has_async_copy = true,
    .max_width = 7680, .max_height = 4320,
};

static const GpuPreset PRESET_ADA = {
    .compute_capability_major = 8, .compute_capability_minor = 9,
    .architecture_name = "Ada Lovelace",
    .decode_block_x = 32, .decode_block_y = 16,
    .decode_smem_bytes = 32768,
    .num_cuda_streams = 6, .num_nvdec_surfaces = 12,
    .batch_frames_per_launch = 32,
    .has_nvdec = true, .has_nvenc = true,
    .has_av1_encode = true, .has_async_copy = true,
    .max_width = 7680, .max_height = 4320,
};

static const GpuPreset PRESET_HOPPER = {
    .compute_capability_major = 9, .compute_capability_minor = 0,
    .architecture_name = "Hopper",
    .decode_block_x = 32, .decode_block_y = 32,
    .decode_smem_bytes = 49152,
    .num_cuda_streams = 8, .num_nvdec_surfaces = 16,
    .batch_frames_per_launch = 48,
    .has_nvdec = true, .has_nvenc = true,
    .has_av1_encode = true, .has_async_copy = true,
    .max_width = 8192, .max_height = 8192,
};

static const GpuPreset PRESET_BLACKWELL_DC = {
    .compute_capability_major = 10, .compute_capability_minor = 0,
    .architecture_name = "Blackwell (DC)",
    .decode_block_x = 32, .decode_block_y = 32,
    .decode_smem_bytes = 65536,
    .num_cuda_streams = 8, .num_nvdec_surfaces = 16,
    .batch_frames_per_launch = 64,
    .has_nvdec = true, .has_nvenc = true,
    .has_av1_encode = true, .has_async_copy = true,
    .max_width = 8192, .max_height = 8192,
};

static const GpuPreset PRESET_BLACKWELL_CONSUMER = {
    .compute_capability_major = 12, .compute_capability_minor = 0,
    .architecture_name = "Blackwell (RTX 50)",
    .decode_block_x = 32, .decode_block_y = 32,
    .decode_smem_bytes = 49152,
    .num_cuda_streams = 8, .num_nvdec_surfaces = 16,
    .batch_frames_per_launch = 64,
    .has_nvdec = true, .has_nvenc = true,
    .has_av1_encode = true, .has_async_copy = true,
    .max_width = 8192, .max_height = 8192,
};

/* ─── Preset lookup ──────────────────────────────────────────────── */
const GpuPreset* gpu_preset_lookup(int major, int minor) {
    int cc = major * 100 + minor;
    if (cc >= 1200) return &PRESET_BLACKWELL_CONSUMER;
    if (cc >= 1000) return &PRESET_BLACKWELL_DC;
    if (cc >= 900)  return &PRESET_HOPPER;
    if (cc >= 890)  return &PRESET_ADA;
    if (cc >= 800)  return &PRESET_AMPERE;
    if (cc >= 750)  return &PRESET_TURING;
    if (cc >= 700)  return &PRESET_VOLTA;
    if (cc >= 600)  return &PRESET_PASCAL;
    if (cc >= 500)  return &PRESET_MAXWELL;
    return NULL;
}

const char* gpu_arch_name(int major, int minor) {
    const GpuPreset *p = gpu_preset_lookup(major, minor);
    return p ? p->architecture_name : "Unknown (CPU fallback)";
}

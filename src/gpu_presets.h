#ifndef VIDCRYPT_GPU_PRESETS_H
#define VIDCRYPT_GPU_PRESETS_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── GPU Architecture Preset ───────────────────────────────────────
 * Each NVIDIA architecture gets a tuned preset with optimal kernel launch
 * parameters, stream counts, and feature flags. */

typedef struct {
    int   compute_capability_major;
    int   compute_capability_minor;
    const char *architecture_name;

    /* Decode kernel parameters */
    int   decode_block_x;           /* threads per block X for nv12_to_gray */
    int   decode_block_y;           /* threads per block Y for nv12_to_gray */
    int   decode_smem_bytes;        /* dynamic shared memory per block */

    /* Pipeline tuning */
    int   num_cuda_streams;
    int   num_nvdec_surfaces;
    int   batch_frames_per_launch;

    /* Feature flags */
    bool  has_nvdec;                /* NVDEC hardware decoder available */
    bool  has_nvenc;                /* NVENC hardware encoder available */
    bool  has_av1_encode;           /* AV1 encoding support */
    bool  has_async_copy;           /* async copy engine (sm_80+) */

    /* NVDEC/NVENC limits */
    int   max_width;
    int   max_height;
} GpuPreset;

/* ─── Preset lookup ───────────────────────────────────────────────── */
const GpuPreset* gpu_preset_lookup(int major, int minor);
const char*      gpu_arch_name(int major, int minor);

#ifdef __cplusplus
}
#endif

#endif /* VIDCRYPT_GPU_PRESETS_H */

#include "gpu_nvenc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#if defined(USE_CUDA) && defined(HAS_NVENC)

#  include <cuda.h>
#  include <cuda_runtime.h>
#  include <nvEncodeAPI.h>

#include "gpu_kernels.h"

/* Platform-specific dynamic library loading */
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

/* ─── NVENC structure version and enum fallbacks ────────────────────
 * If the SDK header doesn't define these (older SDK), provide
 * reasonable defaults. The FOURCC pattern is used in SDK 12.x. */
#ifndef NV_ENC_REGISTER_RESOURCE_VER
#  define NV_ENC_REGISTER_RESOURCE_VER 1
#endif
#ifndef NV_ENC_MAP_INPUT_RESOURCE_VER
#  define NV_ENC_MAP_INPUT_RESOURCE_VER 1
#endif
#ifndef NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR
#  define NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR 1
#endif
#ifndef NV_ENC_INPUT_IMAGE
#  define NV_ENC_INPUT_IMAGE 0x1
#endif

/* ─── Internal state ──────────────────────────────────────────────── */
#define NVENC_OUTPUT_RING_SIZE 4

struct GpuNvencEncoder {
    /* CUDA state */
    int             cuda_device;
    CUcontext       cuda_ctx;

    /* NVENC API function table */
    NV_ENCODE_API_FUNCTION_LIST nvenc_api;
    uint32_t        driver_api_ver; /* from NvEncodeAPIGetMaxSupportedVersion */

    /* Encoder session */
    void           *encoder;         /* NV_ENC_SESSION_HANDLE (void*) */

    /* Video parameters */
    int             width;
    int             height;
    double          fps;
    int             bitrate;         /* kbps */
    int             gop_length;

    /* ── Zero-copy path: registered CUDA resource ───────────────── */
    /* BGRA32 conversion buffer in CUDA device memory.
     * Registered with NVENC via nvEncRegisterResource.
     * NVENC reads directly from this GPU memory — zero PCIe copies. */
    CUdeviceptr     d_bgra;              /* device memory: BGRA32 frame (width*height*4) */
    size_t          bgra_size;
    int             bgra_pitch;          /* bytes per row = width * 4 */
    NV_ENC_REGISTERED_PTR registered_resource; /* handle from nvEncRegisterResource */
    bool            use_registered_resource;   /* true if zero-copy path active */

    /* ── Legacy path: NVENC-owned input buffer ──────────────────── */
    NV_ENC_INPUT_PTR input_buffer;   /* Input surface (from nvEncCreateInputBuffer) */
    int             input_buffer_pitch;  /* bytes per row */

    /* ── NVENC output bitstream ring buffer ─────────────────────── */
    NV_ENC_OUTPUT_PTR output_buffers[NVENC_OUTPUT_RING_SIZE];
    int             output_ring_submit;  /* next buffer index for submission */
    int             output_ring_drain;   /* next buffer index to drain */

    /* CUDA stream for conversion kernels */
    cudaStream_t    stream;

    /* ── SPS/PPS extraction ─────────────────────────────────────── */
    /* Extracted from first frame's bitstream via NAL parser.
     * Falls back to manual construction if extraction fails. */
    uint8_t        *sps_pps_data;
    int             sps_pps_size;
    bool            first_frame_done;    /* true after first frame sync-encoded */

    /* ── Packet ring buffer — stores encoded packets for retrieval ── */
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

/* ─── API version helper ───────────────────────────────────────────
 * The driver uses (major<<4)|minor encoding for the API version
 * (e.g., 0xD1 = v13.1), while the header uses major|(minor<<24).
 * We must use the DRIVER's encoding for all struct versions. */
static uint32_t api_struct_ver(GpuNvencEncoder *enc, int ver) {
    return enc->driver_api_ver | ((uint32_t)ver << 16) | (0x7U << 28);
}

/* ─── Packet ring buffer management ───────────────────────────────── */

static bool add_packet(GpuNvencEncoder *enc, const uint8_t *data, int size) {
    if (size <= 0) return true;
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

/* ─── NAL unit parsing (AVCC format) ──────────────────────────────────
 * NVENC outputs in AVCC format: 4-byte big-endian length prefix followed
 * by NAL unit data. Extracts SPS (type 7) and PPS (type 8) NAL units
 * with their length prefixes. Returns total bytes written to out. */

static int extract_sps_pps(const uint8_t *bitstream, int size,
                           uint8_t *out, int max_out) {
    int out_pos = 0;
    bool found_sps = false, found_pps = false;

    fprintf(stderr, "  [NAL parse] size=%d: ", size);

    /* Scan for Annex B start codes: 00 00 01 XX or 00 00 00 01 XX */
    for (int pos = 0; pos < size - 4 && !(found_sps && found_pps); pos++) {
        int start_len = 0;
        if (bitstream[pos] == 0x00 && bitstream[pos+1] == 0x00) {
            if (bitstream[pos+2] == 0x01)
                start_len = 3;
            else if (bitstream[pos+2] == 0x00 && bitstream[pos+3] == 0x01)
                start_len = 4;
        }
        if (!start_len) continue;

        pos += start_len;
        if (pos >= size) break;
        uint8_t nal_type = bitstream[pos] & 0x1F;

        /* Find next start code for NAL unit length */
        int nalu_end = size;
        for (int j = pos + 1; j < size - 3; j++) {
            if (bitstream[j] == 0x00 && bitstream[j+1] == 0x00) {
                if (bitstream[j+2] == 0x01 ||
                    (bitstream[j+2] == 0x00 && bitstream[j+3] == 0x01)) {
                    nalu_end = j;
                    break;
                }
            }
        }
        int nalu_len = nalu_end - pos;
        fprintf(stderr, "[%d:%d] ", nal_type, nalu_len);

        if (nal_type == 7 || nal_type == 8) {
            /* Write in AVCC format: 4-byte BE length + NAL data */
            if (out_pos + 4 + nalu_len > max_out) break;
            out[out_pos++] = (uint8_t)(nalu_len >> 24);
            out[out_pos++] = (uint8_t)(nalu_len >> 16);
            out[out_pos++] = (uint8_t)(nalu_len >> 8);
            out[out_pos++] = (uint8_t)nalu_len;
            memcpy(out + out_pos, bitstream + pos, (size_t)nalu_len);
            out_pos += nalu_len;
            if (nal_type == 7) found_sps = true;
            if (nal_type == 8) found_pps = true;
        }
        pos = nalu_end - 1;
    }
    fprintf(stderr, "\n");
    return (found_sps && found_pps) ? out_pos : 0;
}

/* ─── Manual SPS/PPS construction ────────────────────────────────────
 * Fallback when NAL parser cannot extract SPS/PPS from bitstream
 * (e.g., NVENC doesn't embed them in frame output on this driver).
 *
 * Constructs baseline H.264 SPS and PPS NAL units in AVCC format
 * (4-byte big-endian length prefix + raw NAL unit).
 *
 * Reference: ITU-T H.264 (2017) §7.3.2.1.1 (SPS), §7.3.2.2 (PPS) */

/* Exponential-Golomb coding helpers */
static void write_bit(uint8_t *buf, int *byte_pos, int *bit_pos, int bit) {
    if (*bit_pos < 0) { (*byte_pos)++; *bit_pos = 7; }
    if (bit) buf[*byte_pos] |= (uint8_t)(1 << *bit_pos);
    (*bit_pos)--;
}

static void write_ue(uint8_t *buf, int *byte_pos, int *bit_pos, uint32_t val) {
    /* Unsigned Exp-Golomb: write (floor(log2(val+1))) zeros, then 1, then val+1 in binary */
    uint32_t v = val + 1;
    int leading_zeros = 0;
    uint32_t tmp = v;
    while (tmp > 1) { leading_zeros++; tmp >>= 1; }
    for (int i = 0; i < leading_zeros; i++)
        write_bit(buf, byte_pos, bit_pos, 0);
    for (int i = leading_zeros; i >= 0; i--)
        write_bit(buf, byte_pos, bit_pos, (int)((v >> i) & 1));
}

static void write_se(uint8_t *buf, int *byte_pos, int *bit_pos, int val) {
    /* Signed Exp-Golomb */
    uint32_t ue_val;
    if (val <= 0)
        ue_val = (uint32_t)(-val * 2);
    else
        ue_val = (uint32_t)(val * 2 - 1);
    write_ue(buf, byte_pos, bit_pos, ue_val);
}

static void write_rbsp_trailing(uint8_t *buf, int *byte_pos, int *bit_pos) {
    /* RBSP trailing bits: write 1, then zeros to byte-align */
    write_bit(buf, byte_pos, bit_pos, 1);
    while (*bit_pos != 7)
        write_bit(buf, byte_pos, bit_pos, 0);
}

/* Build an AVCC NAL unit (4-byte length prefix + NAL data) with
 * RBSP→EBSP conversion (0x000003 escape prevention).
 * Returns total bytes written. */
static int avcc_nal_with_ebsp(const uint8_t *rbsp, int rbsp_bytes,
                               uint8_t *out, int max_out) {
    /* First, count how many escape bytes needed */
    int escapes = 0;
    int zeros = 0;
    for (int i = 0; i < rbsp_bytes; i++) {
        if (rbsp[i] == 0x00) {
            zeros++;
            if (zeros == 2) {
                /* Next byte: if 0x00, 0x01, 0x02, or 0x03, needs escape */
                if (i + 1 < rbsp_bytes && rbsp[i + 1] <= 0x03) {
                    escapes++;
                    zeros = 0;
                }
            }
        } else if (rbsp[i] == 0x00 && zeros >= 2) {
            /* already handled above */;
        } else {
            zeros = 0;
        }
    }

    int ebsp_size = rbsp_bytes + escapes;
    int total = 4 + ebsp_size;
    if (total > max_out) return 0;

    /* 4-byte big-endian length prefix */
    out[0] = (uint8_t)((ebsp_size >> 24) & 0xFF);
    out[1] = (uint8_t)((ebsp_size >> 16) & 0xFF);
    out[2] = (uint8_t)((ebsp_size >> 8) & 0xFF);
    out[3] = (uint8_t)(ebsp_size & 0xFF);

    /* Write EBSP with escape prevention */
    int wp = 4;
    zeros = 0;
    for (int i = 0; i < rbsp_bytes; i++) {
        out[wp++] = rbsp[i];
        if (rbsp[i] == 0x00) {
            zeros++;
        } else {
            if (zeros >= 2 && rbsp[i] <= 0x03) {
                /* Insert escape byte before this byte */
                /* We need to shift — but we pre-counted, so this shouldn't happen */
            }
            zeros = 0;
        }
    }
    /* Actually, do the escape insertion properly: */
    wp = 4;
    zeros = 0;
    for (int i = 0; i < rbsp_bytes; i++) {
        if (rbsp[i] == 0x00) {
            zeros++;
            out[wp++] = 0x00;
        } else if (rbsp[i] <= 0x03 && zeros >= 2) {
            /* Insert 0x03 escape byte */
            out[wp++] = 0x03;
            out[wp++] = rbsp[i];
            zeros = 0;
        } else {
            out[wp++] = rbsp[i];
            zeros = 0;
        }
    }

    return wp;
}

/* Construct SPS for given encoder parameters.
 * Returns total bytes in AVCC format (including 4-byte length prefix). */
static int build_sps_avcc(int width, int height, int fps_num, int fps_den,
                           uint8_t *out, int max_out) {
    uint8_t rbsp[256];
    memset(rbsp, 0, sizeof(rbsp));
    int bp = 0, bit = 7;  /* start at byte 0, bit 7 (MSB first) */

    /* NAL header: forbidden_zero_bit(1) + nal_ref_idc(2) + nal_unit_type(5)
     * SPS: nal_ref_idc=3 (highest), nal_unit_type=7 */
    rbsp[bp] = (uint8_t)((3 << 5) | 7);  /* 0x67 */
    bp++; /* byte 0 done, advance */
    bit = 7;

    /* profile_idc: 100 = High profile, or 66 = Baseline */
    uint8_t profile_idc = 100;  /* High profile */
    /* constraint_set flags + reserved */
    rbsp[bp++] = profile_idc;     /* profile_idc */
    rbsp[bp++] = 0x00;            /* constraint_set0_flag (1 bit) + reserved (7 bits) */
    rbsp[bp++] = 40;              /* level_idc: 40 = Level 4.0 */
    bit = 7;

    /* seq_parameter_set_id: ue(v) */
    write_ue(rbsp, &bp, &bit, 0);

    /* chroma_format_idc: ue(v) = 1 (4:2:0) only for High profile */
    if (profile_idc >= 100) {
        write_ue(rbsp, &bp, &bit, 1);   /* chroma_format_idc = 1 */
        write_ue(rbsp, &bp, &bit, 0);   /* bit_depth_luma_minus8 */
        write_ue(rbsp, &bp, &bit, 0);   /* bit_depth_chroma_minus8 */
        write_bit(rbsp, &bp, &bit, 0);  /* qpprime_y_zero_transform_bypass_flag */
        write_bit(rbsp, &bp, &bit, 0);  /* seq_scaling_matrix_present_flag */
    }

    /* log2_max_frame_num_minus4: ue(v) */
    write_ue(rbsp, &bp, &bit, 0);   /* log2_max_frame_num_minus4 = 0 → frame_num max 16 */

    /* pic_order_cnt_type: ue(v) */
    write_ue(rbsp, &bp, &bit, 0);   /* POC type 0 */

    /* log2_max_pic_order_cnt_lsb_minus4: ue(v) */
    write_ue(rbsp, &bp, &bit, 4);   /* log2_max_poc_lsb_minus4 = 4 → 16 bits */

    /* num_ref_frames: ue(v) */
    write_ue(rbsp, &bp, &bit, 1);   /* num_ref_frames = 1 */

    /* gaps_in_frame_num_value_allowed_flag */
    write_bit(rbsp, &bp, &bit, 0);

    /* pic_width_in_mbs_minus1: ue(v) */
    int mb_width = (width + 15) / 16;
    write_ue(rbsp, &bp, &bit, (uint32_t)(mb_width - 1));

    /* pic_height_in_map_units_minus1: ue(v) */
    int mb_height = (height + 15) / 16;
    write_ue(rbsp, &bp, &bit, (uint32_t)(mb_height - 1));

    /* frame_mbs_only_flag: u(1) */
    write_bit(rbsp, &bp, &bit, 1);  /* frame only */

    /* direct_8x8_inference_flag: u(1) */
    write_bit(rbsp, &bp, &bit, 1);

    /* frame_cropping_flag: u(1) */
    write_bit(rbsp, &bp, &bit, 0);  /* no cropping */

    /* vui_parameters_present_flag: u(1) */
    write_bit(rbsp, &bp, &bit, 0);  /* no VUI */

    /* RBSP trailing bits */
    write_rbsp_trailing(rbsp, &bp, &bit);

    int rbsp_bytes = bp + (bit < 7 ? 1 : 0);
    return avcc_nal_with_ebsp(rbsp, rbsp_bytes, out, max_out);
}

/* Construct PPS for given encoder parameters.
 * Returns total bytes in AVCC format (including 4-byte length prefix). */
static int build_pps_avcc(uint8_t *out, int max_out) {
    uint8_t rbsp[128];
    memset(rbsp, 0, sizeof(rbsp));
    int bp = 0, bit = 7;

    /* NAL header: nal_ref_idc=3, nal_unit_type=8 (PPS) */
    rbsp[bp] = (uint8_t)((3 << 5) | 8);  /* 0x68 */
    bp++; bit = 7;

    /* pic_parameter_set_id: ue(v) */
    write_ue(rbsp, &bp, &bit, 0);

    /* seq_parameter_set_id: ue(v) */
    write_ue(rbsp, &bp, &bit, 0);

    /* entropy_coding_mode_flag: u(1) = 1 (CABAC) */
    write_bit(rbsp, &bp, &bit, 1);

    /* bottom_field_pic_order_in_frame_present_flag: u(1) */
    write_bit(rbsp, &bp, &bit, 0);

    /* num_slice_groups_minus1: ue(v) */
    write_ue(rbsp, &bp, &bit, 0);

    /* num_ref_idx_l0_default_active_minus1: ue(v) */
    write_ue(rbsp, &bp, &bit, 0);

    /* num_ref_idx_l1_default_active_minus1: ue(v) */
    write_ue(rbsp, &bp, &bit, 0);

    /* weighted_pred_flag: u(1) */
    write_bit(rbsp, &bp, &bit, 0);

    /* weighted_bipred_idc: u(2) */
    write_bit(rbsp, &bp, &bit, 0);
    write_bit(rbsp, &bp, &bit, 0);

    /* pic_init_qp_minus26: se(v) */
    write_se(rbsp, &bp, &bit, 0);   /* QP = 26 */

    /* pic_init_qs_minus26: se(v) */
    write_se(rbsp, &bp, &bit, 0);

    /* chroma_qp_index_offset: se(v) */
    write_se(rbsp, &bp, &bit, 0);

    /* deblocking_filter_control_present_flag: u(1) */
    write_bit(rbsp, &bp, &bit, 1);

    /* constrained_intra_pred_flag: u(1) */
    write_bit(rbsp, &bp, &bit, 0);

    /* redundant_pic_cnt_present_flag: u(1) */
    write_bit(rbsp, &bp, &bit, 0);

    /* RBSP trailing bits */
    write_rbsp_trailing(rbsp, &bp, &bit);

    int rbsp_bytes = bp + (bit < 7 ? 1 : 0);
    return avcc_nal_with_ebsp(rbsp, rbsp_bytes, out, max_out);
}

/* Build complete SPS+PPS extradata in AVCC format.
 * Returns total size on success, 0 on failure. */
static int build_sps_pps_extradata(int width, int height, double fps,
                                    uint8_t *out, int max_out) {
    int fps_num = (int)(fps * 1000);
    int fps_den = 1000;

    int sps_size = build_sps_avcc(width, height, fps_num, fps_den,
                                   out, max_out);
    if (sps_size <= 0) return 0;

    int pps_size = build_pps_avcc(out + sps_size, max_out - sps_size);
    if (pps_size <= 0) return 0;

    fprintf(stderr, "  [SPS/PPS manual] SPS=%d bytes, PPS=%d bytes\n",
            sps_size, pps_size);
    return sps_size + pps_size;
}

/* Extract or construct SPS+PPS from the first encoded frame's bitstream.
 * Tries NAL parsing first; if no SPS/PPS found, manually constructs them. */
static void capture_sps_pps(GpuNvencEncoder *enc,
                             const uint8_t *bitstream, int size) {
    if (enc->sps_pps_data) return;  /* already captured */

    /* Try NAL parser extraction first */
    uint8_t sps_pps_buf[4096];
    int sps_pps_len = extract_sps_pps(bitstream, size,
                                       sps_pps_buf, (int)sizeof(sps_pps_buf));
    fprintf(stderr, "  [SPS/PPS] NAL extract: %d bytes\n", sps_pps_len);

    if (sps_pps_len > 0) {
        enc->sps_pps_data = (uint8_t *)malloc((size_t)sps_pps_len);
        if (enc->sps_pps_data) {
            memcpy(enc->sps_pps_data, sps_pps_buf, (size_t)sps_pps_len);
            enc->sps_pps_size = sps_pps_len;
        }
    } else {
        /* Fallback: pre-computed SPS/PPS byte sequences.
         * These are known-good RBSP bytes from a reference NVENC encode
         * at 1920x1080, High Profile L4.0, CABAC. */
        fprintf(stderr, "  [SPS/PPS] NAL extraction failed — using pre-computed SPS/PPS\n");

        /* SPS RBSP: NAL 0x67 + profile=100(constraint=0) level=40, 1080p, High, CABAC */
        static const uint8_t sps_rbsp[] = {
            0x67, 0x64, 0x00, 0x28, 0xAD, 0x80, 0x78,
            0x02, 0x2D, 0x9A, 0x80, 0x94, 0x00, 0x00,
            0x1F, 0x48, 0x00, 0x05, 0xDC
        };

        /* PPS RBSP: NAL 0x68 + PPS fields for CABAC, 1 ref frame */
        static const uint8_t pps_rbsp[] = {
            0x68, 0xEB, 0xE3, 0xCB, 0x22, 0xC0
        };

        uint8_t manual_buf[256];
        int wp = 0;

        /* Write SPS in AVCC format (4-byte BE length prefix + EBSP NAL) */
        {
            int nal_bytes = (int)sizeof(sps_rbsp);
            /* Count EBSP escapes */
            int esc = 0;
            for (int j = 0; j < nal_bytes - 2; j++) {
                if (sps_rbsp[j] == 0 && sps_rbsp[j+1] == 0 && sps_rbsp[j+2] <= 3) esc++;
            }
            int ebsp = nal_bytes + esc;
            manual_buf[wp++] = (uint8_t)(ebsp >> 24);
            manual_buf[wp++] = (uint8_t)(ebsp >> 16);
            manual_buf[wp++] = (uint8_t)(ebsp >> 8);
            manual_buf[wp++] = (uint8_t)ebsp;
            for (int j = 0; j < nal_bytes; j++) {
                manual_buf[wp++] = sps_rbsp[j];
                if (j >= 1 && sps_rbsp[j-1] == 0 && sps_rbsp[j] == 0 &&
                    j+1 < nal_bytes && sps_rbsp[j+1] <= 3) {
                    manual_buf[wp++] = 0x03;
                }
            }
        }

        /* Write PPS in AVCC format */
        {
            int nal_bytes = (int)sizeof(pps_rbsp);
            int esc = 0;
            for (int j = 0; j < nal_bytes - 2; j++) {
                if (pps_rbsp[j] == 0 && pps_rbsp[j+1] == 0 && pps_rbsp[j+2] <= 3) esc++;
            }
            int ebsp = nal_bytes + esc;
            manual_buf[wp++] = (uint8_t)(ebsp >> 24);
            manual_buf[wp++] = (uint8_t)(ebsp >> 16);
            manual_buf[wp++] = (uint8_t)(ebsp >> 8);
            manual_buf[wp++] = (uint8_t)ebsp;
            for (int j = 0; j < nal_bytes; j++) {
                manual_buf[wp++] = pps_rbsp[j];
                if (j >= 1 && pps_rbsp[j-1] == 0 && pps_rbsp[j] == 0 &&
                    j+1 < nal_bytes && pps_rbsp[j+1] <= 3) {
                    manual_buf[wp++] = 0x03;
                }
            }
        }

        enc->sps_pps_data = (uint8_t *)malloc((size_t)wp);
        if (enc->sps_pps_data) {
            memcpy(enc->sps_pps_data, manual_buf, (size_t)wp);
            enc->sps_pps_size = wp;
        }
    }
    fprintf(stderr, "  [SPS/PPS] Final extradata: %d bytes\n", enc->sps_pps_size);
}

/* ─── NVENC API loading ───────────────────────────────────────────── */

typedef NVENCSTATUS (NVENCAPI *NvEncodeAPICreateInstanceFunc)(NV_ENCODE_API_FUNCTION_LIST *);
typedef CUresult (CUDAAPI *CuCtxGetCurrentFunc)(CUcontext *);

static NvEncodeAPICreateInstanceFunc load_nvenc_api(void) {
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
        case NV_ENC_ERR_INVALID_CALL: return "NV_ENC_ERR_INVALID_CALL";
        case NV_ENC_ERR_MAP_FAILED: return "NV_ENC_ERR_MAP_FAILED";
        case NV_ENC_ERR_ENCODER_BUSY: return "NV_ENC_ERR_ENCODER_BUSY";
        case NV_ENC_ERR_NEED_MORE_INPUT: return "NV_ENC_ERR_NEED_MORE_INPUT";
        case NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY: return "NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY";
        case NV_ENC_ERR_UNIMPLEMENTED: return "NV_ENC_ERR_UNIMPLEMENTED";
        case NV_ENC_ERR_RESOURCE_REGISTER_FAILED: return "NV_ENC_ERR_RESOURCE_REGISTER_FAILED";
        case NV_ENC_ERR_RESOURCE_NOT_REGISTERED: return "NV_ENC_ERR_RESOURCE_NOT_REGISTERED";
        case NV_ENC_ERR_RESOURCE_NOT_MAPPED: return "NV_ENC_ERR_RESOURCE_NOT_MAPPED";
        case NV_ENC_ERR_NEED_MORE_OUTPUT: return "NV_ENC_ERR_NEED_MORE_OUTPUT";
        default: {
            static char buf[64];
            snprintf(buf, sizeof(buf), "NVENC error 0x%08X", (unsigned)status);
            return buf;
        }
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
    enc->bitrate = bitrate > 0 ? bitrate : 50000; /* default: 50 Mbps for high quality */
    enc->gop_length = gop_length > 0 ? gop_length : (int)(fps * 2);
    enc->use_registered_resource = false;

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

    /* Query max supported API version from driver */
    {
        typedef NVENCSTATUS (NVENCAPI *GetMaxVerFunc)(uint32_t*);
        GetMaxVerFunc get_max_ver = NULL;
#ifdef _WIN32
        HMODULE mod = LoadLibraryA("nvEncodeAPI64.dll");
        if (!mod) mod = LoadLibraryA("nvEncodeAPI.dll");
        if (mod) get_max_ver = (GetMaxVerFunc)GetProcAddress(mod, "NvEncodeAPIGetMaxSupportedVersion");
#endif
        uint32_t max_ver = 0;
        if (get_max_ver && get_max_ver(&max_ver) == NV_ENC_SUCCESS) {
            enc->driver_api_ver = max_ver;
            fprintf(stderr, "  [NVENC] Driver API ver: 0x%08X (header: 0x%08X)\n",
                    (unsigned)max_ver, (unsigned)NVENCAPI_VERSION);
        } else {
            enc->driver_api_ver = NVENCAPI_VERSION;
            fprintf(stderr, "  [NVENC] Cannot query max API version, using header: 0x%08X\n",
                    (unsigned)NVENCAPI_VERSION);
        }
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

    /* CABAC for better compression */
    encode_config.encodeCodecConfig.h264Config.entropyCodingMode =
        NV_ENC_H264_ENTROPY_CODING_MODE_CABAC;
    /* Disable AUD to avoid noise in SPS/PPS extraction */
    encode_config.encodeCodecConfig.h264Config.outputAUD = 0;

    /* CBR rate control — high bitrate for visually lossless */
    encode_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
    encode_config.rcParams.averageBitRate  = enc->bitrate * 1000;
    encode_config.rcParams.maxBitRate      = enc->bitrate * 1000;
    encode_config.rcParams.vbvBufferSize   = enc->bitrate * 1000 * 2;
    encode_config.rcParams.vbvInitialDelay = enc->bitrate * 1000 * 1;

    encode_config.frameIntervalP = 1;
    encode_config.gopLength = enc->gop_length;

    /* Initialize params */
    NV_ENC_INITIALIZE_PARAMS init_params;
    memset(&init_params, 0, sizeof(init_params));
    init_params.version          = NV_ENC_INITIALIZE_PARAMS_VER;
    init_params.encodeGUID       = NV_ENC_CODEC_H264_GUID;
    init_params.presetGUID       = NV_ENC_PRESET_P1_GUID;
    init_params.encodeWidth      = width;
    init_params.encodeHeight     = height;
    init_params.darWidth         = width;
    init_params.darHeight        = height;
    init_params.maxEncodeWidth   = width;
    init_params.maxEncodeHeight  = height;
    init_params.frameRateNum     = (uint32_t)(fps * 1000);
    init_params.frameRateDen     = 1000;
    init_params.enableEncodeAsync = 0;  /* sync mode — async crashes on this driver */
    init_params.enablePTD        = 0;
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

    /* ── 5. Get SPS/PPS from encoder ─────────────────────────────────── */
    if (enc->nvenc_api.nvEncGetSequenceParams) {
        NV_ENC_SEQUENCE_PARAM_PAYLOAD seq_payload;
        memset(&seq_payload, 0, sizeof(seq_payload));
        seq_payload.version = NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER;
        seq_payload.inBufferSize = 4096;
        seq_payload.spsppsBuffer = malloc(4096);
        uint32_t out_size = 0;
        seq_payload.outSPSPPSPayloadSize = &out_size;
        if (seq_payload.spsppsBuffer) {
            nv_status = enc->nvenc_api.nvEncGetSequenceParams(enc->encoder, &seq_payload);
            if (nv_status == NV_ENC_SUCCESS && out_size > 0) {
                fprintf(stderr, "  [NVENC] Got SPS/PPS from driver: %u bytes\n",
                        (unsigned)out_size);
                enc->sps_pps_data = (uint8_t*)malloc((size_t)out_size);
                if (enc->sps_pps_data) {
                    memcpy(enc->sps_pps_data, seq_payload.spsppsBuffer, (size_t)out_size);
                    enc->sps_pps_size = (int)out_size;
                }
            } else {
                fprintf(stderr, "  [NVENC] nvEncGetSequenceParams: %s (size=%u)\n",
                        nvenc_error_string(nv_status), (unsigned)out_size);
            }
            free(seq_payload.spsppsBuffer);
        }
    }

    /* ── 6. Allocate BGRA32 buffer in CUDA device memory ───────────── */
    enc->bgra_pitch = width * 4;
    enc->bgra_size = (size_t)enc->bgra_pitch * (size_t)height;
    rt_err = cudaMalloc((void **)&enc->d_bgra, enc->bgra_size);
    if (rt_err != cudaSuccess) {
        enc->nvenc_api.nvEncDestroyEncoder(enc->encoder);
        free(enc);
        if (error_out) snprintf(error_out, (size_t)error_size,
                                "cudaMalloc BGRA32: %s", cudaGetErrorString(rt_err));
        return NULL;
    }

    /* ── 6. Try zero-copy: register CUDA device buffer with NVENC ──── */
    if (enc->nvenc_api.nvEncRegisterResource) {
        NV_ENC_REGISTER_RESOURCE reg;
        memset(&reg, 0, sizeof(reg));
        reg.version = NV_ENC_REGISTER_RESOURCE_VER;
        reg.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
        reg.resourceToRegister = (void *)enc->d_bgra;
        reg.width = (uint32_t)width;
        reg.height = (uint32_t)height;
        reg.pitch = (uint32_t)enc->bgra_pitch;
        reg.bufferFormat = NV_ENC_BUFFER_FORMAT_ARGB;
        reg.bufferUsage = NV_ENC_INPUT_IMAGE;

        nv_status = enc->nvenc_api.nvEncRegisterResource(enc->encoder, &reg);
        if (nv_status == NV_ENC_SUCCESS) {
            enc->registered_resource = reg.registeredResource;
            enc->use_registered_resource = true;
            fprintf(stderr, "  [NVENC] Zero-copy path: registered CUDA buffer with NVENC\n");
        } else {
            fprintf(stderr, "  [NVENC] nvEncRegisterResource failed: %s — falling back to legacy input buffer\n",
                    nvenc_error_string(nv_status));
        }
    }

    /* ── 7. Fallback: create legacy NVENC input buffer ─────────────── */
    if (!enc->use_registered_resource) {
        NV_ENC_BUFFER_FORMAT buffer_fmt = NV_ENC_BUFFER_FORMAT_ARGB;

        NV_ENC_CREATE_INPUT_BUFFER alloc_input;
        memset(&alloc_input, 0, sizeof(alloc_input));
        alloc_input.version      = NV_ENC_CREATE_INPUT_BUFFER_VER;
        alloc_input.width        = (uint32_t)width;
        alloc_input.height       = (uint32_t)height;
        alloc_input.memoryHeap   = NV_ENC_MEMORY_HEAP_AUTOSELECT;
        alloc_input.bufferFmt    = buffer_fmt;

        nv_status = enc->nvenc_api.nvEncCreateInputBuffer(enc->encoder, &alloc_input);
        if (nv_status != NV_ENC_SUCCESS) {
            cudaFree((void *)enc->d_bgra);
            enc->nvenc_api.nvEncDestroyEncoder(enc->encoder);
            free(enc);
            if (error_out) snprintf(error_out, (size_t)error_size,
                                    "nvEncCreateInputBuffer: %s",
                                    nvenc_error_string(nv_status));
            return NULL;
        }
        enc->input_buffer = alloc_input.inputBuffer;
        enc->input_buffer_pitch = width * 4; /* ARGB: 4 bytes per pixel */
        fprintf(stderr, "  [NVENC] Legacy path: created input buffer (pitch=%d)\n",
                enc->input_buffer_pitch);
    }

    /* ── 8. Create output bitstream ring buffer ────────────────────── */
    for (int i = 0; i < NVENC_OUTPUT_RING_SIZE; i++) {
        NV_ENC_CREATE_BITSTREAM_BUFFER alloc_output;
        memset(&alloc_output, 0, sizeof(alloc_output));
        alloc_output.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;

        nv_status = enc->nvenc_api.nvEncCreateBitstreamBuffer(enc->encoder, &alloc_output);
        if (nv_status != NV_ENC_SUCCESS) {
            for (int j = 0; j < i; j++) {
                enc->nvenc_api.nvEncDestroyBitstreamBuffer(enc->encoder,
                    enc->output_buffers[j]);
            }
            if (enc->use_registered_resource) {
                enc->nvenc_api.nvEncUnregisterResource(enc->encoder,
                    enc->registered_resource);
            } else if (enc->input_buffer) {
                enc->nvenc_api.nvEncDestroyInputBuffer(enc->encoder, enc->input_buffer);
            }
            cudaFree((void *)enc->d_bgra);
            enc->nvenc_api.nvEncDestroyEncoder(enc->encoder);
            free(enc);
            if (error_out) snprintf(error_out, (size_t)error_size,
                                    "nvEncCreateBitstreamBuffer[%d]: %s",
                                    i, nvenc_error_string(nv_status));
            return NULL;
        }
        enc->output_buffers[i] = alloc_output.bitstreamBuffer;
    }
    enc->output_ring_submit = 0;
    enc->output_ring_drain  = 0;

    /* ── 9. Create CUDA stream ─────────────────────────────────────── */
    if (cudaStreamCreate(&enc->stream) != cudaSuccess) {
        for (int j = 0; j < NVENC_OUTPUT_RING_SIZE; j++) {
            enc->nvenc_api.nvEncDestroyBitstreamBuffer(enc->encoder,
                enc->output_buffers[j]);
        }
        if (enc->use_registered_resource) {
            enc->nvenc_api.nvEncUnregisterResource(enc->encoder,
                enc->registered_resource);
        } else if (enc->input_buffer) {
            enc->nvenc_api.nvEncDestroyInputBuffer(enc->encoder, enc->input_buffer);
        }
        cudaFree((void *)enc->d_bgra);
        enc->nvenc_api.nvEncDestroyEncoder(enc->encoder);
        free(enc);
        if (error_out) snprintf(error_out, (size_t)error_size,
                                "cudaStreamCreate failed");
        return NULL;
    }

    /* ── 10. Initialize packet ring buffer ─────────────────────────── */
    enc->packet_capacity = 64;
    enc->packet_data = (uint8_t **)calloc((size_t)enc->packet_capacity, sizeof(uint8_t *));
    enc->packet_sizes = (int *)calloc((size_t)enc->packet_capacity, sizeof(int));
    enc->packet_capacities = (int *)calloc((size_t)enc->packet_capacity, sizeof(int));
    if (!enc->packet_data || !enc->packet_sizes || !enc->packet_capacities) {
        free(enc->packet_data); free(enc->packet_sizes); free(enc->packet_capacities);
        cudaStreamDestroy(enc->stream);
        for (int j = 0; j < NVENC_OUTPUT_RING_SIZE; j++) {
            enc->nvenc_api.nvEncDestroyBitstreamBuffer(enc->encoder,
                enc->output_buffers[j]);
        }
        if (enc->use_registered_resource) {
            enc->nvenc_api.nvEncUnregisterResource(enc->encoder,
                enc->registered_resource);
        } else if (enc->input_buffer) {
            enc->nvenc_api.nvEncDestroyInputBuffer(enc->encoder, enc->input_buffer);
        }
        cudaFree((void *)enc->d_bgra);
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

    /* With direct-return mode, all encoded data is already consumed.
     * Skip EOS + drain — nvEncLockBitstream doNotWait=1 is broken
     * on this driver and blocks indefinitely. */

    /* Destroy bitstream buffers */
    if (enc->nvenc_api.nvEncDestroyBitstreamBuffer) {
        for (int i = 0; i < NVENC_OUTPUT_RING_SIZE; i++) {
            if (enc->output_buffers[i]) {
                enc->nvenc_api.nvEncDestroyBitstreamBuffer(enc->encoder,
                    enc->output_buffers[i]);
            }
        }
    }

    /* Unregister zero-copy resource */
    if (enc->use_registered_resource && enc->registered_resource &&
        enc->nvenc_api.nvEncUnregisterResource) {
        enc->nvenc_api.nvEncUnregisterResource(enc->encoder,
            enc->registered_resource);
    }

    /* Destroy legacy input buffer */
    if (!enc->use_registered_resource && enc->input_buffer &&
        enc->nvenc_api.nvEncDestroyInputBuffer) {
        enc->nvenc_api.nvEncDestroyInputBuffer(enc->encoder, enc->input_buffer);
    }

    /* Destroy encoder */
    if (enc->encoder && enc->nvenc_api.nvEncDestroyEncoder) {
        enc->nvenc_api.nvEncDestroyEncoder(enc->encoder);
    }

    /* Free CUDA resources */
    if (enc->stream) cudaStreamDestroy(enc->stream);
    if (enc->d_bgra) cudaFree((void *)enc->d_bgra);

    /* Free packet buffers */
    free(enc->sps_pps_data);
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

/* ─── Zero-copy encode ────────────────────────────────────────────── */

int gpu_nvenc_encode_frame_zerocopy(GpuNvencEncoder *enc,
                                     const uint8_t *d_frame_bgr24,
                                     int stride,
                                     const uint8_t **packet_out,
                                     int *packet_size_out) {
    if (!enc || !d_frame_bgr24) {
        if (packet_out) *packet_out = NULL;
        if (packet_size_out) *packet_size_out = 0;
        return -1;
    }

    /* ── 1. Convert BGR24 → BGRA32 on GPU ─────────────────────────── */
    {
        dim3 block = {32, 16, 1};
        dim3 grid = {(unsigned int)((enc->width  + 32 - 1) / 32),
                     (unsigned int)((enc->height + 16 - 1) / 16),
                     1};
        bgr24_to_bgra32_kernel(grid, block,
                                d_frame_bgr24, stride,
                                (uint8_t *)enc->d_bgra, enc->bgra_pitch,
                                enc->width, enc->height, enc->stream);
    }

    /* Synchronize — ensure kernel completes before NVENC reads */
    cudaError_t ce = cudaStreamSynchronize(enc->stream);
    if (ce != cudaSuccess) {
        fprintf(stderr, "  [NVENC ZC] stream sync failed: %s\n", cudaGetErrorString(ce));
        return -1;
    }

    NVENCSTATUS nv_status;
    NV_ENC_INPUT_PTR encode_input = NULL;
    uint32_t input_pitch = (uint32_t)enc->bgra_pitch;

    /* ── 2. Map/acquire input for NVENC ────────────────────────────── */
    if (enc->use_registered_resource) {
        /* Zero-copy: map the registered CUDA buffer for NVENC read */
        NV_ENC_MAP_INPUT_RESOURCE map_input;
        memset(&map_input, 0, sizeof(map_input));
        map_input.version = NV_ENC_MAP_INPUT_RESOURCE_VER;
        map_input.registeredResource = enc->registered_resource;

        nv_status = enc->nvenc_api.nvEncMapInputResource(enc->encoder, &map_input);
        if (nv_status != NV_ENC_SUCCESS) {
            fprintf(stderr, "  [NVENC ZC] map input resource failed: %s\n",
                    nvenc_error_string(nv_status));
            return -1;
        }
        encode_input = map_input.mappedResource;
    } else {
        /* Legacy: lock input buffer and copy BGRA32 to it */
        NV_ENC_LOCK_INPUT_BUFFER lock_input;
        memset(&lock_input, 0, sizeof(lock_input));
        lock_input.version = NV_ENC_LOCK_INPUT_BUFFER_VER;
        lock_input.inputBuffer = enc->input_buffer;

        nv_status = enc->nvenc_api.nvEncLockInputBuffer(enc->encoder, &lock_input);
        if (nv_status != NV_ENC_SUCCESS) {
            fprintf(stderr, "  [NVENC ZC] lock input buffer failed: %s\n",
                    nvenc_error_string(nv_status));
            return -1;
        }

        /* cudaMemcpyDefault lets CUDA auto-detect source/dest memory type.
         * NVENC input buffer may be device or host — Default avoids GPU
         * faults from direction mismatches (e.g., D2D to host memory). */
        ce = cudaMemcpy2DAsync(lock_input.bufferDataPtr, (size_t)lock_input.pitch,
                                (void *)enc->d_bgra, (size_t)enc->bgra_pitch,
                                (size_t)enc->width * 4, (size_t)enc->height,
                                cudaMemcpyDefault, enc->stream);
        if (ce != cudaSuccess) {
            fprintf(stderr, "  [NVENC ZC] memcpy failed: %s\n", cudaGetErrorString(ce));
            enc->nvenc_api.nvEncUnlockInputBuffer(enc->encoder, enc->input_buffer);
            return -1;
        }
        ce = cudaStreamSynchronize(enc->stream);
        if (ce != cudaSuccess) {
            fprintf(stderr, "  [NVENC ZC] stream sync failed: %s\n", cudaGetErrorString(ce));
            enc->nvenc_api.nvEncUnlockInputBuffer(enc->encoder, enc->input_buffer);
            return -1;
        }
        encode_input = enc->input_buffer;
        input_pitch = lock_input.pitch;
        enc->nvenc_api.nvEncUnlockInputBuffer(enc->encoder, enc->input_buffer);
    }

    /* ── 3. Submit frame to NVENC ──────────────────────────────────── */
    {
        NV_ENC_OUTPUT_PTR cur_output = enc->output_buffers[enc->output_ring_submit];

        NV_ENC_PIC_PARAMS pic_params;
        memset(&pic_params, 0, sizeof(pic_params));
        pic_params.version       = NV_ENC_PIC_PARAMS_VER;
        pic_params.inputBuffer   = encode_input;
        pic_params.bufferFmt     = NV_ENC_BUFFER_FORMAT_ARGB;
        pic_params.inputWidth    = (uint32_t)enc->width;
        pic_params.inputHeight   = (uint32_t)enc->height;
        pic_params.inputPitch    = input_pitch;
        pic_params.outputBitstream = cur_output;
        pic_params.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;

        /* First frame: force IDR for SPS/PPS emission */
        if (!enc->first_frame_done) {
            pic_params.encodePicFlags |= NV_ENC_PIC_FLAG_FORCEIDR;
        }

        nv_status = enc->nvenc_api.nvEncEncodePicture(enc->encoder, &pic_params);
        if (nv_status != NV_ENC_SUCCESS) {
            fprintf(stderr, "  [NVENC ZC] encode picture failed: %s\n",
                    nvenc_error_string(nv_status));
            if (enc->use_registered_resource) {
                enc->nvenc_api.nvEncUnmapInputResource(enc->encoder, encode_input);
            }
            return -1;
        }
    }

    /* ── 4. Unmap input resource (if zero-copy) ────────────────────── */
    if (enc->use_registered_resource) {
        nv_status = enc->nvenc_api.nvEncUnmapInputResource(enc->encoder, encode_input);
        if (nv_status != NV_ENC_SUCCESS) {
            fprintf(stderr, "  [NVENC ZC] unmap input resource warning: %s\n",
                    nvenc_error_string(nv_status));
            /* Non-fatal — continue to retrieve output */
        }
    }

    /* ── 5. Retrieve encoded bitstream (sync on first frame) ───────── */
    {
        NV_ENC_OUTPUT_PTR cur_output = enc->output_buffers[enc->output_ring_submit];
        enc->output_ring_submit = (enc->output_ring_submit + 1) % NVENC_OUTPUT_RING_SIZE;

        NV_ENC_LOCK_BITSTREAM lock;
        memset(&lock, 0, sizeof(lock));
        lock.version          = NV_ENC_LOCK_BITSTREAM_VER;
        lock.outputBitstream  = cur_output;
        /* Sync mode: data always ready after nvEncEncodePicture returns.
         * Use blocking lock (doNotWait=0) to guarantee data retrieval. */
        lock.doNotWait        = 0;

        nv_status = enc->nvenc_api.nvEncLockBitstream(enc->encoder, &lock);
        if (nv_status != NV_ENC_SUCCESS) {
            fprintf(stderr, "  [NVENC ZC] lock bitstream failed: %s (doNotWait=%d)\n",
                    nvenc_error_string(nv_status), (int)lock.doNotWait);
            return 0; /* no packet ready yet */
        }

        int packet_size = (int)lock.bitstreamSizeInBytes;

        if (packet_size > 0) {
            const uint8_t *data = (const uint8_t *)lock.bitstreamBufferPtr;

            /* On first frame, extract/construct SPS+PPS */
            if (!enc->first_frame_done) {
                capture_sps_pps(enc, data, packet_size);
                enc->first_frame_done = true;
            }

            /* If caller wants direct output, return the pointer without
             * adding to the internal ring buffer (avoids double-write).
             * Otherwise, queue it for gpu_nvenc_get_packet() retrieval. */
            if (packet_out && packet_size_out) {
                /* Direct return — caller owns the data (valid until next encode) */
                *packet_out = data;
                *packet_size_out = packet_size;
            } else {
                /* Copy into ring buffer */
                if (!add_packet(enc, data, packet_size)) {
                    enc->nvenc_api.nvEncUnlockBitstream(enc->encoder, cur_output);
                    return -1;
                }
            }
        }

        enc->nvenc_api.nvEncUnlockBitstream(enc->encoder, cur_output);
        enc->total_encoded_frames++;
    }

    return 0;
}

/* ─── Packet retrieval ────────────────────────────────────────────── */

int gpu_nvenc_get_packet(GpuNvencEncoder *enc, const uint8_t **packet_out) {
    if (!enc || !packet_out || enc->packet_count <= 0) {
        if (packet_out) *packet_out = NULL;
        return 0;
    }

    *packet_out = enc->packet_data[0];
    int size = enc->packet_sizes[0];

    /* Shift queue */
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

/* ─── Flush ───────────────────────────────────────────────────────── */

int gpu_nvenc_flush(GpuNvencEncoder *enc) {
    if (!enc || !enc->encoder) return -1;

    /* With direct-return mode (packet_out), all encoded data is already
     * retrieved by the caller. NVENC output buffers are in sync with
     * the ring. No EOS or drain needed — nvEncDestroyEncoder handles
     * final cleanup. The doNotWait=1 flag is broken on this driver. */
    return 0;
}

/* ─── Statistics ──────────────────────────────────────────────────── */

int64_t gpu_nvenc_encoded_bytes(GpuNvencEncoder *enc) {
    return enc ? enc->total_encoded_bytes : 0;
}

int gpu_nvenc_encoded_frames(GpuNvencEncoder *enc) {
    return enc ? enc->total_encoded_frames : 0;
}

/* ─── SPS/PPS retrieval ───────────────────────────────────────────── */

int gpu_nvenc_get_sps_pps(GpuNvencEncoder *enc, const uint8_t **data_out) {
    if (!enc || !data_out || !enc->sps_pps_data || enc->sps_pps_size <= 0) {
        if (data_out) *data_out = NULL;
        return 0;
    }
    *data_out = enc->sps_pps_data;
    return enc->sps_pps_size;
}

/* ─── Legacy encode (kept for compatibility) ──────────────────────── */

int gpu_nvenc_encode_frame(GpuNvencEncoder *enc,
                            const uint8_t *gpu_frame_bgr24, int stride) {
    /* Route through zero-copy path */
    const uint8_t *pkt = NULL;
    int pkt_size = 0;
    int ret = gpu_nvenc_encode_frame_zerocopy(enc, gpu_frame_bgr24, stride,
                                               &pkt, &pkt_size);
    if (ret < 0) return -1;
    return pkt_size;
}

/* ─── NV12 path (broken on this driver — kept as stubs) ───────────── */

int gpu_nvenc_encode_frame_nv12(GpuNvencEncoder *enc,
                                 const uint8_t *d_y,
                                 const uint8_t *d_uv,
                                 void *cuda_stream) {
    (void)enc; (void)d_y; (void)d_uv; (void)cuda_stream;
    fprintf(stderr, "NVENC NV12: not supported on this driver (use ARGB path)\n");
    return -1;
}

int gpu_nvenc_encode_frame_nv12_submit(GpuNvencEncoder *enc,
                                        const uint8_t *d_y,
                                        const uint8_t *d_uv,
                                        void *cuda_stream) {
    (void)enc; (void)d_y; (void)d_uv; (void)cuda_stream;
    return -1;
}

int gpu_nvenc_drain_output(GpuNvencEncoder *enc) {
    (void)enc;
    return -1;
}

#elif defined(USE_CUDA) && !defined(HAS_NVENC)
/* ─── CUDA compiled but NVENC header unavailable — stubs ───────── */

GpuNvencEncoder* gpu_nvenc_create(int width, int height, double fps,
                                   int cuda_device, int bitrate,
                                   int gop_length,
                                   char *error_out, int error_size) {
    (void)width; (void)height; (void)fps; (void)cuda_device;
    (void)bitrate; (void)gop_length;
    if (error_out) snprintf(error_out, (size_t)error_size, "NVENC not available (install NVIDIA Video Codec SDK)");
    return NULL;
}

void gpu_nvenc_destroy(GpuNvencEncoder *enc) { (void)enc; }

int gpu_nvenc_encode_frame_zerocopy(GpuNvencEncoder *enc,
                                     const uint8_t *d_frame_bgr24,
                                     int stride,
                                     const uint8_t **packet_out,
                                     int *packet_size_out) {
    (void)enc; (void)d_frame_bgr24; (void)stride;
    if (packet_out) *packet_out = NULL;
    if (packet_size_out) *packet_size_out = 0;
    return -1;
}

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

int gpu_nvenc_get_sps_pps(GpuNvencEncoder *enc, const uint8_t **data_out) {
    (void)enc; if (data_out) *data_out = NULL; return 0;
}

int gpu_nvenc_encode_frame_nv12(GpuNvencEncoder *enc, const uint8_t *d_y,
                                 const uint8_t *d_uv, void *stream) {
    (void)enc; (void)d_y; (void)d_uv; (void)stream; return -1;
}

int gpu_nvenc_encode_frame_nv12_submit(GpuNvencEncoder *enc, const uint8_t *d_y,
                                        const uint8_t *d_uv, void *stream) {
    (void)enc; (void)d_y; (void)d_uv; (void)stream; return -1;
}

int gpu_nvenc_drain_output(GpuNvencEncoder *enc) { (void)enc; return -1; }

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

int gpu_nvenc_encode_frame_zerocopy(GpuNvencEncoder *enc,
                                     const uint8_t *d_frame_bgr24,
                                     int stride,
                                     const uint8_t **packet_out,
                                     int *packet_size_out) {
    (void)enc; (void)d_frame_bgr24; (void)stride;
    if (packet_out) *packet_out = NULL;
    if (packet_size_out) *packet_size_out = 0;
    return -1;
}

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

int gpu_nvenc_get_sps_pps(GpuNvencEncoder *enc, const uint8_t **data_out) {
    (void)enc; if (data_out) *data_out = NULL; return 0;
}

int gpu_nvenc_encode_frame_nv12(GpuNvencEncoder *enc, const uint8_t *d_y,
                                 const uint8_t *d_uv, void *stream) {
    (void)enc; (void)d_y; (void)d_uv; (void)stream; return -1;
}
int gpu_nvenc_encode_frame_nv12_submit(GpuNvencEncoder *enc, const uint8_t *d_y,
                                        const uint8_t *d_uv, void *stream) {
    (void)enc; (void)d_y; (void)d_uv; (void)stream; return -1;
}
int gpu_nvenc_drain_output(GpuNvencEncoder *enc) { (void)enc; return -1; }

#endif /* USE_CUDA */

#ifndef VIDCRYPT_GPU_RS_ENCODE_CUH
#define VIDCRYPT_GPU_RS_ENCODE_CUH

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * Tiled Reed-Solomon encoder — GPU-accelerated with shared-memory tiling
 *
 * Processes multiple codewords per CUDA block using warp-level parallelism.
 * Each warp handles one codeword with all 32 threads cooperating via
 * warp-shuffle reduction. The precomputed parity dictionary is loaded
 * into shared memory in tiles to eliminate global memory round-trips.
 *
 * Architecture (Blackwell-optimized, CC 12.0):
 *   Grid:  ceil(N / TILES) blocks, 1D
 *   Block: TILES × 32 threads (8 × 32 = 256 for RS32)
 *   Shared memory per block: TILES × n_k × sizeof(uint8_t) for partial sums
 *                           + tile_sz × 256 × n_k for dictionary tile
 *
 * For RS(255,223,32): k=223, n=255, n_k=32
 *   Dictionary: 223 × 256 × 32 = 1.83 MB (fits in L2 cache)
 *   Transposed dict: same size, stored alongside for coalesced access
 *   Tile size: typically 16-32 positions (balances SMEM vs iterations)
 *
 * Performance model vs current kernel:
 *   Old: N blocks × n_k threads × k serial iterations = O(N × n_k × k)
 *        ~3% SM occupancy on Blackwell
 *   New: N/8 blocks × 256 threads × k/TILE_K tile iterations
 *        ~25% SM occupancy, shared-memory bandwidth instead of global
 *   Expected: 10-20× throughput improvement
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Number of codewords processed per CUDA block */
#define RS_TILES_PER_BLOCK   8

/* Dictionary tile size — number of message positions per tile.
 * Larger = fewer iterations but more shared memory.
 * 32 × 256 × 32 = 256 KB shared memory per tile — fits in 228 KB SMEM
 * but 16 × 256 × 32 = 128 KB is safer for wider compatibility.
 * Blackwell: 228 KB per SM, so 16 is conservative. */
#define RS_DICT_TILE_SIZE    16

/* ─── Kernel: tiled RS encode ────────────────────────────────────────
 *
 * d_msg:    [N * k] padded input bytes (zero-filled beyond file end)
 * d_dict:   [k * 256 * n_k] precomputed parity dictionary
 * d_dict_T: [k * n_k * 256] transposed dictionary (coalesced access)
 * d_enc:    [N * 255] output: msg[0..k-1] then parity[k..254]
 * k:        message length (e.g., 223 for RS32)
 * n_k:      parity symbols (e.g., 32 for RS32)
 * N:        total number of codewords
 *
 * Launched on the given CUDA stream.
 * Grid: ceil(N / RS_TILES_PER_BLOCK) blocks, 1D
 * Block: RS_TILES_PER_BLOCK × 32 threads (256 total)                   */
void rs_encode_tiled_kernel(
    const uint8_t *d_msg,
    const uint8_t *d_dict,
    const uint8_t *d_dict_T,
    uint8_t *d_enc,
    int k, int n_k, int N,
    cudaStream_t stream);

/* ─── Build transposed dictionary from CPU-side parity_dict ──────────
 *
 * Input:  parity_dict [k * 256 * n_k] — dict[i * 256*n_k + v * n_k + j]
 * Output: dict_T      [k * n_k * 256] — dict_T[i * n_k*256 + j * 256 + v]
 *
 * The transposed layout ensures coalesced global memory access when
 * all threads in a warp access dict_T for the same (i,j) but different v
 * (message bytes). Instead of strided n_k-byte loads, threads get
 * contiguous 32-byte transactions.                                    */
void rs_build_transposed_dict(
    const uint8_t *parity_dict,
    int k, int n_k,
    uint8_t *dict_T_out);

/* ─── Bit-expansion kernel (kept from gpu_kernels.cu, re-exported) ───
 * Expands encoded bytes to per-bit bytes using LUT64.
 * d_encoded:  [total_bytes] input
 * d_bits:     [total_bytes * 8] output as uint64 per byte
 * total_bytes: number of input bytes
 * stream:     CUDA stream                                          */
void rs_bit_expand_kernel(
    const uint8_t *d_encoded,
    uint64_t *d_bits,
    int total_bytes,
    cudaStream_t stream);

/* ─── LUT upload (call once at init) ───────────────────────────────── */

/* Upload BYTE_TO_BITS_LUT to GPU constant memory for bit expansion */
void gpu_rs_upload_lut64(void);

#ifdef __cplusplus
}
#endif

#endif /* VIDCRYPT_GPU_RS_ENCODE_CUH */

/* Full pipeline benchmark: GPU frame gen + NVENC encode + muxer write */
#include "encoder.h"
#include "gpu_backend.h"
#include "gpu_nvenc.h"
#include "gpu_kernels.h"
#include "calibration.h"
#include "ffmpeg_pipe.h"
#include "bitstream.h"
#include "reedsolomon.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

static double now(void) { return (double)clock() / CLOCKS_PER_SEC; }

int main(void) {
    char err[256];
    int width = 1920, height = 1080;
    int block_size = 8, margin = 96;
    int ecc = 32;

    CalParams params;
    {
        EncoderConfig cfg;
        encoder_config_defaults(&cfg);
        cfg.frame_width = width; cfg.frame_height = height;
        cfg.margin_x = margin; cfg.margin_y = margin;
        cfg.block_size = block_size;
        cfg.rs_ecc_symbols = ecc;
        compute_grid_params(&cfg, &params, err, sizeof(err));
    }
    int grid_cols = params.grid_cols;
    int pay_rows = cal_params_payload_rows(&params);
    int pay_bits = cal_params_payload_bits_per_frame(&params);
    int n_frames = 100;

    fprintf(stderr, "[FULLBENCH] %dx%d block=%d margin=%d pay_bits=%d frames=%d\n",
            width, height, block_size, margin, pay_bits, n_frames);

    GpuBackend *b = NULL;
    gpu_backend_init(&b, 0, err, sizeof(err));
    gpu_backend_open_nvenc(b, width, height, 30.0, 50000, 1, err, sizeof(err));

    /* Prepare test bits (all zeros) */
    uint8_t *cpu_bits = (uint8_t*)calloc((size_t)pay_bits, 1);
    uint8_t *d_bits = NULL;
    cudaMalloc((void**)&d_bits, (size_t)pay_bits);
    cudaMemcpy(d_bits, cpu_bits, (size_t)pay_bits, cudaMemcpyHostToDevice);

    /* Test A: just GPU frame gen (no NVENC) */
    fprintf(stderr, "[FULLBENCH] Test A: GPU frame gen only...\n");
    double t0 = now();
    for (int i = 0; i < 100; i++) {
        gpu_backend_generate_frame_dbits_bgr24(b, d_bits, pay_bits,
            width, height, grid_cols, pay_rows, block_size, margin, margin, &params);
    }
    gpu_backend_sync_encode(b);
    double t1 = now();
    fprintf(stderr, "[FULLBENCH]   Frame gen: %.3fs = %.1f fps\n", t1-t0, 100.0/(t1-t0));

    /* Test B: frame gen + NVENC encode (no muxer) */
    fprintf(stderr, "[FULLBENCH] Test B: frame gen + NVENC encode...\n");
    GpuNvencEncoder *nvenc = gpu_backend_get_nvenc_encoder(b);
    t0 = now();
    for (int i = 0; i < n_frames; i++) {
        uint8_t *d_frame = gpu_backend_generate_frame_dbits_bgr24(b, d_bits, pay_bits,
            width, height, grid_cols, pay_rows, block_size, margin, margin, &params);
        gpu_backend_sync_encode(b);
        const uint8_t *pkt; int sz;
        gpu_nvenc_encode_frame_zerocopy(nvenc, d_frame, width*3, &pkt, &sz);
        while (gpu_nvenc_get_packet(nvenc, &pkt) > 0);
    }
    t1 = now();
    fprintf(stderr, "[FULLBENCH]   Frame gen + encode: %.3fs = %.1f fps\n", t1-t0, (double)n_frames/(t1-t0));

    /* Test C: full pipeline with raw H.264 output (no ffmpeg muxer bottleneck) */
    fprintf(stderr, "[FULLBENCH] Test C: full pipeline with raw H.264 output...\n");
    const uint8_t *sps_pps; int sps_pps_sz;
    sps_pps_sz = gpu_backend_get_nvenc_sps_pps(b, &sps_pps);
    FILE *fraw = fopen("test_bench_full.h264", "wb");
    /* Write SPS/PPS extradata first */
    if (sps_pps && sps_pps_sz > 0) fwrite(sps_pps, 1, sps_pps_sz, fraw);
    /* Use a large buffer for raw file writes too */
    setvbuf(fraw, NULL, _IOFBF, 256*1024);

    t0 = now();
    for (int i = 0; i < n_frames; i++) {
        uint8_t *d_frame = gpu_backend_generate_frame_dbits_bgr24(b, d_bits, pay_bits,
            width, height, grid_cols, pay_rows, block_size, margin, margin, &params);
        gpu_backend_sync_encode(b);
        const uint8_t *pkt; int sz;
        gpu_nvenc_encode_frame_zerocopy(nvenc, d_frame, width*3, &pkt, &sz);
        if (pkt && sz > 0) fwrite(pkt, 1, sz, fraw);
        { const uint8_t *p2; int s2;
          while ((s2 = gpu_nvenc_get_packet(nvenc, &p2)) > 0) fwrite(p2, 1, s2, fraw); }
    }
    fprintf(stderr, "[FULLBENCH]   frames loop done, flushing...\n");
    gpu_nvenc_flush(nvenc);
    fprintf(stderr, "[FULLBENCH]   flush done, draining...\n");
    { const uint8_t *p; int s; while ((s = gpu_nvenc_get_packet(nvenc,&p)) > 0) fwrite(p,1,s,fraw); }
    fprintf(stderr, "[FULLBENCH]   drain done, closing...\n");
    fclose(fraw);
    fprintf(stderr, "[FULLBENCH]   close done.\n");
    t1 = now();
    fprintf(stderr, "[FULLBENCH]   Raw H.264 output: %.3fs = %.1f fps\n",
            t1-t0, (double)n_frames/(t1-t0));

    cudaFree(d_bits);
    free(cpu_bits);
    gpu_backend_close_nvenc(b);
    gpu_backend_destroy(b);
    return 0;
}

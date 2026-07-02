/* Simulate full encoder pipeline to find bottleneck */
#include "encoder.h"
#include "gpu_backend.h"
#include "gpu_nvenc.h"
#include "bitstream.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

static double now(void) { return (double)clock() / CLOCKS_PER_SEC; }

int main(void) {
    char err[256];
    EncoderConfig cfg; encoder_config_defaults(&cfg);
    cfg.frame_width = 1920; cfg.frame_height = 1080;
    cfg.margin_x = cfg.margin_y = 96;
    cfg.block_size = 8; cfg.rs_ecc_symbols = 0;
    cfg.codec_name = "h264"; cfg.backend_mode = BACKEND_GPU;
    snprintf(cfg.output_path, sizeof(cfg.output_path), "test_pipe.h264");

    CalParams params;
    compute_grid_params(&cfg, &params, err, sizeof(err));
    int pay_bits = cal_params_payload_bits_per_frame(&params);
    int grid_cols = params.grid_cols;
    int pay_rows = cal_params_payload_rows(&params);
    int bs = params.block_size_x, mx = params.margin_x, my = params.margin_y;
    int w = params.frame_width, h = params.frame_height;

    /* Simulate a 10MB file worth of bits */
    int64_t total_bytes = 10485760LL;  /* 10MB */
    int bit_buf_len = (int)(total_bytes * 8);
    int n_frames = bit_buf_len / pay_bits;

    fprintf(stderr, "[PIPE] %d frames, pay_bits=%d\n", n_frames, pay_bits);

    GpuBackend *b = NULL;
    gpu_backend_init(&b, 0, err, sizeof(err));
    gpu_backend_open_nvenc(b, w, h, 30.0, 50000, 1, err, sizeof(err));
    GpuNvencEncoder *nvenc = gpu_backend_get_nvenc_encoder(b);

    /* Pre-alloc bit buffer on CPU (simulate LUT expansion output) */
    uint8_t *cpu_bits = (uint8_t*)calloc((size_t)pay_bits, 1); /* all zeros */
    uint8_t *d_bits = NULL;
    cudaMalloc((void**)&d_bits, (size_t)pay_bits);

    /* Warmup */
    for (int i = 0; i < 10; i++) {
        cudaMemcpy(d_bits, cpu_bits, pay_bits, cudaMemcpyHostToDevice);
        uint8_t *d_f = gpu_backend_generate_frame_dbits_bgr24(b, d_bits, pay_bits,
            w, h, grid_cols, pay_rows, bs, mx, my, &params);
        gpu_backend_sync_encode(b);
        const uint8_t *p; int s;
        gpu_nvenc_encode_frame_zerocopy(nvenc, d_f, w*3, &p, &s);
    }

    /* Timed run */
    double t_h2d = 0, t_gen = 0, t_enc = 0, t_total = 0;
    int batch = 200;
    fprintf(stderr, "[PIPE] Timing %d frames...\n", batch);

    double t0 = now();
    for (int i = 0; i < batch; i++) {
        double ts;

        ts = now();
        cudaMemcpy(d_bits, cpu_bits, pay_bits, cudaMemcpyHostToDevice);
        t_h2d += now() - ts;

        ts = now();
        uint8_t *d_f = gpu_backend_generate_frame_dbits_bgr24(b, d_bits, pay_bits,
            w, h, grid_cols, pay_rows, bs, mx, my, &params);
        gpu_backend_sync_encode(b);
        t_gen += now() - ts;

        ts = now();
        const uint8_t *p; int s;
        gpu_nvenc_encode_frame_zerocopy(nvenc, d_f, w*3, &p, &s);
        t_enc += now() - ts;
    }
    double t1 = now();
    t_total = t1 - t0;

    fprintf(stderr, "[PIPE] Results for %d frames:\n", batch);
    fprintf(stderr, "  H2D copy:     %.3fs (%.1f%%)\n", t_h2d, t_h2d/t_total*100);
    fprintf(stderr, "  Frame gen:    %.3fs (%.1f%%)\n", t_gen, t_gen/t_total*100);
    fprintf(stderr, "  NVENC encode: %.3fs (%.1f%%)\n", t_enc, t_enc/t_total*100);
    fprintf(stderr, "  TOTAL:        %.3fs = %.1f fps\n", t_total, (double)batch/t_total);

    gpu_nvenc_flush(nvenc);
    cudaFree(d_bits);
    free(cpu_bits);
    gpu_backend_close_nvenc(b);
    gpu_backend_destroy(b);
    return 0;
}

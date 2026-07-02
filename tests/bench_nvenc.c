/* Minimal NVENC benchmark — isolates each step to find the hang */
#include "gpu_backend.h"
#include "gpu_nvenc.h"
#include "gpu_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

int main(void) {
    int width = 1920, height = 1080;
    char err[256];

    fprintf(stderr, "[BENCH] Init GPU...\n");
    GpuBackend *backend = NULL;
    if (gpu_backend_init(&backend, 0, err, sizeof(err)) != GPU_INIT_OK) {
        fprintf(stderr, "GPU init: %s\n", err); return 1;
    }

    fprintf(stderr, "[BENCH] Opening NVENC...\n");
    if (!gpu_backend_open_nvenc(backend, width, height, 30.0,
                                 50000, 1, err, sizeof(err))) {
        fprintf(stderr, "NVENC open: %s\n", err);
        gpu_backend_destroy(backend); return 1;
    }

    GpuNvencEncoder *nvenc = gpu_backend_get_nvenc_encoder(backend);

    /* Test frame */
    size_t bgr_size = (size_t)width * (size_t)height * 3;
    uint8_t *d_frame = NULL;
    cudaMalloc((void **)&d_frame, bgr_size);
    cudaMemset(d_frame, 128, bgr_size);

    /* Test 1: just run BGR24->BGRA32 kernel */
    fprintf(stderr, "[BENCH] Test 1: BGR24->BGRA32 kernel...\n");
    {
        uint8_t *d_bgra_test = NULL;
        cudaMalloc((void **)&d_bgra_test, (size_t)width * height * 4);
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        dim3 block = {32, 16, 1};
        dim3 grid = {(unsigned)((width+31)/32), (unsigned)((height+15)/16), 1};
        bgr24_to_bgra32_kernel(grid, block, d_frame, width*3,
                                d_bgra_test, width*4, width, height, stream);
        cudaStreamSynchronize(stream);
        fprintf(stderr, "[BENCH] Test 1: OK\n");
        cudaFree(d_bgra_test);
        cudaStreamDestroy(stream);
    }

    /* Test 2: encode one frame with timeout guard */
    fprintf(stderr, "[BENCH] Test 2: encode 1 frame...\n");
    {
        const uint8_t *pkt; int sz;
        int ret = gpu_nvenc_encode_frame_zerocopy(nvenc, d_frame, width*3, &pkt, &sz);
        fprintf(stderr, "[BENCH] Test 2: ret=%d sz=%d\n", ret, sz);
    }

    /* Test 3: encode 3 more frames */
    fprintf(stderr, "[BENCH] Test 3: encode 3 more...\n");
    for (int i = 0; i < 3; i++) {
        fprintf(stderr, "[BENCH]   frame %d...\n", i);
        const uint8_t *pkt; int sz;
        int ret = gpu_nvenc_encode_frame_zerocopy(nvenc, d_frame, width*3, &pkt, &sz);
        fprintf(stderr, "[BENCH]   frame %d: ret=%d sz=%d\n", i, ret, sz);
    }

    /* Test 4: full throughput */
    fprintf(stderr, "[BENCH] Test 4: throughput (100 frames)...\n");
    int n = 100;
    double t0 = (double)clock() / CLOCKS_PER_SEC;
    for (int i = 0; i < n; i++) {
        const uint8_t *pkt; int sz;
        if (gpu_nvenc_encode_frame_zerocopy(nvenc, d_frame, width*3, &pkt, &sz) < 0) {
            fprintf(stderr, "[BENCH] Frame %d FAILED\n", i);
            break;
        }
        if (i == 50) {
            double t1 = (double)clock() / CLOCKS_PER_SEC;
            fprintf(stderr, "[BENCH]   50 frames in %.3fs = %.1f fps\n", t1-t0, 50.0/(t1-t0));
        }
    }
    double t1 = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stderr, "[BENCH]   %d frames in %.3fs = %.1f fps\n", n, t1-t0, (double)n/(t1-t0));

    gpu_nvenc_flush(nvenc);

    fprintf(stderr, "[BENCH] Cleanup...\n");
    cudaFree(d_frame);
    gpu_backend_close_nvenc(backend);
    gpu_backend_destroy(backend);
    fprintf(stderr, "[BENCH] Done.\n");
    return 0;
}

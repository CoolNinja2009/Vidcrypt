/* Dump NVENC bitstream to file for analysis */
#include "gpu_backend.h"
#include "gpu_nvenc.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int main(void) {
    char err[256];
    GpuBackend *b = NULL;
    gpu_backend_init(&b, 0, err, sizeof(err));
    gpu_backend_open_nvenc(b, 1920, 1080, 30.0, 50000, 1, err, sizeof(err));
    GpuNvencEncoder *nvenc = gpu_backend_get_nvenc_encoder(b);

    uint8_t *d_frame = NULL;
    cudaMalloc((void**)&d_frame, 1920ULL*1080*3);
    cudaMemset(d_frame, 128, 1920ULL*1080*3);

    /* Encode first 3 frames, dump ALL bitstream data to file */
    FILE *f = fopen("nv_bs_dump.bin", "wb");
    for (int i = 0; i < 3; i++) {
        const uint8_t *pkt; int sz;
        if (gpu_nvenc_encode_frame_zerocopy(nvenc, d_frame, 1920*3, &pkt, &sz) == 0 && sz > 0) {
            fprintf(stderr, "Frame %d: %d bytes\n", i, sz);
            /* Print first 32 bytes as hex */
            fprintf(stderr, "  hex: ");
            for (int j = 0; j < (sz < 32 ? sz : 32); j++)
                fprintf(stderr, "%02X ", pkt[j]);
            fprintf(stderr, "\n");
            fwrite(pkt, 1, sz, f);
        }
    }
    fclose(f);
    fprintf(stderr, "Dumped to nv_bs_dump.bin\n");

    cudaFree(d_frame);
    gpu_backend_close_nvenc(b);
    gpu_backend_destroy(b);
    return 0;
}

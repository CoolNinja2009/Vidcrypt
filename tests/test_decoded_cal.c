/* Test: extract_calibration from decoded raw frame */
#include "calibration.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <raw_frame_1920x1080_gray>\n", argv[0]);
        return 1;
    }

    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("fopen"); return 1; }

    uint8_t *frame = (uint8_t *)malloc(1920 * 1080);
    if (!frame) { printf("OOM\n"); return 1; }
    size_t n = fread(frame, 1, 1920 * 1080, f);
    fclose(f);
    if (n != 1920 * 1080) { printf("Short read: %zu\n", n); return 1; }

    printf("Frame loaded: %zu bytes\n", n);

    /* Check calibration region manually */
    uint8_t cal_bits[192];
    int nbits = read_calibration_dots(frame, 1920, 1080, cal_bits);
    printf("read_calibration_dots: %d bits\n", nbits);

    /* Show first 24 bits as hex */
    printf("First 24 bits: ");
    for (int i = 0; i < 24; i++) printf("%d", cal_bits[i]);
    printf("\n");

    /* Try interpret */
    CalParams params;
    memset(&params, 0, sizeof(params));
    bool ok = interpret_calibration_bits(cal_bits, &params);
    printf("interpret_calibration_bits: %s\n", ok ? "OK" : "FAIL");

    /* Try full extract (with error correction) */
    CalParams params2;
    memset(&params2, 0, sizeof(params2));
    bool ok2 = extract_calibration(frame, 1920, 1080, &params2);
    printf("extract_calibration: %s\n", ok2 ? "OK" : "FAIL");
    if (ok2) {
        printf("  grid_cols=%d block=%d margin=%d,%d rows=%d\n",
               params2.grid_cols, params2.block_size_x,
               params2.margin_x, params2.margin_y,
               params2.grid_rows);
        printf("  rs_ecc=%d rs_data=%d header_ver=%d sync=%d\n",
               params2.rs_ecc_symbols, params2.rs_data_bytes,
               params2.header_version, params2.sync_rows);
    } else {
        printf("  Legacy fallback: grid_cols=%d block=%d\n",
               params2.grid_cols, params2.block_size_x);
    }

    free(frame);
    return 0;
}

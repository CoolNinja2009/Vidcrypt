/* Quick test: does write_calibration_dots → read_calibration_dots roundtrip? */
#include "calibration.h"
#include "framegen.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    CalParams params;
    memset(&params, 0, sizeof(params));
    params.frame_width  = 1920;
    params.frame_height = 1080;
    params.margin_x     = 8;
    params.margin_y     = 8;
    params.block_size_x = 8;
    params.block_size_y = 8;
    params.grid_cols    = 238;
    params.grid_rows    = 118;
    params.rs_ecc_symbols = 32;
    params.rs_data_bytes  = 223;
    params.header_version = 3;
    params.calibration_rows = 4;
    params.sync_rows = 1;

    /* Generate a frame */
    PrecomputedFrame pf;
    if (!precomputed_frame_init(&pf, &params)) {
        printf("FAIL: precomputed_frame_init\n");
        return 1;
    }

    /* Check calibration region directly */
    int cal_top    = (int)(1080.0f * 0.02f);
    int cal_height = (int)(1080.0f * 0.04f);
    int cal_left   = (int)(1920.0f * 0.04f);
    int cal_width  = (int)(1920.0f * 0.92f);

    int non_zero = 0;
    for (int y = cal_top; y < cal_top + cal_height; y++)
        for (int x = cal_left; x < cal_left + cal_width; x++)
            if (pf.work[y * 1920 + x] != 0) non_zero++;

    printf("Calibration region: %d non-zero pixels out of %d (%d x %d)\n",
           non_zero, cal_height * cal_width, cal_width, cal_height);

    /* Try reading calibration back */
    uint8_t cal_bits[192];
    int n = read_calibration_dots(pf.work, 1920, 1080, cal_bits);
    printf("read_calibration_dots returned: %d bits\n", n);

    /* Try interpreting */
    CalParams read_params;
    bool ok = interpret_calibration_bits(cal_bits, &read_params);
    printf("interpret_calibration_bits: %s\n", ok ? "OK" : "FAIL");

    if (ok) {
        printf("  grid_cols=%d block_size=%d margin=%d margin_y=%d\n",
               read_params.grid_cols, read_params.block_size_x,
               read_params.margin_x, read_params.margin_y);
    }

    /* Try single-bit correction */
    int ok2;
    ok2 = extract_calibration(pf.work, 1920, 1080, &read_params);
    printf("extract_calibration: %s\n", ok2 ? "OK" : "FAIL");
    if (ok2) {
        printf("  grid_cols=%d block_size=%d\n",
               read_params.grid_cols, read_params.block_size_x);
    }

    precomputed_frame_destroy(&pf);
    return 0;
}

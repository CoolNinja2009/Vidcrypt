#include "framegen.h"
#include "simd_decode.h"
#include <stdlib.h>
#include <string.h>

bool precomputed_frame_init(PrecomputedFrame *pf, const CalParams *params) {
    memset(pf, 0, sizeof(PrecomputedFrame));
    pf->params = *params;
    pf->width  = (int)params->frame_width;
    pf->height = (int)params->frame_height;
    pf->stride = pf->width;              /* gray8: 1 byte per pixel */
    pf->block_size = (int)params->block_size_y;

    int grid_top_y = cal_params_grid_top_y(params);
    pf->pay_y_start = grid_top_y + (int)params->sync_rows * pf->block_size;
    pf->pay_y_end   = pf->pay_y_start + cal_params_payload_rows(params) * pf->block_size;
    pf->pay_x_start = (int)params->margin_x;
    pf->pay_x_end   = pf->pay_x_start + (int)params->grid_cols * pf->block_size;
    pf->pay_rows    = cal_params_payload_rows(params);
    pf->grid_cols   = (int)params->grid_cols;

    /* Allocate working frame buffer (gray8: width * height bytes) */
    size_t frame_size = (size_t)pf->height * (size_t)pf->stride;
    pf->work = (uint8_t *)calloc(1, frame_size);
    if (!pf->work) return false;

    /* Pre-render calibration bar */
    write_calibration_dots(pf->work, pf->width, pf->height, params);

    /* Pre-render sync row (alternating black/white tiles) */
    for (int col = 0; col < (int)params->grid_cols; ++col) {
        uint8_t val = (col % 2) ? 255 : 0;
        int x = (int)params->margin_x + col * pf->block_size;
        for (int yy = grid_top_y; yy < grid_top_y + pf->block_size; ++yy)
            memset(pf->work + yy * pf->stride + x, val, (size_t)pf->block_size);
    }

    return true;
}

void precomputed_frame_destroy(PrecomputedFrame *pf) {
    free(pf->work);
    pf->work = NULL;
}

uint8_t *precomputed_frame_generate(PrecomputedFrame *pf, const uint8_t *bits) {
    /* Zero out payload region */
    for (int y = pf->pay_y_start; y < pf->pay_y_end; ++y)
        memset(pf->work + (size_t)y * (size_t)pf->stride + (size_t)pf->pay_x_start,
               0, (size_t)(pf->pay_x_end - pf->pay_x_start));

    /* Expand bits into payload region */
    tile_expand_bits(bits, pf->work, pf->stride,
                     pf->pay_y_start, pf->pay_x_start,
                     pf->block_size,
                     pf->grid_cols, pf->pay_rows);

    return pf->work;
}

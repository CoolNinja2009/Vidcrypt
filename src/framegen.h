#ifndef VIDCRYPT_FRAMEGEN_H
#define VIDCRYPT_FRAMEGEN_H

#include <stdint.h>
#include "calibration.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Precomputed frame template with calibration bar + sync row pre-rendered.
 * Payload blocks are inserted via tile expansion. */

typedef struct {
    uint8_t *work;       /* grayscale frame buffer (width * height bytes) */
    int width;
    int height;
    int stride;          /* = width (1 byte per pixel) */

    /* Payload region */
    int pay_y_start;
    int pay_y_end;
    int pay_x_start;
    int pay_x_end;
    int pay_rows;
    int grid_cols;
    int block_size;

    CalParams params;
} PrecomputedFrame;

bool precomputed_frame_init(PrecomputedFrame *pf, const CalParams *params);
void precomputed_frame_destroy(PrecomputedFrame *pf);

/* Generate a frame from packed bits (pay_rows * grid_cols bytes, 0 or 1 per cell).
 * Returns pointer to pf->work (grayscale, stride = width). */
uint8_t *precomputed_frame_generate(PrecomputedFrame *pf, const uint8_t *bits);

#ifdef __cplusplus
}
#endif

#endif /* VIDCRYPT_FRAMEGEN_H */

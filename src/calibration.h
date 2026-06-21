#ifndef VIDCRYPT_CALIBRATION_H
#define VIDCRYPT_CALIBRATION_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CAL_TOP_FRAC    0.02f
#define CAL_HEIGHT_FRAC 0.04f
#define CAL_LEFT_FRAC   0.04f
#define CAL_WIDTH_FRAC  0.92f

#define CAL_COLS        48
#define CAL_ROWS        4
#define CAL_DATA_BYTES  24
#define CAL_BITS_TOTAL  (CAL_COLS * CAL_ROWS)

#define CAL_MAGIC_V1    0xCB01u
#define CAL_MAGIC_V2    0xCB02u

typedef struct {
    uint16_t frame_width;
    uint16_t frame_height;
    uint16_t margin_x;
    uint16_t margin_y;
    uint16_t block_size_x;
    uint16_t block_size_y;
    uint16_t grid_cols;
    uint16_t grid_rows;
    uint8_t  rs_ecc_symbols;
    uint8_t  rs_data_bytes;
    uint8_t  header_version;
    uint8_t  calibration_rows;
    uint8_t  sync_rows;
} CalParams;

static inline int cal_params_cal_top(const CalParams *p) {
    return (int)(p->frame_height * CAL_TOP_FRAC);
}
static inline int cal_params_cal_bottom(const CalParams *p) {
    return (int)(p->frame_height * (CAL_TOP_FRAC + CAL_HEIGHT_FRAC));
}
static inline int cal_params_grid_top_y(const CalParams *p) {
    return cal_params_cal_bottom(p) + (int)p->margin_y;
}
static inline int cal_params_legacy_grid_top_y(const CalParams *p) {
    return (int)p->margin_y;
}
static inline int cal_params_payload_rows(const CalParams *p) {
    return (int)(p->grid_rows - p->sync_rows);
}
static inline int cal_params_payload_bits_per_frame(const CalParams *p) {
    return cal_params_payload_rows(p) * (int)p->grid_cols;
}

#define LEGACY_FRAME_WIDTH      1920u
#define LEGACY_FRAME_HEIGHT     1080u
#define LEGACY_MARGIN_X         96u
#define LEGACY_MARGIN_Y         72u
#define LEGACY_BLOCK_SIZE       24u
#define LEGACY_GRID_COLS        72u
#define LEGACY_GRID_ROWS        39u
#define LEGACY_RS_ECC_SYMBOLS   32u
#define LEGACY_RS_DATA_BYTES    223u
#define LEGACY_HEADER_VERSION   2u

#define DEFAULT_FRAME_WIDTH     1920u
#define DEFAULT_FRAME_HEIGHT    1080u
#define DEFAULT_MARGIN_X        96u
#define DEFAULT_MARGIN_Y        72u
#define DEFAULT_BLOCK_SIZE      8u
#define DEFAULT_RS_ECC_SYMBOLS  32u
#define DEFAULT_RS_DATA_BYTES   223u

#define THRESHOLD               128

void build_calibration_bytes(const CalParams *params, uint8_t out[24]);
bool parse_calibration_bytes(const uint8_t data[24], CalParams *params, bool *is_v2);
void cal_params_legacy_defaults(CalParams *params);
void write_calibration_dots(uint8_t *frame, int width, int height, const CalParams *params);
int read_calibration_dots(const uint8_t *gray, int width, int height, uint8_t *bits);
bool extract_calibration(const uint8_t *gray, int width, int height, CalParams *params);
bool interpret_calibration_bits(const uint8_t bits[192], CalParams *params);

#ifdef __cplusplus
}
#endif

#endif /* VIDCRYPT_CALIBRATION_H */

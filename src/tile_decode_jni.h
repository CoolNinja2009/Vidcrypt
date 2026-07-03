#ifndef VIDCRYPT_TILE_DECODE_JNI_H
#define VIDCRYPT_TILE_DECODE_JNI_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize the JVM for tile decode acceleration.
 * Must be called once before tile_decode_grid_jni().
 * Returns true if JVM loaded successfully, false if Java unavailable.
 * classpath: path to vidcrypt-java/out (compiled Java classes). */
bool tile_decode_jni_init(const char *classpath);

/* Shut down the JVM. Call at program exit. */
void tile_decode_jni_shutdown(void);

/* Check if JNI tile decode is available. */
bool tile_decode_jni_available(void);

/* Decode tiles using Java TileDecoder via JNI.
 * Signature matches tile_decode_grid() exactly.
 * Falls back to C internally if JNI unavailable.
 * c_fallback: the C tile_decode_grid function pointer. */
int tile_decode_grid_jni(
    const uint8_t *gray, int stride,
    int grid_top_y, int grid_left_x, int block_size,
    int grid_cols, int grid_rows, int sync_rows,
    uint8_t *bits_out, int max_bits,
    bool *sync_ok);

#ifdef __cplusplus
}
#endif

#endif /* VIDCRYPT_TILE_DECODE_JNI_H */

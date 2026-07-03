/* JNI/Native bridge for tile decode acceleration.
 * Three tiers (first available wins):
 *   1. USE_NATIVE_TILES — GraalVM native-image shared lib (no runtime Java)
 *   2. USE_JNI_TILES — JVM at runtime, loads JAR
 *   3. C SIMD fallback — always available
 */
#include "tile_decode_jni.h"
#include "simd_decode.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── Tier 1: Native library (GraalVM) ─────────────────────────── */
#ifdef USE_NATIVE_TILES
/* Declared in the native-image generated header */
extern int tile_decode_grid_native(
    const uint8_t *gray, int stride,
    int grid_top_y, int grid_left_x, int block_size,
    int grid_cols, int grid_rows, int sync_rows,
    uint8_t *bits_out, int max_bits,
    int *sync_ok);

static bool native_tiles_available(void) { return true; }
#endif

/* ── Tier 2: JNI (JVM at runtime) ─────────────────────────────── */
#ifdef USE_JNI_TILES
#include <jni.h>

static JavaVM *jvm = NULL;
static JNIEnv *jenv = NULL;
static jclass jniTileDecoderClass = NULL;
static jmethodID tileDecodeGridMethod = NULL;
static bool jni_available = false;

bool tile_decode_jni_init(const char *classpath) {
    JavaVMInitArgs vm_args;
    JavaVMOption options[2];
    char cp_buf[4096];

    snprintf(cp_buf, sizeof(cp_buf), "-Djava.class.path=%s", classpath);
    options[0].optionString = cp_buf;
    options[1].optionString = "-Xms64m";

    vm_args.version = JNI_VERSION_1_8;
    vm_args.nOptions = 2;
    vm_args.options = options;
    vm_args.ignoreUnrecognized = JNI_TRUE;

    jint rc = JNI_CreateJavaVM(&jvm, (void**)&jenv, &vm_args);
    if (rc != JNI_OK) {
        fprintf(stderr, "[vidcrypt] JVM creation failed (rc=%d), falling back to C tile decode\n", rc);
        return false;
    }

    jniTileDecoderClass = (*jenv)->FindClass(jenv, "com/vidcrypt/frame/JniTileDecoder");
    if (!jniTileDecoderClass) {
        (*jenv)->ExceptionClear(jenv);
        fprintf(stderr, "[vidcrypt] JniTileDecoder class not found, falling back to C tile decode\n");
        return false;
    }

    tileDecodeGridMethod = (*jenv)->GetStaticMethodID(
        jenv, jniTileDecoderClass, "tileDecodeGrid",
        "([BIIIIIII[BI[Z)I");

    if (!tileDecodeGridMethod) {
        (*jenv)->ExceptionClear(jenv);
        fprintf(stderr, "[vidcrypt] tileDecodeGrid method not found, falling back to C tile decode\n");
        return false;
    }

    jni_available = true;
    fprintf(stderr, "[vidcrypt] Java tile decoder loaded (JNI)\n");
    return true;
}

void tile_decode_jni_shutdown(void) {
    if (jvm) { (*jvm)->DestroyJavaVM(jvm); jvm = NULL; jenv = NULL; }
    jni_available = false;
}

bool tile_decode_jni_available(void) { return jni_available; }
#endif /* USE_JNI_TILES */

/* ── Dispatcher: try each tier ────────────────────────────────── */
int tile_decode_grid_jni(
    const uint8_t *gray, int stride,
    int grid_top_y, int grid_left_x, int block_size,
    int grid_cols, int grid_rows, int sync_rows,
    uint8_t *bits_out, int max_bits,
    bool *sync_ok)
{
#ifdef USE_NATIVE_TILES
    /* Tier 1: GraalVM native library — zero overhead */
    int native_sync_ok = 0;
    int n = tile_decode_grid_native(gray, stride, grid_top_y, grid_left_x,
                                     block_size, grid_cols, grid_rows, sync_rows,
                                     bits_out, max_bits, &native_sync_ok);
    if (n > 0) {
        *sync_ok = (bool)native_sync_ok;
        return n;
    }
#endif

#ifdef USE_JNI_TILES
    /* Tier 2: JVM via JNI */
    if (jni_available) {
        int frame_size = stride * (grid_top_y + grid_rows * block_size + block_size);
        jbyteArray jgray = (*jenv)->NewByteArray(jenv, frame_size);
        if (!jgray) goto fallback;
        (*jenv)->SetByteArrayRegion(jenv, jgray, 0, frame_size, (const jbyte*)gray);

        jbyteArray jbits = (*jenv)->NewByteArray(jenv, max_bits);
        if (!jbits) { (*jenv)->DeleteLocalRef(jenv, jgray); goto fallback; }

        jbooleanArray jsyncOk = (*jenv)->NewBooleanArray(jenv, 1);
        if (!jsyncOk) {
            (*jenv)->DeleteLocalRef(jenv, jgray);
            (*jenv)->DeleteLocalRef(jenv, jbits);
            goto fallback;
        }

        jint result = (*jenv)->CallStaticIntMethod(
            jenv, jniTileDecoderClass, tileDecodeGridMethod,
            jgray, (jint)stride,
            (jint)grid_top_y, (jint)grid_left_x, (jint)block_size,
            (jint)grid_cols, (jint)grid_rows, (jint)sync_rows,
            jbits, (jint)max_bits, jsyncOk);

        if ((*jenv)->ExceptionCheck(jenv)) {
            (*jenv)->ExceptionClear(jenv);
            (*jenv)->DeleteLocalRef(jenv, jgray);
            (*jenv)->DeleteLocalRef(jenv, jbits);
            (*jenv)->DeleteLocalRef(jenv, jsyncOk);
            goto fallback;
        }

        jboolean jok;
        (*jenv)->GetBooleanArrayRegion(jenv, jsyncOk, 0, 1, &jok);
        *sync_ok = (bool)jok;
        (*jenv)->GetByteArrayRegion(jenv, jbits, 0, result, (jbyte*)bits_out);

        (*jenv)->DeleteLocalRef(jenv, jgray);
        (*jenv)->DeleteLocalRef(jenv, jbits);
        (*jenv)->DeleteLocalRef(jenv, jsyncOk);

        if (result > 0) return (int)result;
    }
#endif

fallback:
    /* Tier 3: C SIMD — always works */
    int subsample = block_size / 2;
    if (subsample < 2) subsample = 2;
    return tile_decode_grid(gray, stride, grid_top_y, grid_left_x,
                            block_size, grid_cols, grid_rows, sync_rows,
                            subsample, bits_out, max_bits, sync_ok);
}

/* Stubs when JNI not compiled */
#ifndef USE_JNI_TILES
bool tile_decode_jni_init(const char *cp) { (void)cp; return false; }
void tile_decode_jni_shutdown(void) {}
bool tile_decode_jni_available(void) { return false; }
#endif

#include "h264_writer.h"
#include <stdlib.h>
#include <string.h>
#include <errno.h>

struct H264Writer {
    FILE   *file;
    int64_t bytes_written;
    char    error_buf[256];
};

H264Writer* h264_writer_create(const char *path,
                                const uint8_t *extradata,
                                int extradata_size) {
    if (!path) return NULL;

    H264Writer *w = (H264Writer *)calloc(1, sizeof(H264Writer));
    if (!w) return NULL;

    w->file = fopen(path, "wb");
    if (w->file) setvbuf(w->file, NULL, _IOFBF, 256 * 1024);
    if (!w->file) {
        snprintf(w->error_buf, sizeof(w->error_buf),
                 "Cannot open %s: %s", path, strerror(errno));
        fprintf(stderr, "  [H264Writer] %s\n", w->error_buf);
        free(w);
        return NULL;
    }

    /* Write SPS+PPS extradata at the very beginning */
    if (extradata && extradata_size > 0) {
        size_t nwritten = fwrite(extradata, 1, (size_t)extradata_size, w->file);
        if (nwritten != (size_t)extradata_size) {
            snprintf(w->error_buf, sizeof(w->error_buf),
                     "Failed to write extradata: %s", strerror(errno));
            fprintf(stderr, "  [H264Writer] %s\n", w->error_buf);
            fclose(w->file);
            free(w);
            return NULL;
        }
        w->bytes_written += extradata_size;
        fflush(w->file);
        fprintf(stderr, "  [H264Writer] Wrote %d bytes SPS/PPS extradata to %s\n",
                extradata_size, path);
    }

    return w;
}

bool h264_writer_write_packet(H264Writer *w,
                               const uint8_t *data, int size) {
    if (!w || !w->file || !data || size <= 0) return false;

    size_t nwritten = fwrite(data, 1, (size_t)size, w->file);
    if (nwritten != (size_t)size) {
        snprintf(w->error_buf, sizeof(w->error_buf),
                 "Write failed: %s", strerror(errno));
        fprintf(stderr, "  [H264Writer] %s\n", w->error_buf);
        return false;
    }
    w->bytes_written += size;
    fflush(w->file);
    return true;
}

bool h264_writer_close(H264Writer *w) {
    if (!w) return true;

    bool ok = true;
    if (w->file) {
        fflush(w->file);
        if (fclose(w->file) != 0) {
            fprintf(stderr, "  [H264Writer] Close failed: %s\n", strerror(errno));
            ok = false;
        }
    }
    fprintf(stderr, "  [H264Writer] Closed: %lld total bytes written\n",
            (long long)w->bytes_written);
    free(w);
    return ok;
}

int64_t h264_writer_bytes_written(H264Writer *w) {
    return w ? w->bytes_written : 0;
}

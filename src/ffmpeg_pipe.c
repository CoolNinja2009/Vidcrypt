#include "ffmpeg_pipe.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <stdarg.h>

#ifdef _WIN32
#include <windows.h>
#if defined(_MSC_VER)
#define popen  _popen
#define pclose _pclose
#define strdup _strdup
#endif
static int file_exists(const char *path) {
    return GetFileAttributesA(path) != INVALID_FILE_ATTRIBUTES;
}
#else
#include <unistd.h>
static int file_exists(const char *path) {
    return access(path, F_OK) == 0;
}
#endif

#define ERR_BUF_SIZE 512
static char g_error_buf[ERR_BUF_SIZE] = {0};

void set_error(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_error_buf, ERR_BUF_SIZE, fmt, args);
    va_end(args);
}

const char *ffmpeg_last_error(void) {
    return g_error_buf;
}

static char *join_path(const char *dir, const char *name) {
#ifdef _WIN32
    const char *ext = ".exe";
    const char sep = '\\';
#else
    const char *ext = "";
    const char sep = '/';
#endif
    size_t dirlen = strlen(dir);
    size_t namelen = strlen(name) + strlen(ext) + 2;
    char *path = (char *)malloc(dirlen + namelen + 1);
    if (!path) return NULL;
    snprintf(path, dirlen + namelen + 1, "%s%c%s%s", dir, sep, name, ext);
    return path;
}

char *find_ffmpeg_binary(const char *hint, const char *name) {
    char *result = NULL;
    if (hint) {
        result = strdup(hint);
        if (result && file_exists(result)) return result;
        free(result);
        result = NULL;
    }
    const char *path_env = getenv("PATH");
    if (path_env) {
        char *path_copy = strdup(path_env);
        if (path_copy) {
            const char *sep = 
#ifdef _WIN32
                ";";
#else
                ":";
#endif
            char *dir = strtok(path_copy, sep);
            while (dir) {
                char *candidate = join_path(dir, name);
                if (candidate && file_exists(candidate)) {
                    free(candidate);
                    result = strdup(name);
                    break;
                }
                free(candidate);
                dir = strtok(NULL, sep);
            }
            free(path_copy);
        }
        if (result) return result;
    }
    const char *common_dirs[] = {
#ifdef _WIN32
        "C:\\Program Files\\ffmpeg\\bin",
        "C:\\Program Files (x86)\\ffmpeg\\bin",
        "C:\\ffmpeg\\bin",
        "C:\\tools\\ffmpeg\\bin",
        "C:\\Users\\cooln\\Documents\\ffmpeg-master-latest-win64-lgpl-shared\\bin",
        NULL
#else
        "/usr/bin", "/usr/local/bin", "/opt/homebrew/bin", NULL
#endif
    };
    for (int i = 0; common_dirs[i]; ++i) {
        char *candidate = join_path(common_dirs[i], name);
        if (candidate && file_exists(candidate)) return candidate;
        free(candidate);
    }
    result = strdup(name);
    return result;
}

typedef struct {
    int width, height, frame_count;
    double fps;
    int valid;
} ProbeResult;

static int probe_video(const char *ffprobe_path, const char *video_path, ProbeResult *out) {
    memset(out, 0, sizeof(ProbeResult));
    char cmd[4096];
    int n;
    if (strchr(ffprobe_path, ' ') || strchr(video_path, ' '))
        n = snprintf(cmd, sizeof(cmd), "\"%s\" -v quiet -print_format json -show_streams \"%s\"",
                     ffprobe_path, video_path);
    else
        n = snprintf(cmd, sizeof(cmd), "%s -v quiet -print_format json -show_streams \"%s\"",
                     ffprobe_path, video_path);
    if ((size_t)n >= sizeof(cmd)) { set_error("Command too long"); return -1; }

    FILE *fp = popen(cmd, "r");
    if (!fp) { set_error("Failed to run ffprobe: %s", strerror(errno)); return -1; }

    char buf[65536];
    size_t total = 0, nread;
    while (total < sizeof(buf) - 1 && (nread = fread(buf + total, 1, sizeof(buf) - 1 - total, fp)) > 0)
        total += nread;
    buf[total] = '\0';
    pclose(fp);

    out->width = out->height = 0;
    out->fps = 0.0;
    out->frame_count = 0;

    const char *w = strstr(buf, "\"width\":"); if (w) out->width = atoi(w + 8);
    const char *h = strstr(buf, "\"height\":"); if (h) out->height = atoi(h + 9);
    const char *nf = strstr(buf, "\"nb_frames\":");
    if (nf) { const char *val = nf + 12; while (*val == ' ') val++; if (*val != '\"') out->frame_count = atoi(val); }
    const char *rf = strstr(buf, "\"r_frame_rate\":");
    if (rf) { const char *val = rf + 16; while (*val == ' ') val++; int num=0, den=1; if (sscanf(val, "%d/%d", &num, &den)==2 && den>0) out->fps = (double)num/(double)den; }
    if (out->fps <= 0.0) {
        const char *af = strstr(buf, "\"avg_frame_rate\":");
        if (af) { const char *val = af + 18; while (*val == ' ') val++; int num=0, den=1; if (sscanf(val, "%d/%d", &num, &den)==2 && den>0) out->fps = (double)num/(double)den; }
    }
    if (out->width <= 0 || out->height <= 0) { set_error("ffprobe could not determine video dimensions"); return -1; }
    if (out->fps <= 0.0) out->fps = 30.0;
    out->valid = 1;
    return 0;
}

struct VideoReader {
    char   *ffmpeg_path;
    int     width, height, stride;
    double  fps;
    int     frame_count;
    FILE   *pipe;
    uint8_t *frame_buf;
    int     frame_size;
};

VideoReader *video_reader_open_nvdec(const char *path, const char *ffmpeg_path) {
    char *ffmpeg  = find_ffmpeg_binary(ffmpeg_path, "ffmpeg");
    char *ffprobe = find_ffmpeg_binary(ffmpeg_path, "ffprobe");
    if (!ffmpeg)  { set_error("ffmpeg not found"); free(ffprobe); return NULL; }
    if (!ffprobe) { set_error("ffprobe not found"); free(ffmpeg); return NULL; }

    ProbeResult probe;
    if (probe_video(ffprobe, path, &probe) < 0) { free(ffmpeg); free(ffprobe); return NULL; }

    VideoReader *vr = (VideoReader *)calloc(1, sizeof(VideoReader));
    if (!vr) { free(ffmpeg); free(ffprobe); set_error("Out of memory"); return NULL; }

    vr->ffmpeg_path = ffmpeg;
    vr->width       = probe.width;
    vr->height      = probe.height;
    vr->stride      = probe.width;
    vr->fps         = probe.fps;
    vr->frame_count = probe.frame_count;
    vr->frame_size  = probe.width * probe.height;

    vr->frame_buf = (uint8_t *)malloc((size_t)vr->frame_size);
    if (!vr->frame_buf) { video_reader_close(vr); free(ffprobe); set_error("Out of memory"); return NULL; }

    /* ffmpeg -c:v h264_cuvid -i <input> -f rawvideo -pix_fmt gray -an -sn -dn -
     * Uses NVDEX CUVID hardware decoder for H.264. Note: -hwaccel cuda is NOT
     * used here because it conflicts with -c:v h264_cuvid (both try to manage
     * CUDA resources, causing crashes). -c:v h264_cuvid alone is sufficient. */
    char cmd[8192];
    int n;
    if (strchr(ffmpeg, ' ') || strchr(path, ' '))
        n = snprintf(cmd, sizeof(cmd),
                     "\"%s\" -c:v h264_cuvid -i \"%s\" -f rawvideo -pix_fmt gray -an -sn -dn -",
                     ffmpeg, path);
    else
        n = snprintf(cmd, sizeof(cmd),
                     "%s -c:v h264_cuvid -i \"%s\" -f rawvideo -pix_fmt gray -an -sn -dn -",
                     ffmpeg, path);
    if ((size_t)n >= sizeof(cmd)) { video_reader_close(vr); free(ffprobe); set_error("Command too long"); return NULL; }

    vr->pipe = popen(cmd, "rb");
    if (!vr->pipe) { video_reader_close(vr); free(ffprobe); set_error("Failed to launch ffmpeg: %s", strerror(errno)); return NULL; }

    free(ffprobe);
    return vr;
}

VideoReader *video_reader_open(const char *path, const char *ffmpeg_path) {
    char *ffmpeg  = find_ffmpeg_binary(ffmpeg_path, "ffmpeg");
    char *ffprobe = find_ffmpeg_binary(ffmpeg_path, "ffprobe");
    if (!ffmpeg)  { set_error("ffmpeg not found"); free(ffprobe); return NULL; }
    if (!ffprobe) { set_error("ffprobe not found"); free(ffmpeg); return NULL; }

    ProbeResult probe;
    if (probe_video(ffprobe, path, &probe) < 0) { free(ffmpeg); free(ffprobe); return NULL; }

    VideoReader *vr = (VideoReader *)calloc(1, sizeof(VideoReader));
    if (!vr) { free(ffmpeg); free(ffprobe); set_error("Out of memory"); return NULL; }

    vr->ffmpeg_path = ffmpeg;
    vr->width       = probe.width;
    vr->height      = probe.height;
    vr->stride      = probe.width;        /* gray8: 1 byte per pixel */
    vr->fps         = probe.fps;
    vr->frame_count = probe.frame_count;
    vr->frame_size  = probe.width * probe.height;  /* gray8: w*h */

    vr->frame_buf = (uint8_t *)malloc((size_t)vr->frame_size);
    if (!vr->frame_buf) { video_reader_close(vr); free(ffprobe); set_error("Out of memory"); return NULL; }

    /* ffmpeg -i <input> -f rawvideo -pix_fmt gray8 -an -sn -dn - */
    char cmd[8192];
    int n;
    if (strchr(ffmpeg, ' ') || strchr(path, ' '))
        n = snprintf(cmd, sizeof(cmd), "\"%s\" -i \"%s\" -f rawvideo -pix_fmt gray -an -sn -dn -",
                     ffmpeg, path);
    else
        n = snprintf(cmd, sizeof(cmd), "%s -i \"%s\" -f rawvideo -pix_fmt gray -an -sn -dn -",
                     ffmpeg, path);
    if ((size_t)n >= sizeof(cmd)) { video_reader_close(vr); free(ffprobe); set_error("Command too long"); return NULL; }

    vr->pipe = popen(cmd, "rb");
    if (!vr->pipe) { video_reader_close(vr); free(ffprobe); set_error("Failed to launch ffmpeg: %s", strerror(errno)); return NULL; }

    free(ffprobe);
    return vr;
}

VideoReader *video_reader_open_scaled(const char *path, const char *ffmpeg_path,
                                       int target_width, int target_height) {
    if (target_width <= 0 || target_height <= 0)
        return video_reader_open(path, ffmpeg_path);

    char *ffmpeg  = find_ffmpeg_binary(ffmpeg_path, "ffmpeg");
    char *ffprobe = find_ffmpeg_binary(ffmpeg_path, "ffprobe");
    if (!ffmpeg)  { set_error("ffmpeg not found"); free(ffprobe); return NULL; }
    if (!ffprobe) { set_error("ffprobe not found"); free(ffmpeg); return NULL; }

    /* Skip probe — we don't need the source dimensions, just output at target */

    VideoReader *vr = (VideoReader *)calloc(1, sizeof(VideoReader));
    if (!vr) { free(ffmpeg); free(ffprobe); set_error("Out of memory"); return NULL; }

    vr->ffmpeg_path = ffmpeg;
    vr->width       = target_width;
    vr->height      = target_height;
    vr->stride      = target_width;
    vr->fps         = 0.0;  /* filled by probe but unused for scaled path */
    vr->frame_count = 0;
    vr->frame_size  = target_width * target_height;

    vr->frame_buf = (uint8_t *)malloc((size_t)vr->frame_size);
    if (!vr->frame_buf) { video_reader_close(vr); free(ffprobe); set_error("Out of memory"); return NULL; }

    /* ffmpeg -i <input> -vf scale=W:H -f rawvideo -pix_fmt gray -an -sn -dn - */
    char cmd[8192];
    int n;
    if (strchr(ffmpeg, ' ') || strchr(path, ' '))
        n = snprintf(cmd, sizeof(cmd),
                     "\"%s\" -i \"%s\" -vf scale=%d:%d -f rawvideo -pix_fmt gray -an -sn -dn -",
                     ffmpeg, path, target_width, target_height);
    else
        n = snprintf(cmd, sizeof(cmd),
                     "%s -i \"%s\" -vf scale=%d:%d -f rawvideo -pix_fmt gray -an -sn -dn -",
                     ffmpeg, path, target_width, target_height);
    if ((size_t)n >= sizeof(cmd)) { video_reader_close(vr); free(ffprobe); set_error("Command too long"); return NULL; }

    vr->pipe = popen(cmd, "rb");
    if (!vr->pipe) { video_reader_close(vr); free(ffprobe); set_error("Failed to launch ffmpeg: %s", strerror(errno)); return NULL; }

    free(ffprobe);
    return vr;
}

bool video_reader_read_frame(VideoReader *vr, const uint8_t **frame_out,
                              int *stride_out, int *width_out, int *height_out) {
    if (!vr || !vr->pipe) return false;
    size_t nread = fread(vr->frame_buf, 1, (size_t)vr->frame_size, vr->pipe);
    if ((int)nread != vr->frame_size) return false;
    *frame_out  = vr->frame_buf;
    *stride_out = vr->stride;
    *width_out  = vr->width;
    *height_out = vr->height;
    return true;
}

int video_reader_frame_count(VideoReader *vr) { return vr ? vr->frame_count : 0; }
double video_reader_fps(VideoReader *vr) { return vr ? vr->fps : 0.0; }

void video_reader_close(VideoReader *vr) {
    if (!vr) return;
    if (vr->pipe) pclose(vr->pipe);
    free(vr->ffmpeg_path);
    free(vr->frame_buf);
    free(vr);
}

struct VideoWriter {
    char   *ffmpeg_path;
    int     width, height;
    int     frame_size;
    FILE   *pipe;
    int     frames_written;
};

VideoWriter *video_writer_create(const char *ffmpeg_path,
                                  const char *path,
                                  int width, int height,
                                  double fps,
                                  const char *codec_name) {
    char *ffmpeg = find_ffmpeg_binary(ffmpeg_path, "ffmpeg");
    if (!ffmpeg) { set_error("ffmpeg not found"); return NULL; }
    if (!codec_name) codec_name = "ffv1";

    VideoWriter *vw = (VideoWriter *)calloc(1, sizeof(VideoWriter));
    if (!vw) { free(ffmpeg); set_error("Out of memory"); return NULL; }

    vw->ffmpeg_path = ffmpeg;
    vw->width       = width;
    vw->height      = height;
    vw->frame_size  = width * height;  /* gray8: w*h */

    /* Build ffmpeg command with codec and quality options.
     * For H.264 we use high-quality lossy encoding (-crf 10 -preset fast)
     * because NVDEC hardware decoders DO NOT support lossless H.264
     * (transform-bypass / QP 0 mode). CRF 10 is visually lossless with
     * sharp tile edges preserved — any pixel-level errors are handled
     * by RS(255,223,32) ECC which corrects up to 16 bytes per block.
     * Other codecs (ffv1) are inherently lossless. */
    char quality[64] = "";
    if (strcmp(codec_name, "h264") == 0 || strcmp(codec_name, "libx264") == 0)
        snprintf(quality, sizeof(quality), " -crf 10 -preset fast");

    char cmd[8192];
    int n;
    if (strchr(ffmpeg, ' ') || strchr(path, ' '))
        n = snprintf(cmd, sizeof(cmd),
                     "\"%s\" -y -f rawvideo -pix_fmt gray -s %dx%d -r %.2f -i - -c %s%s -pix_fmt gray -an \"%s\"",
                     ffmpeg, width, height, fps, codec_name, quality, path);
    else
        n = snprintf(cmd, sizeof(cmd),
                     "%s -y -f rawvideo -pix_fmt gray -s %dx%d -r %.2f -i - -c %s%s -pix_fmt gray -an \"%s\"",
                     ffmpeg, width, height, fps, codec_name, quality, path);
    if ((size_t)n >= sizeof(cmd)) { video_writer_close(vw); set_error("Command too long"); return NULL; }

    vw->pipe = popen(cmd, "wb");
    if (!vw->pipe) { video_writer_close(vw); set_error("Failed to launch ffmpeg: %s", strerror(errno)); return NULL; }

    return vw;
}

bool video_writer_write_frame(VideoWriter *vw, const uint8_t *frame, int stride) {
    (void)stride;
    if (!vw || !vw->pipe) return false;
    size_t nwritten = fwrite(frame, 1, (size_t)vw->frame_size, vw->pipe);
    if ((int)nwritten != vw->frame_size) {
        set_error("Failed to write frame to ffmpeg: %s", strerror(errno));
        return false;
    }
    fflush(vw->pipe);
    vw->frames_written++;
    return true;
}

void video_writer_close(VideoWriter *vw) {
    if (!vw) return;
    if (vw->pipe) {
        int exit_code = pclose(vw->pipe);
        if (exit_code != 0)
            fprintf(stderr, "Warning: ffmpeg exited with code %d\n", exit_code);
    }
    free(vw->ffmpeg_path);
    free(vw);
}

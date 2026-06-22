#include "libav_decoder.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#ifdef HAS_FFMPEG

/* ─── FFmpeg headers ───────────────────────────────────────────────── */
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/pixdesc.h>
#include <libavutil/imgutils.h>
#include <libavutil/error.h>
#include <libswscale/swscale.h>

/* ─── FFmpeg log suppression (called once) ─────────────────────────── */
static int g_ffmpeg_quieted = 0;
static void quiet_ffmpeg(void) {
    if (!g_ffmpeg_quieted) {
        av_log_set_level(AV_LOG_QUIET);
        g_ffmpeg_quieted = 1;
    }
}

/* ─── Internal state ──────────────────────────────────────────────── */
struct LibAvDecoder {
    /* Demuxer */
    AVFormatContext *fmt_ctx;
    int              video_stream_idx;
    enum AVCodecID   codec_id;

    /* Decoder */
    AVCodecContext  *decoder_ctx;
    const AVCodec   *decoder;
    bool             is_hardware;   /* true if using h264_cuvid etc. */
    char             decoder_name[64];

    /* Reusable FFmpeg objects */
    AVPacket        *pkt;
    AVFrame         *frame;

    /* Scaler (libswscale) for resolution conversion */
    struct SwsContext *sws_ctx;
    int              sws_width;
    int              sws_height;
    uint8_t         *sws_buffer;    /* output buffer for scaled gray frame */
    int              sws_buffer_size;
    int              sws_pitch;     /* bytes per row of scaled output */

    /* Frame info */
    int              width;
    int              height;
    int              pix_fmt;       /* AVPixelFormat of decoded frames */
    int              frame_count;   /* actual decoded count */
    double           fps;
    int64_t          pts;          /* presentation timestamp */
    bool             eos;

    /* Error buffer */
    char             error_buf[256];
};

#define SET_ERR(dec, fmt, ...) do { \
    snprintf((dec)->error_buf, sizeof((dec)->error_buf), fmt, ##__VA_ARGS__); \
} while(0)

/* ─── Helpers ─────────────────────────────────────────────────────── */
static void log_err(const char *context, int errnum) {
    char errbuf[128];
    av_strerror(errnum, errbuf, sizeof(errbuf));
    fprintf(stderr, "FFmpeg %s: %s\n", context, errbuf);
}

/* ─── Open decoder (core) ─────────────────────────────────────────── */
static bool open_decoder_internal(LibAvDecoder *dec,
                                   bool prefer_hw,
                                   char *error_out, int error_size) {
    AVStream *stream = dec->fmt_ctx->streams[dec->video_stream_idx];
    dec->codec_id    = stream->codecpar->codec_id;
    dec->width       = stream->codecpar->width;
    dec->height      = stream->codecpar->height;
    dec->frame_count = 0;
    dec->pix_fmt     = AV_PIX_FMT_NONE;

    /* Frame rate */
    if (stream->avg_frame_rate.den > 0 && stream->avg_frame_rate.num > 0)
        dec->fps = (double)stream->avg_frame_rate.num / (double)stream->avg_frame_rate.den;
    else if (stream->r_frame_rate.den > 0 && stream->r_frame_rate.num > 0)
        dec->fps = (double)stream->r_frame_rate.num / (double)stream->r_frame_rate.den;
    else
        dec->fps = 30.0;

    /* ── Try hardware decoder first ──────────────────────────────── */
    dec->is_hardware = false;
    if (prefer_hw) {
        const char *hw_name = NULL;
        switch (dec->codec_id) {
            case AV_CODEC_ID_H264: hw_name = "h264_cuvid"; break;
            case AV_CODEC_ID_HEVC: hw_name = "hevc_cuvid"; break;
            case AV_CODEC_ID_AV1:  hw_name = "av1_cuvid";  break;
            case AV_CODEC_ID_VP9:  hw_name = "vp9_cuvid";  break;
            default: break;
        }
        if (hw_name) {
            const AVCodec *hw_dec = avcodec_find_decoder_by_name(hw_name);
            if (hw_dec) {
                dec->decoder_ctx = avcodec_alloc_context3(hw_dec);
                if (dec->decoder_ctx) {
                    int ret = avcodec_parameters_to_context(dec->decoder_ctx,
                                                             stream->codecpar);
                    if (ret == 0) {
                        ret = avcodec_open2(dec->decoder_ctx, hw_dec, NULL);
                        if (ret == 0) {
                            dec->decoder    = hw_dec;
                            dec->is_hardware = true;
                            snprintf(dec->decoder_name, sizeof(dec->decoder_name),
                                     "%s", hw_name);
                        }
                    }
                    if (!dec->is_hardware) {
                        avcodec_free_context(&dec->decoder_ctx);
                    }
                }
            }
        }
    }

    /* ── Fallback to software decoder ─────────────────────────────── */
    if (!dec->is_hardware) {
        dec->decoder = avcodec_find_decoder(dec->codec_id);
        if (!dec->decoder) {
            if (error_out)
                snprintf(error_out, (size_t)error_size,
                         "Unsupported codec: %d", dec->codec_id);
            return false;
        }

        dec->decoder_ctx = avcodec_alloc_context3(dec->decoder);
        if (!dec->decoder_ctx) {
            if (error_out)
                snprintf(error_out, (size_t)error_size, "Out of memory");
            return false;
        }

        int ret = avcodec_parameters_to_context(dec->decoder_ctx, stream->codecpar);
        if (ret < 0) {
            log_err("avcodec_parameters_to_context", ret);
            SET_ERR(dec, "Cannot copy codec parameters");
            return false;
        }

        /* Enable multi-threaded decoding for software path.
         * Hardware decoders (cuvid) handle threading internally. */
        dec->decoder_ctx->thread_count = 0; /* auto */
        dec->decoder_ctx->thread_type  = FF_THREAD_SLICE | FF_THREAD_FRAME;

        ret = avcodec_open2(dec->decoder_ctx, dec->decoder, NULL);
        if (ret < 0) {
            log_err("avcodec_open2", ret);
            SET_ERR(dec, "Cannot open decoder");
            return false;
        }

        /* Record the actual pixel format for scaler source format selection */
        dec->pix_fmt = dec->decoder_ctx->pix_fmt;

        snprintf(dec->decoder_name, sizeof(dec->decoder_name),
                 "%s (multi-thread)", dec->decoder->name);
        dec->is_hardware = false;
    }

    /* Record pixel format for hardware decoder too */
    dec->pix_fmt = dec->decoder_ctx->pix_fmt;

    return true;
}

/* ─── Public API ──────────────────────────────────────────────────── */

LibAvDecoder* libav_decoder_open(const char *path, int threads,
                                  bool prefer_hw,
                                  int target_width, int target_height,
                                  char *error_out, int error_size) {
    if (!path) {
        if (error_out) snprintf(error_out, (size_t)error_size, "NULL path");
        return NULL;
    }

    quiet_ffmpeg();

    LibAvDecoder *dec = (LibAvDecoder *)calloc(1, sizeof(LibAvDecoder));
    if (!dec) {
        if (error_out) snprintf(error_out, (size_t)error_size, "Out of memory");
        return NULL;
    }

    /* ── 1. Open container ───────────────────────────────────────── */
    int ret = avformat_open_input(&dec->fmt_ctx, path, NULL, NULL);
    if (ret < 0) {
        log_err("avformat_open_input", ret);
        SET_ERR(dec, "Cannot open %s", path);
        goto fail;
    }

    ret = avformat_find_stream_info(dec->fmt_ctx, NULL);
    if (ret < 0) {
        log_err("avformat_find_stream_info", ret);
        SET_ERR(dec, "Cannot find stream info");
        goto fail;
    }

    /* ── 2. Find video stream ────────────────────────────────────── */
    dec->video_stream_idx = av_find_best_stream(dec->fmt_ctx,
                                                  AVMEDIA_TYPE_VIDEO,
                                                  -1, -1, NULL, 0);
    if (dec->video_stream_idx < 0) {
        SET_ERR(dec, "No video stream found");
        goto fail;
    }

    /* ── 3. Open decoder ─────────────────────────────────────────── */
    if (!open_decoder_internal(dec, prefer_hw, error_out, error_size))
        goto fail;

    /* Override thread count if specified */
    if (threads > 0 && !dec->is_hardware)
        dec->decoder_ctx->thread_count = threads;

    /* ── 4. Allocate reusable objects ────────────────────────────── */
    dec->pkt   = av_packet_alloc();
    dec->frame = av_frame_alloc();
    if (!dec->pkt || !dec->frame) {
        SET_ERR(dec, "Out of memory for FFmpeg objects");
        goto fail;
    }

    /* ── 5. Set up scaler if target resolution specified ─────────── */
    if (target_width > 0 && target_height > 0) {
        if (!libav_decoder_set_scale(dec, target_width, target_height)) {
            SET_ERR(dec, "Cannot create scaler context");
            goto fail;
        }
    }

    if (error_out) error_out[0] = '\0';
    return dec;

fail:
    if (error_out && error_out[0] == '\0')
        snprintf(error_out, (size_t)error_size, "%s", dec->error_buf);
    libav_decoder_close(dec);
    return NULL;
}

void libav_decoder_close(LibAvDecoder *dec) {
    if (!dec) return;

    if (dec->sws_ctx)   sws_freeContext(dec->sws_ctx);
    free(dec->sws_buffer);

    av_frame_free(&dec->frame);
    av_packet_free(&dec->pkt);
    if (dec->decoder_ctx) avcodec_free_context(&dec->decoder_ctx);
    if (dec->fmt_ctx)     avformat_close_input(&dec->fmt_ctx);

    memset(dec, 0, sizeof(LibAvDecoder));
    free(dec);
}

bool libav_decoder_read_frame(LibAvDecoder *dec,
                               uint8_t **gray_out,
                               int *pitch_out,
                               int *width_out,
                               int *height_out) {
    if (!dec || dec->eos) {
        if (gray_out)   *gray_out = NULL;
        if (pitch_out)  *pitch_out = 0;
        if (width_out)  *width_out = 0;
        if (height_out) *height_out = 0;
        return false;
    }

    /* ─── Iterative decode loop ──────────────────────────────────── */
    while (1) {
        int ret = avcodec_receive_frame(dec->decoder_ctx, dec->frame);

        if (ret >= 0) {
            /* Got a frame */
            dec->frame_count++;

            uint8_t *y_plane = NULL;
            int      y_pitch = 0;
            int      fw      = dec->frame->width;
            int      fh      = dec->frame->height;

            if (dec->sws_ctx && dec->sws_buffer) {
                /* ── Rescale via libswscale ────────────────────────
                 * Convert decoded frame to gray8 at target resolution.
                 * sws_scale handles pixel format conversion (NV12→gray,
                 * YUV420P→gray, etc.) and scaling in one step. */
                const uint8_t *src_slices[4];
                int            src_strides[4];
                for (int i = 0; i < 4; ++i) {
                    src_slices[i]  = dec->frame->data[i];
                    src_strides[i] = dec->frame->linesize[i];
                }

                uint8_t *dst_slices[4] = {dec->sws_buffer, NULL, NULL, NULL};
                int      dst_strides[4] = {dec->sws_pitch, 0, 0, 0};

                sws_scale(dec->sws_ctx,
                          src_slices, src_strides, 0, fh,
                          dst_slices, dst_strides);

                y_plane = dec->sws_buffer;
                y_pitch = dec->sws_pitch;
                fw      = dec->sws_width;
                fh      = dec->sws_height;
            } else {
                /* ── Direct Y-plane (grayscale) ────────────────────
                 * For NV12, the Y-plane IS grayscale (1 byte/pixel).
                 * For YUV420P, data[0] is the Y-plane (also grayscale).
                 * For gray8 (ffv1), data[0] is already grayscale. */
                y_plane = dec->frame->data[0];
                y_pitch = dec->frame->linesize[0];
            }

            *gray_out  = y_plane;
            *pitch_out = y_pitch;
            *width_out  = fw;
            *height_out = fh;

            /* Frame data is valid until next read_frame call.
             * We DON'T unref here — caller must process before next call.
             * The next call to read_frame will unref this frame. */
            return true;
        }

        if (ret == AVERROR_EOF) {
            dec->eos = true;
            *gray_out = NULL;
            *pitch_out = 0;
            *width_out = 0;
            *height_out = 0;
            return false;
        }

        if (ret < 0 && ret != AVERROR(EAGAIN)) {
            log_err("avcodec_receive_frame", ret);
            SET_ERR(dec, "Decode error");
            dec->eos = true;
            *gray_out = NULL;
            *pitch_out = 0;
            *width_out = 0;
            *height_out = 0;
            return false;
        }

        /* ── EAGAIN: need more packets ────────────────────────────── */
        int pkt_ret = av_read_frame(dec->fmt_ctx, dec->pkt);
        if (pkt_ret < 0) {
            /* EOF or error — flush decoder */
            avcodec_send_packet(dec->decoder_ctx, NULL);
            continue;
        }

        if (dec->pkt->stream_index == dec->video_stream_idx) {
            ret = avcodec_send_packet(dec->decoder_ctx, dec->pkt);
            if (ret < 0) {
                log_err("avcodec_send_packet", ret);
            }
        }
        av_packet_unref(dec->pkt);
    }
}

/* ─── Metadata ────────────────────────────────────────────────────── */

int libav_decoder_width(LibAvDecoder *dec) {
    return dec ? (dec->sws_width > 0 ? dec->sws_width : dec->width) : 0;
}

int libav_decoder_height(LibAvDecoder *dec) {
    return dec ? (dec->sws_height > 0 ? dec->sws_height : dec->height) : 0;
}

int libav_decoder_frame_count(LibAvDecoder *dec) {
    return dec ? dec->frame_count : 0;
}

double libav_decoder_fps(LibAvDecoder *dec) {
    return dec ? dec->fps : 0.0;
}

bool libav_decoder_using_hardware(LibAvDecoder *dec) {
    return dec ? dec->is_hardware : false;
}

const char* libav_decoder_name(LibAvDecoder *dec) {
    return dec ? dec->decoder_name : "none";
}

bool libav_decoder_set_scale(LibAvDecoder *dec,
                              int target_width, int target_height) {
    if (!dec) return false;

    /* Free old scaler and buffer */
    if (dec->sws_ctx) {
        sws_freeContext(dec->sws_ctx);
        dec->sws_ctx = NULL;
    }
    free(dec->sws_buffer);
    dec->sws_buffer = NULL;

    if (target_width <= 0 || target_height <= 0) {
        dec->sws_width  = 0;
        dec->sws_height = 0;
        return true;
    }

    /* Create scaler: source format is whatever the decoder outputs,
     * destination format is AV_PIX_FMT_GRAY8 (grayscale, 1 byte/pixel) */
    enum AVPixelFormat src_fmt = dec->pix_fmt != AV_PIX_FMT_NONE
                                 ? (enum AVPixelFormat)dec->pix_fmt
                                 : AV_PIX_FMT_NV12;

    dec->sws_ctx = sws_getContext(
        dec->width, dec->height, src_fmt,
        target_width, target_height, AV_PIX_FMT_GRAY8,
        SWS_BILINEAR, NULL, NULL, NULL);
    if (!dec->sws_ctx) return false;

    dec->sws_width  = target_width;
    dec->sws_height = target_height;
    dec->sws_pitch  = target_width;  /* GRAY8: 1 byte/pixel */

    /* Allocate output buffer */
    dec->sws_buffer_size = target_width * target_height;
    dec->sws_buffer = (uint8_t *)malloc((size_t)dec->sws_buffer_size);
    if (!dec->sws_buffer) return false;

    return true;
}

#else /* !HAS_FFMPEG — stubs for non-CUDA builds */

/* When HAS_FFMPEG is not defined (non-CUDA build), the decoder cannot
 * use libavcodec directly. These stubs return errors gracefully. */

LibAvDecoder* libav_decoder_open(const char *path, int threads,
                                  bool prefer_hw,
                                  int target_width, int target_height,
                                  char *error_out, int error_size) {
    (void)path; (void)threads; (void)prefer_hw;
    (void)target_width; (void)target_height;
    if (error_out)
        snprintf(error_out, (size_t)error_size,
                 "libav_decoder requires FFmpeg (enable USE_CUDA)");
    return NULL;
}

void libav_decoder_close(LibAvDecoder *dec) { (void)dec; }

bool libav_decoder_read_frame(LibAvDecoder *dec,
                               uint8_t **g, int *p, int *w, int *h) {
    (void)dec; *g = NULL; *p = *w = *h = 0; return false;
}

int libav_decoder_width(LibAvDecoder *dec) { (void)dec; return 0; }
int libav_decoder_height(LibAvDecoder *dec) { (void)dec; return 0; }
int libav_decoder_frame_count(LibAvDecoder *dec) { (void)dec; return 0; }
double libav_decoder_fps(LibAvDecoder *dec) { (void)dec; return 0.0; }
bool libav_decoder_using_hardware(LibAvDecoder *dec) { (void)dec; return false; }
const char* libav_decoder_name(LibAvDecoder *dec) { (void)dec; return "no-ffmpeg"; }

bool libav_decoder_set_scale(LibAvDecoder *dec,
                              int tw, int th) {
    (void)dec; (void)tw; (void)th; return false;
}

#endif /* HAS_FFMPEG */

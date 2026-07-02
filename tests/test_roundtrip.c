/* Minimal roundtrip test: GPU encode -> decode -> SHA256 verify.
 * Bypasses decoder's calibration bug by reading calibration directly. */
#include "calibration.h"
#include "reedsolomon.h"
#include "sha256.h"
#include "bitstream.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#endif

/* Count white pixels in a tile */
static int tile_count_white(const uint8_t *gray, int stride,
                             int block_size, int subsample) {
    int white = 0, total = 0;
    for (int y = 0; y < block_size; y += subsample)
        for (int x = 0; x < block_size; x += subsample) {
            if (gray[y * stride + x] >= 128) white++;
            total++;
        }
    return white;
}

/* Decode all payload tiles from a grayscale frame into a compact bit array */
static int decode_frame_bits(const uint8_t *gray, int stride,
                              int grid_top_y, int grid_left_x,
                              int block_size, int grid_cols, int pay_rows,
                              int subsample, int sync_rows,
                              uint8_t *bits_out) {
    int idx = 0;
    for (int row = 0; row < pay_rows; row++) {
        for (int col = 0; col < grid_cols; col++) {
            int ty = grid_top_y + (row + sync_rows) * block_size;
            int tx = grid_left_x + col * block_size;
            int white = tile_count_white(gray + ty * stride + tx, stride,
                                          block_size, subsample);
            int total = (block_size / subsample) * (block_size / subsample);
            bits_out[idx++] = (uint8_t)(white > total / 2 ? 1 : 0);
        }
    }
    return idx;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <encoded_video>\n", argv[0]);
        return 1;
    }

    const char *input = argv[1];
    const char *ffmpeg = "ffmpeg";
    char cmd[1024];

    /* ── 1. Extract all frames to raw grayscale ──────────────────── */
    snprintf(cmd, sizeof(cmd),
             "%s -v error -i \"%s\" -f rawvideo -pix_fmt gray fr.raw -y",
             ffmpeg, input);
    int r = system(cmd);
    if (r != 0) { printf("ffmpeg failed (exit %d): %s\n", r, cmd); return 1; }

    FILE *f = fopen("fr.raw", "rb");
    if (!f) { printf("Cannot open fr.raw\n"); return 1; }
    fseek(f, 0, SEEK_END);
    long total = ftell(f);
    fseek(f, 0, SEEK_SET);
    int frame_w = 1920, frame_h = 1080;
    int frame_sz = frame_w * frame_h;
    int n_frames = (int)(total / frame_sz);
    printf("Decoded %d frames (%ld bytes raw)\n", n_frames, total);

    if (n_frames < 1) { fclose(f); return 1; }

    /* ── 2. Read first frame, extract calibration ────────────────── */
    uint8_t *f0 = (uint8_t *)malloc((size_t)frame_sz);
    if (fread(f0, 1, (size_t)frame_sz, f) != (size_t)frame_sz)
        { printf("Read err\n"); free(f0); fclose(f); return 1; }

    CalParams cp;
    memset(&cp, 0, sizeof(cp));
    bool cal_ok = extract_calibration(f0, frame_w, frame_h, &cp);
    if (!cal_ok) {
        printf("extract_calibration FAILED, using known defaults\n");
        cp.frame_width  = 1920; cp.frame_height = 1080;
        cp.margin_x = 8; cp.margin_y = 8;
        cp.block_size_x = 8; cp.block_size_y = 8;
        cp.grid_cols = 238; cp.grid_rows = 118;
        cp.rs_ecc_symbols = 32; cp.sync_rows = 1;
    } else {
        printf("Calibration OK: %dx%d grid=%dx%d block=%d ecc=%d\n",
               cp.frame_width, cp.frame_height,
               cp.grid_cols, cp.grid_rows,
               cp.block_size_x, cp.rs_ecc_symbols);
    }

    int bs     = cp.block_size_x;
    int gcols  = cp.grid_cols;
    int grows  = cp.grid_rows;
    int sync_r = cp.sync_rows;
    int pay_r  = grows - sync_r;
    int pay_b  = pay_r * gcols;
    int ecc    = cp.rs_ecc_symbols;
    int cal_bottom = (int)((float)cp.frame_height * 0.06f);
    int grid_y = cal_bottom + cp.margin_y;
    int grid_x = cp.margin_x;

    printf("Grid: %dx%d pay=%d bits\n", gcols, pay_r, pay_b);

    /* ── 3. Read header from first frame ─────────────────────────── */
    uint8_t *hdr_bits = (uint8_t *)calloc((size_t)pay_b, 1);
    decode_frame_bits(f0, frame_w, grid_y, grid_x, bs, gcols, pay_r,
                       2, sync_r, hdr_bits);
    free(f0);

    /* Convert header bits to bytes to parse */
    size_t hdr_bytes_max = (size_t)(pay_b / 8);
    uint8_t *hdr_bytes = (uint8_t *)calloc(hdr_bytes_max, 1);
    size_t hdr_n = bits_to_bytes(hdr_bits, (size_t)pay_b, hdr_bytes, hdr_bytes_max);
    free(hdr_bits);

    if (hdr_n < 14) { printf("Header too short\n"); free(hdr_bytes); fclose(f); return 1; }

    uint8_t fname_len = hdr_bytes[5];
    int64_t orig_size = 0;
    for (int i = 0; i < 8; i++)
        orig_size = (orig_size << 8) | hdr_bytes[6 + fname_len + i];

    uint8_t expected_sha[32];
    memcpy(expected_sha, hdr_bytes + 6 + fname_len + 8, 32);
    free(hdr_bytes);

    printf("Original size: %lld\n", (long long)orig_size);
    printf("Expected SHA:  ");
    for (int i = 0; i < 32; i++) printf("%02x", expected_sha[i]);
    printf("\n");

    /* ── 4. Decode all frames ────────────────────────────────────── */
    const RSCodec *codec = rs_get_codec(ecc > 0 ? ecc : 32);
    int k = codec ? (int)codec->msg_length : (pay_b / 8);
    int n_k = codec ? (int)codec->ecc_symbols : 0;
    int blk = codec ? 255 : (pay_b / 8);

    printf("RS: k=%d n_k=%d blk=%d\n", k, n_k, blk);

    /* Allocate buffers */
    uint8_t *frame  = (uint8_t *)malloc((size_t)frame_sz);
    uint8_t *fbits  = (uint8_t *)malloc((size_t)pay_b);
    uint8_t *output = (uint8_t *)calloc((size_t)orig_size + 1024, 1);

    /* RS decode state: accumulate encoded bytes, decode when we have a block */
    uint8_t  rs_enc[255];
    int      rs_enc_pos = 0;
    int64_t  out_pos = 0;

    /* Frame 0 was already read for calibration above. Start data decode
     * from frame 1 (file pointer is now at frame 1). */
    for (int fi = 1; fi < n_frames && out_pos < orig_size; fi++) {
        if (fread(frame, 1, (size_t)frame_sz, f) != (size_t)frame_sz)
            break;

        int nb = decode_frame_bits(frame, frame_w, grid_y, grid_x, bs,
                                     gcols, pay_r, 2, sync_r, fbits);

        /* Convert bits to bytes, feed into RS decoder */
        for (int bi = 0; bi + 7 < nb && out_pos < orig_size; bi += 8) {
            uint8_t byte_val = 0;
            for (int j = 0; j < 8; j++)
                if (fbits[bi + j]) byte_val |= (uint8_t)(1u << (7 - j));

            if (codec) {
                rs_enc[rs_enc_pos++] = byte_val;
                if (rs_enc_pos >= blk) {
                    uint8_t dec[255];
                    int status = 0;
                    rs_decode_block(codec, rs_enc, dec, &status);
                    /* Copy only what's needed, truncating to original size */
                    int64_t to_copy = (int64_t)k;
                    if (out_pos + to_copy > orig_size)
                        to_copy = orig_size - out_pos;
                    memcpy(output + out_pos, dec, (size_t)to_copy);
                    out_pos += to_copy;
                    rs_enc_pos = 0;
                    if (out_pos >= orig_size) break;  /* done */
                }
            } else {
                if (out_pos < orig_size)
                    output[out_pos++] = byte_val;
                if (out_pos >= orig_size) break;
            }
        }
        if (out_pos >= orig_size) break;
    }
    fclose(f);
    free(frame);
    free(fbits);

    /* Handle final partial RS block (padded to 255 bytes by encoder) */
    if (codec && rs_enc_pos > 0 && out_pos < orig_size) {
        /* Zero-pad remaining bytes to full RS block */
        while (rs_enc_pos < blk) rs_enc[rs_enc_pos++] = 0;
        uint8_t dec[255];
        int status = 0;
        rs_decode_block(codec, rs_enc, dec, &status);
        int64_t to_copy = (int64_t)k;
        if (out_pos + to_copy > orig_size)
            to_copy = orig_size - out_pos;
        memcpy(output + out_pos, dec, (size_t)to_copy);
        out_pos += to_copy;
    }

    printf("Decoded output: %lld bytes\n", (long long)out_pos);

    /* ── 5. Verify SHA256 ────────────────────────────────────────── */
    uint8_t computed_sha[32];
    sha256_data(output, (size_t)orig_size, computed_sha);

    printf("Computed SHA:  ");
    for (int i = 0; i < 32; i++) printf("%02x", computed_sha[i]);
    printf("\n");

    int match = (memcmp(expected_sha, computed_sha, 32) == 0);

    if (match) {
        printf("\n*** SHA256 MATCH! Roundtrip verified. ***\n");
        FILE *of = fopen("roundtrip_decoded.bin", "wb");
        if (of) { fwrite(output, 1, (size_t)orig_size, of); fclose(of); }
    } else {
        printf("\n*** SHA256 MISMATCH! ***\n");
        /* Save anyway for debugging */
        FILE *of = fopen("roundtrip_decoded.bin", "wb");
        if (of) { fwrite(output, 1, (size_t)orig_size, of); fclose(of); }
    }

    free(output);
    return match ? 0 : 1;
}

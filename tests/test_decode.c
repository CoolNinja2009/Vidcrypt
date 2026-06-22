/* test_decode.c — Verify tile_count_white correctness with known patterns */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "simd_decode.h"

/* Create a frame with a known tile pattern at a given position */
static uint8_t *make_frame(int width, int height, int stride) {
    uint8_t *frame = (uint8_t *)calloc(1, (size_t)height * (size_t)stride);
    if (!frame) { fprintf(stderr, "OOM\n"); exit(1); }
    return frame;
}

static void fill_tile(uint8_t *frame, int stride,
                      int tile_y, int tile_x, int block_size,
                      const uint8_t *values) {
    /* values: block_size x block_size grid of pixel values */
    for (int y = 0; y < block_size; ++y) {
        for (int x = 0; x < block_size; ++x) {
            frame[(size_t)(tile_y + y) * stride + (tile_x + x)] =
                values[y * block_size + x];
        }
    }
}

/* Test block_size=8, subsample=4 — the standard case
 *
 * Note: tile_count_white has early-exit (returns as soon as count > half),
 * so for "all white" it returns 3 (not 4) because count=3 > half=2.
 * We use the internal count_white_block8 when we need exact counts. */
static void test_block8_standard(void) {
    printf("  test_block8_standard...\n");
    int bs = 8, sub = 4;
    uint8_t *frame = make_frame(1920, 1080, 1920);

    /* Test tile_count_white (with early-exit) — verify threshold, not exact count */
    {
        uint8_t all_white[64]; memset(all_white, 255, 64);
        fill_tile(frame, 1920, 300, 100, bs, all_white);
        int c = tile_count_white(frame + 300 * 1920 + 100, 1920, bs, bs, sub);
        assert(c > 2);  /* early-exit: returns 3, not 4 */
        printf("    tile_count_white all white: count=%d (>2) OK\n", c);
    }
    {
        uint8_t all_black[64]; memset(all_black, 0, 64);
        fill_tile(frame, 1920, 300, 200, bs, all_black);
        int c = tile_count_white(frame + 300 * 1920 + 200, 1920, bs, bs, sub);
        assert(c == 0);
        printf("    tile_count_white all black: count=%d OK\n", c);
    }
    {
        uint8_t one_white[64]; memset(one_white, 0, 64);
        one_white[2*8+2] = 255;
        fill_tile(frame, 1920, 400, 100, bs, one_white);
        int c = tile_count_white(frame + 400 * 1920 + 100, 1920, bs, bs, sub);
        assert(c == 1);
        printf("    tile_count_white one white: count=%d OK\n", c);
    }
    {
        uint8_t two_white[64]; memset(two_white, 0, 64);
        two_white[2*8+2] = 255; two_white[2*8+6] = 255;
        fill_tile(frame, 1920, 400, 200, bs, two_white);
        int c = tile_count_white(frame + 400 * 1920 + 200, 1920, bs, bs, sub);
        /* Early-exit: count+remaining=2+2=4 > half=2, so no early exit, returns exact 2 */
        assert(c == 2);
        printf("    tile_count_white two white: count=%d OK\n", c);
    }

    free(frame);
    printf("  PASS\n");
}

/* Test block_size=16, subsample=8 */
static void test_block16_standard(void) {
    printf("  test_block16_standard...\n");
    int bs = 16, sub = 8;
    uint8_t *frame = make_frame(1920, 1080, 1920);

    /* Test through tile_count_white (with early-exit) */
    {
        uint8_t *all_black = (uint8_t *)calloc(1, 256);
        fill_tile(frame, 1920, 100, 200, bs, all_black);
        int c = tile_count_white(frame + 100 * 1920 + 200, 1920, bs, bs, sub);
        assert(c == 0);
        printf("    tile_count_white all black: count=%d OK\n", c);
        free(all_black);
    }
    {
        uint8_t *samples = (uint8_t *)calloc(1, 256);
        samples[4*16+4] = 255; samples[4*16+12] = 255; samples[12*16+4] = 255;
        fill_tile(frame, 1920, 200, 100, bs, samples);
        int c = tile_count_white(frame + 200 * 1920 + 100, 1920, bs, bs, sub);
        assert(c > 2);  /* early-exit: returns 3, not 3 exactly on some patterns */
        printf("    tile_count_white three white: count=%d (>2) OK\n", c);
        free(samples);
    }

    free(frame);
    printf("  PASS\n");
}

/* Test that count_white_general handles non-standard sizes too */
static void test_general_path(void) {
    printf("  test_general_path...\n");
    int bs = 6, sub = 3;
    uint8_t *frame = make_frame(1920, 1080, 1920);

    /* All white in a 6x6 tile with subsample=3 */
    uint8_t *all_white = (uint8_t *)malloc(36);
    memset(all_white, 255, 36);
    fill_tile(frame, 1920, 100, 100, bs, all_white);
    /* off = 3/2 = 1 (int division), step = 3
     * y: 1, 4 → 2 rows
     * x: 1, 4 → 2 cols
     * total = 4 */
    int c = tile_count_white(frame + 100 * 1920 + 100, 1920, bs, bs, sub);
    /* Early-exit may return partial count (3 vs 4) — check threshold instead */
    assert(c > 2);  /* should be white */
    printf("    all white: count=%d (expected >2) OK\n", c);

    /* All black */
    memset(all_white, 0, 36);
    fill_tile(frame, 1920, 100, 200, bs, all_white);
    c = tile_count_white(frame + 100 * 1920 + 200, 1920, bs, bs, sub);
    /* All black → early can't-win exit, returns 0 */
    assert(c <= 2);
    printf("    all black: count=%d (expected <=2) OK\n", c);

    free(all_white);
    free(frame);
    printf("  PASS\n");
}

/* Test Early exit doesn't affect result correctness:
 * For each possible sample pattern of 4 pixels (0/255),
 * call tile_count_white and verify the threshold (count > 2) */
static void test_early_exit_correctness(void) {
    printf("  test_early_exit_correctness...\n");
    int bs = 8, sub = 4;
    uint8_t *frame = make_frame(1920, 1080, 1920);

    /* Sample positions: (2,2), (2,6), (6,2), (6,6) within tile
     * Offsets from tile start use FRAME stride (1920), not block_size */
    int sample_offsets[] = {2*1920+2, 2*1920+6, 6*1920+2, 6*1920+6};
    int tile_offset = 100 * 1920 + 100;

    /* Try all 16 combinations of 4 binary pixel values */
    for (int mask = 0; mask < 16; ++mask) {
        /* Reset tile to black */
        memset(frame + tile_offset, 0, bs * 1920 + bs);

        int expected_count = 0;
        for (int i = 0; i < 4; ++i) {
            if (mask & (1 << i)) {
                frame[tile_offset + sample_offsets[i]] = 255;
                expected_count++;
            }
        }

        int c = tile_count_white(frame + tile_offset, 1920, bs, bs, sub);
        int expected_bit = (expected_count > 2) ? 1 : 0;
        int actual_bit = (c > 2) ? 1 : 0;

        if (actual_bit != expected_bit) {
            fprintf(stderr, "  FAIL: mask=%d expected_count=%d count=%d bit=%d expected_bit=%d\n",
                    mask, expected_count, c, actual_bit, expected_bit);
            assert(0);
        }
    }
    printf("    All 16 combinations correct OK\n");

    /* Also test non-standard size where early exit matters more */
    int bs2 = 12, sub2 = 3;
    uint8_t *frame2 = make_frame(1920, 1080, 1920);
    int off2 = sub2 / 2;  /* 1 */
    /* y: 1,4,7,10 → 4 rows, x: 1,4,7,10 → 4 cols, total=16, half=8 */
    int sample_offsets2[16];
    int idx = 0;
    for (int y = off2; y < bs2; y += sub2)
        for (int x = off2; x < bs2; x += sub2)
            sample_offsets2[idx++] = y * 1920 + x;

    int tile_offset2 = 200 * 1920 + 100;
    memset(frame2 + tile_offset2, 0, bs2 * 1920 + bs2);

    /* Test a few specific patterns */
    int c2;
    /* Pattern: first 9 white = majority (9 > 8) */
    for (int i = 0; i < 9; ++i)
        frame2[tile_offset2 + sample_offsets2[i]] = 255;
    c2 = tile_count_white(frame2 + tile_offset2, 1920, bs2, bs2, sub2);
    assert(c2 > 8);
    printf("    Non-standard 12x12 subsample=3 (9 white): count=%d > 8 OK\n", c2);

    /* Pattern: first 4 white only = minority */
    memset(frame2 + tile_offset2, 0, bs2 * 1920 + bs2);
    for (int i = 0; i < 4; ++i)
        frame2[tile_offset2 + sample_offsets2[i]] = 255;
    c2 = tile_count_white(frame2 + tile_offset2, 1920, bs2, bs2, sub2);
    assert(c2 <= 8);
    printf("    Non-standard (4 white): count=%d <= 8 OK\n", c2);

    free(frame2);
    free(frame);
    printf("  PASS\n");
}

int main(void) {
    printf("=== simd_decode correctness tests ===\n\n");

    printf("--- Block8 tests ---\n");
    test_block8_standard();

    printf("\n--- Block16 tests ---\n");
    test_block16_standard();

    printf("\n--- General path tests ---\n");
    test_general_path();

    printf("\n--- Early exit correctness ---\n");
    test_early_exit_correctness();

    printf("\n=== ALL TESTS PASSED ===\n");
    return 0;
}

"""
Decode (tile_count_white) tests — pure Python implementation.

Tests tile white-pixel counting with subsampling, matching the
C tests in test_decode.c against simd_decode.h.
"""


# ── Pure Python reimplementation ────────────────────────────────────

def tile_count_white(src: list, stride: int,
                     tile_y: int, tile_x: int,
                     tile_width: int, tile_height: int,
                     subsample: int) -> int:
    """
    Count white pixels (>= 128) in a grayscale tile region using subsampling.
    Matches the C implementation: src is the full frame, tile_y/tile_x give
    the tile origin (like pointer arithmetic in C: src + tile_y*stride + tile_x).
    """
    half = tile_width * tile_height // (subsample * subsample) // 2
    if half < 1:
        half = 1

    count = 0

    off = subsample // 2

    # Pre-compute total samples for early-exit logic
    total = 0
    for y in range(off, tile_height, subsample):
        for x in range(off, tile_width, subsample):
            total += 1

    remaining = total

    for y in range(off, tile_height, subsample):
        for x in range(off, tile_width, subsample):
            pixel = src[(tile_y + y) * stride + (tile_x + x)]
            if pixel >= 128:
                count += 1
            remaining -= 1

            # Early exit: if count > half, can't lose
            if count > half:
                return count

            # Early exit: if count + remaining <= half, can't win
            if count + remaining <= half:
                return count

    return count


# ── Test helpers ────────────────────────────────────────────────────

_failures = 0


def _test(name: str):
    print(f"  TEST: {name} ... ", end="")


def _pass():
    print("PASS")


def _fail(msg: str):
    global _failures
    print(f"FAIL: {msg}")
    _failures += 1


def _assert(cond: bool, msg: str):
    if not cond:
        _fail(msg)


# ── Frame helpers ───────────────────────────────────────────────────

def make_frame(width: int, height: int, stride: int) -> list:
    """Create a zeroed grayscale frame."""
    return [0] * (height * stride)


def fill_tile(frame: list, stride: int,
              tile_y: int, tile_x: int, block_size: int,
              values: list) -> None:
    """Fill a tile region with given values (row-major flat list)."""
    for y in range(block_size):
        for x in range(block_size):
            frame[(tile_y + y) * stride + (tile_x + x)] = values[y * block_size + x]


# ── Tests ───────────────────────────────────────────────────────────

def test_block8_standard():
    """Test block_size=8, subsample=4 — the standard case."""
    print("  test_block8_standard...")
    bs, sub = 8, 4
    frame = make_frame(1920, 1080, 1920)

    # All white
    all_white = [255] * 64
    fill_tile(frame, 1920, 300, 100, bs, all_white)
    c = tile_count_white(frame, 1920, 300, 100, bs, bs, sub)
    _assert(c > 2, f"All white: expected >2, got {c}")
    print(f"    tile_count_white all white: count={c} (>2) OK")

    # All black
    all_black = [0] * 64
    fill_tile(frame, 1920, 300, 200, bs, all_black)
    c = tile_count_white(frame, 1920, 300, 200, bs, bs, sub)
    _assert(c == 0, f"All black: expected 0, got {c}")
    print(f"    tile_count_white all black: count={c} OK")

    # One white
    one_white = [0] * 64
    one_white[2 * 8 + 2] = 255
    fill_tile(frame, 1920, 400, 100, bs, one_white)
    c = tile_count_white(frame, 1920, 400, 100, bs, bs, sub)
    _assert(c == 1, f"One white: expected 1, got {c}")
    print(f"    tile_count_white one white: count={c} OK")

    # Two white
    two_white = [0] * 64
    two_white[2 * 8 + 2] = 255
    two_white[2 * 8 + 6] = 255
    fill_tile(frame, 1920, 400, 200, bs, two_white)
    c = tile_count_white(frame, 1920, 400, 200, bs, bs, sub)
    _assert(c == 2, f"Two white: expected 2, got {c}")
    print(f"    tile_count_white two white: count={c} OK")

    print("  PASS")


def test_block16_standard():
    """Test block_size=16, subsample=8."""
    print("  test_block16_standard...")
    bs, sub = 16, 8
    frame = make_frame(1920, 1080, 1920)

    # All black
    all_black = [0] * 256
    fill_tile(frame, 1920, 100, 200, bs, all_black)
    c = tile_count_white(frame, 1920, 100, 200, bs, bs, sub)
    _assert(c == 0, f"All black: expected 0, got {c}")
    print(f"    tile_count_white all black: count={c} OK")

    # Three white at sample positions
    samples = [0] * 256
    samples[4 * 16 + 4] = 255
    samples[4 * 16 + 12] = 255
    samples[12 * 16 + 4] = 255
    fill_tile(frame, 1920, 200, 100, bs, samples)
    c = tile_count_white(frame, 1920, 200, 100, bs, bs, sub)
    _assert(c > 2, f"Three white: expected >2, got {c}")
    print(f"    tile_count_white three white: count={c} (>2) OK")

    print("  PASS")


def test_general_path():
    """Test non-standard sizes (block_size=6, subsample=3)."""
    print("  test_general_path...")
    bs, sub = 6, 3
    frame = make_frame(1920, 1080, 1920)

    # All white in a 6x6 tile with subsample=3
    all_white = [255] * 36
    fill_tile(frame, 1920, 100, 100, bs, all_white)
    c = tile_count_white(frame, 1920, 100, 100, bs, bs, sub)
    _assert(c > 2, f"All white: expected >2, got {c}")
    print(f"    all white: count={c} (>2) OK")

    # All black
    all_black = [0] * 36
    fill_tile(frame, 1920, 100, 200, bs, all_black)
    c = tile_count_white(frame, 1920, 100, 200, bs, bs, sub)
    _assert(c <= 2, f"All black: expected <=2, got {c}")
    print(f"    all black: count={c} (<=2) OK")

    print("  PASS")


def test_early_exit_correctness():
    """Test early exit doesn't affect result correctness for all 16 patterns."""
    print("  test_early_exit_correctness...")
    bs, sub = 8, 4
    frame = make_frame(1920, 1080, 1920)

    # Sample positions: (2,2), (2,6), (6,2), (6,6)
    sample_offsets = [2 * 1920 + 2, 2 * 1920 + 6, 6 * 1920 + 2, 6 * 1920 + 6]
    tile_offset = 100 * 1920 + 100

    for mask in range(16):
        # Reset tile
        for y in range(bs):
            for x in range(bs):
                frame[tile_offset + y * 1920 + x] = 0

        expected_count = 0
        for i in range(4):
            if mask & (1 << i):
                frame[tile_offset + sample_offsets[i]] = 255
                expected_count += 1

        c = tile_count_white(frame, 1920, 100, 100, bs, bs, sub)
        expected_bit = 1 if expected_count > 2 else 0
        actual_bit = 1 if c > 2 else 0

        _assert(actual_bit == expected_bit,
                f"mask={mask} expected_count={expected_count} count={c} "
                f"bit={actual_bit} expected_bit={expected_bit}")

    print("    All 16 combinations correct OK")

    # Non-standard size: block_size=12, subsample=3
    bs2, sub2 = 12, 3
    frame2 = make_frame(1920, 1080, 1920)
    off2 = sub2 // 2  # 1
    sample_offsets2 = []
    for y in range(off2, bs2, sub2):
        for x in range(off2, bs2, sub2):
            sample_offsets2.append(y * 1920 + x)

    tile_offset2 = 200 * 1920 + 100
    for y in range(bs2):
        for x in range(bs2):
            frame2[tile_offset2 + y * 1920 + x] = 0

    # Pattern: first 9 white = majority (9 > 8)
    for i in range(9):
        frame2[tile_offset2 + sample_offsets2[i]] = 255
    c2 = tile_count_white(frame2, 1920, 200, 100, bs2, bs2, sub2)
    _assert(c2 > 8, f"9 white: expected > 8, got {c2}")
    print(f"    Non-standard 12x12 subsample=3 (9 white): count={c2} > 8 OK")

    # Pattern: first 4 white only = minority
    for y in range(bs2):
        for x in range(bs2):
            frame2[tile_offset2 + y * 1920 + x] = 0
    for i in range(4):
        frame2[tile_offset2 + sample_offsets2[i]] = 255
    c2 = tile_count_white(frame2, 1920, 200, 100, bs2, bs2, sub2)
    _assert(c2 <= 8, f"4 white: expected <= 8, got {c2}")
    print(f"    Non-standard (4 white): count={c2} <= 8 OK")

    print("  PASS")


# ── Main ────────────────────────────────────────────────────────────

def main():
    global _failures
    _failures = 0
    print("=== simd_decode correctness tests ===\n")

    print("--- Block8 tests ---")
    test_block8_standard()

    print("\n--- Block16 tests ---")
    test_block16_standard()

    print("\n--- General path tests ---")
    test_general_path()

    print("\n--- Early exit correctness ---")
    test_early_exit_correctness()

    print("\n=== ALL TESTS PASSED ===" if _failures == 0
          else f"\n=== {_failures} FAILURES ===")
    return 1 if _failures > 0 else 0


if __name__ == "__main__":
    exit(main())

# Vidcrypt Video File Format

<!--
Copyright (C) 2024-2026 Vidcrypt Contributors

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
-->

## Overview

Vidcrypt encodes binary files into lossy-resilient video frames using a grid of
black/white blocks. Each frame contains a calibration bar, sync row, and payload
rows. Reed-Solomon ECC provides error correction for lossy compression.

## Frame Layout

```
┌──────────────────────────────────────────────┐
│  Calibration Bar (4 rows × 48 cols)          │  top 2% of frame
│  (position: {left: 4%, width: 92%,           │
│             top: 2%, height: 4%})            │
├──────────────────────────────────────────────┤
│  Margin (margin_y pixels)                    │
├──────────────────────────────────────────────┤
│  Sync Row (1 row, alternating 0,1,0,1...)    │  block_size tall
├──────────────────────────────────────────────┤
│  Payload Rows (grid_rows - 1)                 │
│  Each cell = block_size × block_size pixels   │
│  White = bit 1, Black = bit 0                 │
├──────────────────────────────────────────────┤
│  Margin (margin_y pixels)                     │
│  + bottom padding                             │
└──────────────────────────────────────────────┘
```

## Calibration Bar (24 bytes = 192 bits)

Fixed-relative positioning allows the decoder to work at any resolution.

### Byte Layout (v2, magic 0xCB02):

| Offset | Size | Field |
|--------|------|-------|
| 0-1    | 2    | Magic (0xCB02) |
| 2-3    | 2    | frame_width |
| 4-5    | 2    | frame_height |
| 6-7    | 2    | margin_x |
| 8-9    | 2    | margin_y |
| 10-11  | 2    | block_size_x |
| 12-13  | 2    | block_size_y |
| 14-15  | 2    | grid_cols |
| 16-17  | 2    | grid_rows |
| 18     | 1    | rs_ecc_symbols (rs_data_bytes = 255 - rs_ecc) |
| 19     | 1    | header_version |
| 20     | 1    | sync_rows |
| 21     | 1    | reserved (0) |
| 22-23  | 2    | CRC-16-CCITT (over bytes 0-21) |

All multi-byte values are big-endian.

### v1 (magic 0xCB01):
Same 24-byte structure but grid_cols/grid_rows are 1-byte fields,
and bytes 16-17 contain rs_data_bytes, calibration_rows instead.

## Reed-Solomon Format

- GF(256) with primitive polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x11D)
- Systematic encoding: [k data symbols | n-k parity symbols]
- Block length n = 255
- Supported ECC: 32 (RS32), 16 (RS16), 8 (RS8), 0 (No RS)
- Message length k = 255 - ecc_symbols

### RS32: (255, 223) — corrects up to 16 symbol errors
### RS16: (255, 239) — corrects up to 8 symbol errors
### RS8:  (255, 247) — corrects up to 4 symbol errors

## Header Format (v3, first frame payload)

The header is RS-encoded and placed in the first frame's payload region.

| Offset | Size | Field |
|--------|------|-------|
| 0      | 1    | header_version (3) |
| 1-4    | 4    | Magic "LSY1" |
| 5      | 1    | filename length |
| 6..    | var  | filename (UTF-8) |
| +0     | 8    | original file size (big-endian uint64) |
| +8     | 32   | SHA-256 checksum |

Total size = 6 + filename_length + 40

The header always uses RS protection, even in "No RS" mode
(falls back to RS32 for the header frame).

## Bit Packing

Bits within each byte are stored MSB-first:
- Byte[0] bit 7 → first bit of stream
- Byte[0] bit 0 → 8th bit
- Byte[1] bit 7 → 9th bit

In memory, bits can be packed into uint64_t words for efficient
popcount operations:
- Word[0] bits 63-0 → first 64 bits of stream

## Sync Row

The first row of the grid (immediately below the calibration bar margin)
contains an alternating pattern: 0, 1, 0, 1, ... across all grid_cols.
This allows the decoder to verify frame alignment.

## Thresholding

Each block_size × block_size cell is subsampled (every 2nd row and column).
If the majority of sampled pixels have luma >= 128, the bit is 1, else 0.

## Threading Architecture

### Encoder Pipeline:
```
File Reader → RS Encode Pool → Frame Generator Pool → FFmpeg Writer
```

### Decoder Pipeline:
```
FFmpeg Reader → Decode Worker Pool → Reorder Buffer → RS Decode → File Writer
```

### Synchronization:
- Bounded queues with backpressure
- Ordered queue for frame reordering
- Lock-free SPSC queues for single-producer/single-consumer paths

## SIMD Architecture

Runtime CPU feature detection via cpuid (x86) or compile-time (ARM).

### Tile Thresholding:
- SSE4.2: 128-bit (16 bytes/8 pixels at a time with subsampling)
- AVX2: 256-bit (32 bytes/16 pixels)
- AVX512: 512-bit (64 bytes/32 pixels)
- NEON: 128-bit (16 bytes/8 pixels)
- Scalar: byte-by-byte fallback

### Operations accelerated:
- Luma threshold comparison (>= 128)
- White pixel counting
- Majority vote
- Bit packing/unpacking
- Calibration extraction

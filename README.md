# Vidcrypt

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

Vidcrypt is a **video-based data steganography and archival tool** written in C. It converts arbitrary binary files into synthetic video frames using a grid of black/white blocks, protected by **Reed-Solomon forward-error correction** and verified with **SHA-256 checksums**.

The encoded data survives lossy video compression (with sufficient ECC), making it suitable for archival on video platforms, cloud storage, or distribution through video pipelines. An optional **CUDA GPU backend** accelerates encode and decode using NVDEC/NVENC hardware.

---

## Features

- **Encode** any binary file (ZIP, PDF, EXE, etc.) into a video of black/white block patterns
- **Decode** video back to the original file with SHA-256 verification
- **Reed-Solomon ECC** — corrects up to 16 symbol errors per 255-byte block (RS32)
- **CUDA GPU Acceleration** — NVDEC decode + GPU grayscale conversion + NVENC encode
- **Multi-threaded** — thread pool for parallel frame processing
- **SIMD-friendly** — SSE4.2, AVX2, AVX512, NEON scalar fallback for pixel thresholding
- **Lossy-tolerant** — RS ECC compensates for compression artifacts
- **Self-describing calibration bar** — frame geometry encoded in every frame

---

## Repository Layout

```
.
├── CMakeLists.txt          # Build system (CMake 3.16+)
├── LICENSE                 # GNU General Public License v3.0
├── README.md               # This file
├── main_encoder.c          # vidcrypt-encoder CLI entrypoint
├── main_decoder.c          # vidcrypt-decoder CLI entrypoint
├── src/
│   ├── backend.c           # GPU/CPU dispatch, hybrid pipeline
│   ├── backend.h
│   ├── calibration.c/h     # Calibration bar read/write, frame geometry
│   ├── bitstream.c/h       # Bit-level I/O utilities
│   ├── crc16.c/h           # CRC-16-CCITT checksum
│   ├── decoder.c/h         # CPU decoder pipeline
│   ├── encoder.c/h         # CPU encoder pipeline
│   ├── ffmpeg_pipe.c/h     # FFmpeg subprocess I/O
│   ├── framedecode.c/h     # Frame decode geometry + tile extraction
│   ├── framegen.c/h        # Frame generation (template + block fill)
│   ├── gpu_backend.c/h     # GPU backend: memory pool, streams, dispatch
│   ├── gpu_kernels.cu/h    # CUDA kernels: nv12_to_gray, frame_generate, etc.
│   ├── gpu_nvdec.c/h       # NVDEC hardware decoder wrapper
│   ├── gpu_nvenc.c/h       # NVENC hardware encoder wrapper
│   ├── gpu_presets.c/h     # Per-architecture GPU tuning presets
│   ├── profiling.c/h       # Instrumentation (compile-time optional)
│   ├── reedsolomon.c/h     # RS(255,k) encoder/decoder
│   ├── sha256.c/h          # SHA-256 file checksum
│   ├── simd_decode.c/h     # SIMD tile thresholding (SSE/AVX/NEON)
│   └── threadpool.c/h      # Worker thread pool
├── tests/
│   ├── test_bitstream.c
│   ├── test_crc16.c
│   └── test_reedsolomon.c
├── docs/
│   ├── FORMAT.md           # Full spec: calibration, sync, payload, RS
│   └── GPU_ACCELERATION.md # GPU architecture design doc
└── builder/                # Windows CMake helper scripts
```

---

## Requirements

### Minimum (CPU-only)
- CMake 3.16+
- C17 compiler (MSVC 19.44+, GCC, Clang)
- pthreads (Windows: via MSVC or MinGW)
- FFmpeg + ffprobe (in PATH) for video I/O

### Optional (GPU acceleration)
- NVIDIA GPU with Compute Capability 7.5+ (Turing/RTX 20 or newer)
- CUDA 12.6+
- NVIDIA Video Codec SDK 13.1.15+
- Visual Studio 2022 Build Tools (Windows) or GCC/Clang (Linux)

---

## Build

### CPU-only build (default, works everywhere)

```sh
mkdir -p build && cd build
cmake ..
cmake --build .
```

### GPU-accelerated build

```sh
mkdir -p build_cuda && cd build_cuda
cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

### With profiling instrumentation

```sh
cmake .. -DUSE_CUDA=ON -DENABLE_PROFILING=ON
cmake --build .
```

### Build variants

| Build Type | Command | Use Case |
|-----------|---------|----------|
| CPU-only | `cmake -DUSE_CUDA=OFF ..` | CI/CD, VMs, any system |
| GPU production | `cmake -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release ..` | End-user with NVIDIA GPU |
| GPU debug | `cmake -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Debug ..` | Development, kernel debugging |
| Profiling | `cmake -DUSE_CUDA=ON -DENABLE_PROFILING=ON ..` | Performance analysis |

---

## Usage

### Encode a file

```sh
./vidcrypt-encoder -i <input-file> -o <output-video> [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input <path>` | Input file (required) | — |
| `-o, --output <path>` | Output video (required) | — |
| `-B, --backend <mode>` | Backend: `cpu`, `gpu`, `auto` | `auto` |
| `-W, --width <px>` | Frame width | 1920 |
| `-H, --height <px>` | Frame height | 1080 |
| `-m, --margin <px>` | Margin size | 96 |
| `-b, --block-size <px>` | Block size | 24 |
| `-r, --rs <n>` | RS ECC symbols: 32, 16, 8, 0 | 32 |
| `-f, --fps <n>` | Frames per second | 15.0 |
| `-j, --workers <n>` | Worker threads | auto |
| `-c, --codec <name>` | FFmpeg codec (ffv1, libx264, etc.) | ffv1 |
| `-h, --help` | Show help | — |

**Examples:**

```sh
# Basic encode with defaults (1920x1080, CPU)
vidcrypt-encoder -i document.pdf -o encoded.avi

# GPU-accelerated encode
vidcrypt-encoder -i archive.zip -o archive.avi -B gpu

# Small frames, faster encode
vidcrypt-encoder -i data.bin -o output.avi -W 640 -H 480 -b 16 -f 30

# Lossy video codec with extra ECC protection
vidcrypt-encoder -i important.bin -o output.mkv -c libx264 -r 32
```

### Decode a video

```sh
./vidcrypt-decoder -i <input-video> [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input <path>` | Input video (required) | — |
| `-o, --output-dir <dir>` | Output directory | current dir |
| `-B, --backend <mode>` | Backend: `cpu`, `gpu`, `auto` | `auto` |
| `-j, --workers <n>` | Worker threads | auto |
| `-h, --help` | Show help | — |

**Examples:**

```sh
# Decode to current directory (original filename is restored)
vidcrypt-decoder -i encoded.avi

# GPU-accelerated decode
vidcrypt-decoder -i encoded.avi -B gpu

# Specify output directory
vidcrypt-decoder -i archive.avi -o ./recovered -j 8
```

The decoder automatically extracts the original filename from the embedded header and writes the recovered file using that name.

### Backend selection

| Mode | Behavior |
|------|----------|
| `auto` (default) | Probes for NVIDIA GPU; uses GPU if available, falls back to CPU |
| `gpu` | Requires NVIDIA GPU with CUDA; fails if unavailable |
| `cpu` | Pure CPU pipeline, no GPU needed |

### Run tests

```sh
cd build_cuda
ctest --output-on-failure
```

The project includes unit tests for CRC16, bitstream operations, and Reed-Solomon encode/decode.

---

## Format Overview

Vidcrypt stores data as a grid of black-and-white blocks in synthetic video frames. Each frame contains:

```
┌──────────────────────────────────────────────┐
│  Calibration Bar (4×48 dots, relative %)     │
├──────────────────────────────────────────────┤
│  Sync Row (alternating 0,1,0,1... pattern)   │
├──────────────────────────────────────────────┤
│  Payload Rows (grid of black/white blocks)   │
│  White = bit 1, Black = bit 0                │
└──────────────────────────────────────────────┘
```

- The **calibration bar** encodes all frame geometry (resolution, margins, block size, RS parameters) in 24 bytes with CRC-16, located at a fixed relative position (2-6% from top, 4-96% width). This means the decoder can work at any resolution.
- The **sync row** validates frame alignment.
- **Payload** bits are thresholded at luma >= 128, with 2× subsampling.
- **Reed-Solomon** GF(256) blocks: RS32 (255,223), RS16 (255,239), RS8 (255,247), or no RS.
- **SHA-256** checksum verifies the decoded output against the original.

See [`docs/FORMAT.md`](docs/FORMAT.md) for the full specification.

---

## GPU Acceleration

The CUDA backend accelerates three pipeline stages:

| Stage | Speedup | Details |
|-------|---------|---------|
| **Decode: NVDEC** | ~2-5× over CPU ffmpeg | Hardware H.264/HEVC decode direct to GPU memory |
| **Decode: nv12_to_gray** | ~10-20× | Converts NV12 Y-plane to planar grayscale on GPU |
| **Encode: NVENC** | ~10-30× over CPU ffmpeg | Hardware H.264/HEVC/AV1 encode from GPU memory |

Supported GPUs: Maxwell (sm_50) through Blackwell (sm_120), with per-architecture tuning presets.

See [`docs/GPU_ACCELERATION.md`](docs/GPU_ACCELERATION.md) for the full architecture design.

---

## How It Works

1. **Encoding:** 
   Input file → RS encode (CPU thread pool) → convert to bits → generate frame (CPU or CUDA kernel) → render calibration bar + sync row → write video via FFmpeg pipe or NVENC

2. **Decoding:**
   Video → read frames (FFmpeg pipe or NVDEC) → calibrate (extract frame geometry) → grayscale conversion → tile thresholding (SIMD or CUDA) → RS decode (CPU) → SHA-256 verify → reconstructed file

3. **Error correction:**
   Reed-Solomon (255, k) corrects up to `(255-k)/2` symbol errors per block, making Vidcrypt robust against lossy video compression artifacts.

---

## License

Copyright (C) 2024-2026 Vidcrypt Contributors

This program is free software: you can redistribute it and/or modify it under the terms of the **GNU General Public License** as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the [GNU General Public License](LICENSE) for more details.
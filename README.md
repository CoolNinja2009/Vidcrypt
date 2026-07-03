# VIDCRYPT-V8

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

VIDCRYPT-V8 is a high-performance file-to-video encoding system designed to preserve arbitrary files through aggressive video compression.

Instead of storing data in traditional containers, VIDCRYPT converts files into grids of black-and-white tiles embedded inside video frames. Each frame contains payload data, synchronization markers, calibration metadata, and error-correction information, allowing files to survive transcoding, resizing, and heavy compression on platforms such as YouTube, Instagram, Discord, and other video-sharing services.

---

## Features

* File ↔ Video conversion
* Designed for compression-heavy platforms
* Reed-Solomon error correction
* SHA-256 integrity verification
* Automatic calibration and synchronization recovery
* Multi-threaded encoding and decoding
* **CUDA GPU acceleration** — 2378+ FPS decode (2.85× over CPU), with `h264_cuvid` hardware decode + CUDA tile extraction
* **Direct libavcodec C API** — eliminates ffmpeg CLI pipe overhead for GPU path
* **Async quad-buffered pipeline** — CPU frame decode overlaps with GPU upload + kernel + copyback
* **Cross-platform** — Windows, macOS, Linux (ARM64 scalar fallback on Apple Silicon)
* SIMD-accelerated calibration reader (AVX2, SSE4.2 on x86-64)

---

## How It Works

1. Files are converted into binary payloads.
2. Payload bits are mapped into visual tile grids.
3. Calibration and synchronization data are embedded into every frame.
4. Frames are assembled into a video stream.
5. The video can be uploaded, shared, compressed, or transcoded.
6. The decoder reconstructs the original file and verifies integrity using SHA-256.

---

## Performance

Measured on **Ryzen 7 7700X (8C/16T)** + **NVIDIA RTX 5070 (Blackwell)**, 100 MB payload, 1920×1080, block size 8, RS(255,223,32):

| Codec | Encode | CPU Decode | GPU Decode | GPU Decoder | Output Size |
|-------|--------|-----------|-----------|-------------|-------------|
| **FFV1** (lossless) | 51s / 803 FPS | 53s / 771 FPS | **18s / 2297 FPS** | sw ffv1 + CUDA tiles | 839 MB |
| **H.264** (CRF 10 ultrafast) | 80s / 513 FPS | 64s / 646 FPS | **36s / 1127 FPS** | **h264_cuvid (NVDEC)** | 2.4 GB |

GPU decode achieves **3.0× speedup** (FFV1) to **3.2×** (H.264 NVDEC) over CPU by:
1. **Direct libavcodec C API** — eliminates ffmpeg CLI pipe overhead
2. **Quad-buffered async pipeline** — 4 slots overlap CPU decode, GPU upload, kernel, D2H copyback
3. **CUDA `extract_bits` kernel** — massively parallel tile thresholding (one thread per block)
4. **NVDEC hardware decode** (H.264 only) — offloads video decode to dedicated GPU silicon

**YouTube workflow:** Encode to H.264 yuv420p (required for NVDEC), upload, download the re-encoded MP4, GPU-decode with `-b gpu`. RS ECC corrects any compression artifacts.
Actual performance depends on resolution, block size, codec settings, GPU model, and system hardware.

---

## Codec Guide

| Codec | Lossless | Encode Speed | GPU Decode | Best For |
|-------|----------|-------------|-----------|----------|
| **ffv1** (default) | Yes | Fast | CUDA tiles only | Local archiving, max fidelity |
| **ffvhuff** | Yes | Fast | CUDA tiles only | Same as ffv1 but 25× larger output |
| **h264** | Near (CRF 10) | Medium | **NVDEC hardware** | YouTube upload, NVDEC decode |
| **hevc** | Near (CRF 10) | Very slow | **NVDEC hardware** | Not recommended (encode too slow) |

> **NVDEC requires yuv420p pixel format.** The encoder automatically uses yuv420p output for h264/hevc codecs so NVDEC can hardware-decode them.


## Requirements

| Component | Windows | macOS | Linux |
|-----------|---------|-------|-------|
| **Compiler** | MSVC or MinGW/GCC | Clang (Xcode CLT) | GCC or Clang |
| **CMake** | 3.16+ | 3.16+ | 3.16+ |
| **FFmpeg** | FFmpeg + FFprobe in PATH | FFmpeg + FFprobe in PATH | FFmpeg + FFprobe in PATH |
| **CUDA GPU** | CUDA Toolkit 12.x + MSVC + Visual Studio | N/A | CUDA Toolkit 12.x + GCC |
| **NVDEC** | BtbN FFmpeg shared build in `ffmpeg_shared/` | FFmpeg dev (pkg-config) | FFmpeg dev (pkg-config) |
| **NVENC** | NVIDIA Video Codec SDK (optional) | N/A | NVIDIA Video Codec SDK (optional) |

---

## Build

### Auto-detect (recommended)

The build scripts auto-detect your platform, compiler, CUDA availability, and FFmpeg — then build with optimal settings.

**Windows:**
```batch
cd builder
build.bat                  :: auto-detect (CUDA if available, CPU fallback)
build.bat --gpu            :: force CUDA build (fails if unavailable)
build.bat --cpu --no-gpu   :: force CPU-only build
```

**macOS / Linux:**
```bash
cd builder
./build.sh                 # auto-detect everything
```

### Manual CMake

**Windows (CPU, any compiler):**
```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

**Windows (GPU, requires MSVC + CUDA):**
```bash
mkdir build_cuda && cd build_cuda
cmake .. -G "Visual Studio 17 2022" -A x64 -DUSE_CUDA=ON
cmake --build . --config Release
```

**macOS / Linux (CPU):**
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

**Linux (GPU):**
```bash
mkdir build_cuda && cd build_cuda
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON
cmake --build . --parallel
```

### CMake Options

| Option | Default | Description |
| ------ | ------- | ----------- |
| `USE_CUDA` | `OFF` | Enable CUDA GPU acceleration |
| `ENABLE_AVX2` | `ON` | Enable AVX2 SIMD for calibration reader |
| `ENABLE_PROFILING` | `OFF` | Enable per-stage profiling instrumentation |
| `ENABLE_LOGGING` | `OFF` | Enable detailed session logging to `log/` |
---

## Usage

### Encode a File

```bash
# Lossless (fastest encode, best for archiving)
vidcrypt-encoder -i input.zip -o encoded.mkv

# YouTube-optimized (H.264, NVDEC-compatible)
vidcrypt-encoder -i input.zip -o encoded.mkv -c h264
```

### Decode a Video (CPU)

```bash
vidcrypt-decoder -i encoded.mkv
```

### Decode with GPU Acceleration

```bash
vidcrypt-decoder -b gpu -i encoded.mkv
```

The `-b` / `--backend` flag accepts:
* `cpu` — Force CPU-only decode
* `gpu` — Force GPU-accelerated decode (fails if unavailable)
* `auto` — Auto-detect GPU, fall back to CPU

Additional options:
* `-o <dir>` — Output directory (default: current dir)
* `-j <n>` — Worker threads for CPU RS decode (default: 4)

The decoder automatically restores the original filename and verifies the recovered file before writing it to disk.

---

## Repository Structure

```text
.
├── src/                  Core encoder and decoder implementation
│   ├── backend.c/h       Backend mode selection (cpu/gpu/auto)
│   ├── decoder.c/h       Main decoder with async GPU pipeline
│   ├── encoder.c/h       Main encoder
│   ├── libav_decoder.c/h Direct libavcodec C API frame reader (GPU path)
│   ├── ffmpeg_pipe.c/h   FFmpeg CLI pipe frame reader (CPU path)
│   ├── gpu_backend.c/h   CUDA GPU backend lifecycle and pipeline
│   ├── gpu_kernels.cu/.h CUDA kernels (bit extraction, frame gen)
│   ├── gpu_presets.c/h   GPU architecture presets and auto-detection
│   ├── gpu_nvenc.c/h     NVENC hardware encoder wrapper
│   ├── ffmpeg_hwdecoder.c/h NVDEC (obsolete, replaced by libav_decoder)
│   ├── calibration.c/h   Calibration bar extraction
│   ├── framedecode.c/h   Frame tile geometry and bit extraction
│   ├── framegen.c/h      Frame generation (encoder)
│   ├── reedsolomon.c/h   Reed-Solomon ECC
│   ├── simd_decode.c/h   SIMD-accelerated tile thresholding
│   ├── threadpool.c/h    Multi-threaded worker pool
│   ├── profiling.c/h     Per-stage profiling instrumentation
│   ├── logutil.c/h       Structured logging utility
│   ├── sha256.c/h        SHA-256 integrity verification
│   └── crc16.c/h         CRC-16-CCITT checksum
├── tests/                Unit tests
├── docs/                 Documentation
├── builder/              Build utilities
├── main_encoder.c
├── main_decoder.c
├── CMakeLists.txt
└── README.md
```

---

## Applications

* Data archival through video
* Platform-independent file transfer
* Storage inside video-hosting services
* Compression-resilient file transport
* Research into visual data storage systems

---

## Status

VIDCRYPT-V8 is an active experimental project focused on maximizing recovery reliability from heavily compressed video while maintaining extremely high throughput.

**Platforms:** Windows (x86-64 + CUDA), macOS (ARM64/x86-64), Linux (x86-64 + CUDA). ARM64 uses a scalar fallback for calibration with full functionality. GPU acceleration is implemented for the **decode path** (2378+ FPS on RTX 5070). GPU encode path with NVENC is under development.

---

## License

Licensed under the GNU General Public License v3.0 (GPL-3.0).

See the [LICENSE](LICENSE) file for details.

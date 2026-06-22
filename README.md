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
* **Async double-buffered pipeline** — CPU frame decode overlaps with GPU upload + kernel + copyback
* High-speed CPU implementation (no GPU required)
* SIMD-accelerated calibration reader (AVX2, SSE4.2)

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

Measured on **NVIDIA RTX 5070 (Blackwell)** with 1080p H.264 video:

| Backend | Decode | Encode |
| ------- | ------ | ------ |
| **CPU** | ~836 FPS | ~900 FPS |
| **GPU** | **~2378 FPS** | *TBD* |

GPU decode achieves **2.85× speedup** over CPU by:
1. **Direct libavcodec C API** — eliminates ffmpeg CLI pipe (`popen`/`fread`) overhead
2. **Async double-buffering** — CPU frame decode overlaps with GPU upload + kernel + D2H copyback
3. **CUDA `extract_bits` kernel** — massively parallel tile thresholding (one thread per block)

Actual performance depends on resolution, block size, codec settings, GPU model, and system hardware.

---

## Requirements

* CMake 3.16+
* C17-compatible compiler
* FFmpeg + FFprobe (for CPU path)
* **For GPU acceleration:** NVIDIA GPU (Maxwell+) + CUDA Toolkit 12.x + NVIDIA Video Codec SDK (for NVENC)

---

## Build

### CPU-only (any system)

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

### GPU-accelerated (requires NVIDIA CUDA Toolkit)

```bash
mkdir build && cd build
cmake .. -DUSE_CUDA=ON
cmake --build . --config Release
```

### Advanced options

| Option | Default | Description |
| ------ | ------- | ----------- |
| `USE_CUDA` | `OFF` | Enable CUDA GPU acceleration |
| `ENABLE_AVX2` | `ON` | Enable AVX2 SIMD for calibration reader |
| `ENABLE_PROFILING` | `OFF` | Enable per-stage profiling instrumentation |

---

## Usage

### Encode a File

```bash
vidcrypt-encoder -i input.zip -o encoded.mkv
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
│   ├── sha256.c/h        SHA-256 integrity verification
│   ├── bitstream.c/h     Bit packing/unpacking utilities
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

GPU acceleration is currently implemented for the **decode path only** (2378+ FPS on RTX 5070). GPU encode path with NVENC is under development.

---

## License

Licensed under the GNU General Public License v3.0 (GPL-3.0).

See the [LICENSE](LICENSE) file for details.

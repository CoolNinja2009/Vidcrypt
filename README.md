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
* FFmpeg-based video pipeline
* High-speed CPU implementation
* Experimental optimization framework for future SIMD/GPU acceleration

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

Current CPU performance on modern desktop hardware:

| Operation | Speed    |
| --------- | -------- |
| Encoding  | ~900 FPS |
| Decoding  | ~790 FPS |

Actual performance depends on resolution, block size, codec settings, and system hardware.

---

## Requirements

* CMake 3.16+
* C17-compatible compiler
* FFmpeg
* FFprobe

---

## Build

```bash
mkdir build
cd build

cmake ..
cmake --build . --config Release
```

---

## Usage

### Encode a File

```bash
vidcrypt-encoder -i input.zip -o encoded.mkv
```

### Decode a Video

```bash
vidcrypt-decoder -i encoded.mkv
```

The decoder automatically restores the original filename and verifies the recovered file before writing it to disk.

---

## Repository Structure

```text
.
├── src/          Core encoder and decoder implementation
├── tests/        Unit tests
├── docs/         Documentation
├── builder/      Build utilities
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

---

## License

Licensed under the GNU General Public License v3.0 (GPL-3.0).

See the [LICENSE](LICENSE) file for details.

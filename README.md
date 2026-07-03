# VIDCRYPT-V8 — Build Reference

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

File-to-video encoding: C engine + optional Java JNI tile decoder + optional CUDA GPU.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  C engine (primary)                             │
│  ├─ encoder.c/h      file → RS → frames → pipe  │
│  ├─ decoder.c/h      pipe → frames → RS → file  │
│  ├─ reedsolomon.c/h  RS(255,255-ecc) ECC        │
│  ├─ simd_decode.c/h  C SIMD tile decode         │
│  └─ threadpool.c/h   worker pool, SPSC queue    │
├─────────────────────────────────────────────────┤
│  Java JNI (optional, USE_JNI_TILES=ON)          │
│  └─ TileDecoder.java  3-4× faster tile decode   │
│     Called from simd_decode.c via JNI            │
├─────────────────────────────────────────────────┤
│  CUDA GPU (optional, USE_CUDA=ON)               │
│  └─ gpu_kernels.cu   extract_bits, calibration  │
└─────────────────────────────────────────────────┘
```

## File Map

### C Source (`src/`)

| File | Role | Depends On | Lines |
|---|---|---|---|
| `backend.c/h` | Backend mode selection (cpu/gpu/auto) | — | ~35 |
| `bitstream.c/h` | MSB-first bit pack/unpack, popcount | — | ~100 |
| `calibration.c/h` | CalParams, cal bar R/W, v1/v2 parse, CRC guard | `crc16.h` | ~300 |
| `crc16.c/h` | CRC-16-CCITT (poly 0x1021, init 0xFFFF) | — | ~20 |
| `decoder.c/h` | Decode orchestrator: frame→tiles→RS→SHA-256→file | `calibration.h`, `framedecode.h`, `reedsolomon.h`, `sha256.h`, `threadpool.h`, `profiling.h`, `logutil.h`, `backend.h`, `libav_decoder.h` (GPU), `ffmpeg_pipe.h` (CPU) | ~870 |
| `encoder.c/h` | Encode orchestrator: file→RS→frames→FFmpeg pipe | `bitstream.h`, `reedsolomon.h`, `simd_decode.h`, `threadpool.h`, `ffmpeg_pipe.h`, `sha256.h`, `profiling.h` | ~465 |
| `ffmpeg_hwdecoder.c/h` | NVDEC direct C API (**obsolete** — use libav_decoder.c) | CUDA, FFmpeg | ~700 |
| `ffmpeg_pipe.c/h` | FFmpeg CLI pipe reader (CPU path), pipes to ffmpeg/ffprobe | FFmpeg in PATH | ~470 |
| `framegen.c/h` | PrecomputedFrame: cal bar + sync row + corner markers | `calibration.h`, `simd_decode.h` | ~75 |
| `framedecode.c/h` | DecodeGeometry, sync check, corner offset detection | `calibration.h`, `simd_decode.h` | ~180 |
| `gpu_backend.c/h` | CUDA context, stream, async pipeline management | CUDA, `gpu_kernels.cu`, `gpu_presets.h`, `decoder.h` | ~750 |
| `gpu_kernels.cu/h` | CUDA kernels: `extract_bits`, `frame_upload`, `calibration_read` | CUDA 12.x | ~450 |
| `gpu_nvenc.c/h` | NVENC hardware encoder (**stub**) | NVIDIA Video Codec SDK | ~800 |
| `gpu_presets.c/h` | Per-arch GPU presets (sm_61..sm_120), auto-detect | CUDA | ~200 |
| `libav_decoder.c/h` | Direct libavcodec C API reader (GPU h264_cuvid path) | FFmpeg libs | ~600 |
| `logutil.c/h` | Structured logging to `log/` directory | — | ~350 |
| `profiling.c/h` | Per-stage timing instrumentation | — | ~200 |
| `pthread.h` | POSIX threads compat shim (Windows) | — | ~50 |
| `reedsolomon.c/h` | RS(n=255,k=255-ecc): LFSR encode, B-M, Chien, Forney decode | — | ~260 |
| `sha256.c/h` | SHA-256 (file + data) | — | ~30 |
| `simd_decode.c/h` | Tile thresholding (branchless block8/16), cal SSE2/AVX2 reader | `calibration.h` | ~530 |
| `threadpool.c/h` | Thread pool, ordered queue, lock-free SPSC queue | pthread | ~280 |
| `tile_decode_jni.c/h` | **JNI bridge**: loads JVM, calls `JniTileDecoder.tileDecodeGrid` | JVM, `simd_decode.h` (fallback) | ~130 |

### Entry Points

| File | Binary | Role |
|---|---|---|
| `main_encoder.c` | `vidcrypt-encoder` | CLI: file → video |
| `main_decoder.c` | `vidcrypt-decoder` | CLI: video → file |

### Java Module (`vidcrypt-java/`)

| File | Role | Used by |
|---|---|---|
| `frame/JniTileDecoder.java` | **JNI entry point** — wraps `TileDecoder.decodeGrid` | C via `tile_decode_jni.c` |
| `frame/TileDecoder.java` | Branchless tile decode, 3-4× faster than C SIMD | `JniTileDecoder.java` |
| `frame/DecodeGeometry.java` | Frame tile geometry | `TileDecoder.java` |
| `frame/FrameGenerator.java` | Frame generation (standalone, not used by C) | — |
| `calibration/CalParams.java` | Calibration builder/parser matching C `calibration.h` | `TileDecoder.java`, `FrameGenerator.java` |
| `bitstream/BitStream.java` | MSB-first bit packing (standalone, not used by C) | — |
| `hash/Crc16.java` | CRC-16-CCITT matching C `crc16.c` | `CalParams.java` |
| `hash/Sha256.java` | SHA-256 via `java.security.MessageDigest` | — |
| `concurrent/VidcryptExecutors.java` | Virtual-thread pool, ordered queue, SPSC (standalone) | — |

### Tests

| File | Framework | Tests |
|---|---|---|
| `tests/test_crc16.c` | CTest | CRC-16 known vectors |
| `tests/test_bitstream.c` | CTest | Bit pack round-trip |
| `tests/test_reedsolomon.c` | CTest | RS encode/decode identity, error correction |
| `tests/test_decode.c` | CTest | Integration decode |
| `vidcrypt-java/tests/test_reedsolomon.py` | pytest | RS(255,223,32): encode/decode, GF(256) tables, 1-17 errors |
| `vidcrypt-java/tests/test_bitstream.py` | pytest | Bits↔bytes round-trip, word packing |
| `vidcrypt-java/tests/test_crc16.py` | pytest | CRC-16-CCITT known vectors |
| `vidcrypt-java/tests/test_calibration.py` | pytest | CalParams build/parse, dot R/W, noise resilience |
| `vidcrypt-java/tests/conftest.py` | pytest | Shared fixtures |

## Dependencies

| Component | Version | Required For |
|---|---|---|
| CMake | ≥ 3.16 | All builds |
| C compiler | GCC 10+ / Clang 12+ / MSVC 2019+ | All builds |
| FFmpeg + FFprobe | ≥ 4.0 (in PATH) | Runtime (CPU pipe) |
| BtbN FFmpeg shared | Latest (`ffmpeg_shared/`) | Windows NVDEC (libavcodec C API) |
| pthreads | — | All builds |
| CUDA Toolkit | 12.x | `USE_CUDA=ON` |
| JDK | 21+ | `USE_JNI_TILES=ON` |
| Python | ≥ 3.10 | pytest suite |
| pytest | ≥ 7.0 | `vidcrypt-java/tests/` |

## Build

### Single command (auto-detect everything)

```bash
# Windows
cd builder
build.bat                  # auto-detect: CUDA + Java + AVX2
build.bat --gpu            # force CUDA GPU
build.bat --cpu --no-gpu   # CPU only, no CUDA
build.bat --no-java        # skip Java tile decoder

# Linux / macOS
cd builder
./build.sh                 # auto-detect everything
./build.sh --no-java       # skip Java tile decoder
```

This compiles Java classes → JAR, then C → exe, and copies `vidcrypt-tiles.jar`
alongside the binaries. The decoder auto-loads the JAR at runtime.

Output: `build/vidcrypt-encoder`, `build/vidcrypt-decoder`, `build/vidcrypt-tiles.jar`

### Manual CMake (advanced)

```bash
# 1. Build Java JAR first
cd vidcrypt-java
javac --enable-preview --release 21 -d out --add-modules jdk.incubator.vector \
  vidcrypt-core/src/main/java/com/vidcrypt/frame/JniTileDecoder.java \
  vidcrypt-core/src/main/java/com/vidcrypt/frame/TileDecoder.java \
  vidcrypt-core/src/main/java/com/vidcrypt/frame/DecodeGeometry.java \
  vidcrypt-core/src/main/java/com/vidcrypt/calibration/CalParams.java \
  vidcrypt-core/src/main/java/com/vidcrypt/hash/Crc16.java
jar cf ../vidcrypt-tiles.jar -C out .
cd ..

# 2. Build C with Java tiles
cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_JNI_TILES=ON
cmake --build build --parallel

# Or: GPU + Java tiles
cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DUSE_JNI_TILES=ON
cmake --build build --parallel
```

### CMake Options

| Option | Default | Description |
|---|---|---|
| `USE_CUDA` | OFF | CUDA GPU NVDEC + tile kernel |
| `USE_JNI_TILES` | OFF | Java JNI tile decode (3-4× speedup) |
| `ENABLE_AVX2` | ON | AVX2 SIMD for calibration reader |
| `ENABLE_PROFILING` | OFF | Per-stage timing instrumentation |
| `ENABLE_LOGGING` | OFF | Structured logging to `log/` |

## Usage

```bash
# Encode (lossless FFV1)
vidcrypt-encoder -i input.zip -o encoded.mkv

# Encode (YouTube H.264, NVDEC-compatible)
vidcrypt-encoder -i input.zip -o encoded.mkv -c h264

# Decode (CPU C SIMD)
vidcrypt-decoder -i encoded.mkv

# Decode (GPU NVDEC + Java tiles, auto-loaded)
vidcrypt-decoder -b gpu -i encoded.mkv

CLI flags: `-b cpu|gpu|auto` (backend), `-J <path>` (Java classpath, JNI tiles), `-j <n>` (worker threads), `-o <dir>` (output dir).

## Tests

```bash
# C tests
cd build && ctest --output-on-failure && cd ..

# Python tests (Java module)
cd vidcrypt-java && python -m pytest tests/ -v && cd ..
```

## Debug Trace

**`tile_decode_grid` execution path:**
1. `decoder_decode_file` (`decoder.c:214`) calls `tile_decode_grid` (`simd_decode.c:224`)
2. If `USE_JNI_TILES` defined and JNI available → `tile_decode_grid_jni` (`tile_decode_jni.c`)
   → JNI → `JniTileDecoder.tileDecodeGrid` → `TileDecoder.decodeGrid` (Java)
3. Fallback → C `tile_validate_sync` + `count_white_block8` (`simd_decode.c:89`)

**RS encode path:**
`encoder_encode_file` (`encoder.c:147`) → loop → `rs_encode` (`reedsolomon.c:72`)
→ LFSR remainder in `bb[]` → write ECC symbols to `encoded[k+i]`

**Frame layout:**
`calibration.h` constants → `calibration.c:build_calibration_bytes` (24 bytes)
→ `framegen.c:precomputed_frame_init` (pre-renders cal bar + sync + corners)
→ `framegen.c:precomputed_frame_generate` → `tile_expand_bits` (payload)

**Calibration read:**
`extract_calibration` (`calibration.c:183`) → `read_calibration_dots` (192 bits)
→ SSE2/AVX2 (`simd_decode.c:290`) or scalar → `parse_calibration_bytes` (CRC verify)

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).

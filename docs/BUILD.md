# Vidcrypt Build Manual

## Prerequisites

### All platforms
- CMake >= 3.16
- C compiler (GCC 10+, Clang 12+, MSVC 2019+)
- pthreads (system on Linux/macOS; winpthreads or pthreadVC2 on Windows)

### Runtime dependencies (not linked, found in PATH)
- FFmpeg >= 4.0 (for ffmpeg CLI pipe)
- FFprobe >= 4.0 (for video metadata)

### Optional: GPU acceleration (USE_CUDA=ON)
- CUDA Toolkit 12.x
- NVIDIA GPU with driver >= 525
- Windows only: BtbN FFmpeg shared build in `ffmpeg_shared/`

### Optional: Java JNI tiles (USE_JNI_TILES=ON)
- JDK 21+ (JAVA_HOME must be set)
- Compiled Java classes in `vidcrypt-java/out/com/vidcrypt/frame/`

## Step-by-step

### 1. Clone / extract

```bash
cd Vidcrypt-main
```

### 2. Verify prerequisites

```bash
cmake --version          # >= 3.16
gcc --version || clang --version || cl.exe  # C compiler
ffmpeg -version          # runtime requirement
ffprobe -version         # runtime requirement
java -version            # only if USE_JNI_TILES=ON
nvcc --version           # only if USE_CUDA=ON
```

### 3. Build Java module (if USE_JNI_TILES=ON)

```bash
cd vidcrypt-java
# Clean
rm -rf out
mkdir out

# Compile (JDK 21+)
javac --enable-preview --release 21 -d out \
  --add-modules jdk.incubator.vector \
  vidcrypt-core/src/main/java/com/vidcrypt/frame/JniTileDecoder.java \
  vidcrypt-core/src/main/java/com/vidcrypt/frame/TileDecoder.java \
  vidcrypt-core/src/main/java/com/vidcrypt/frame/DecodeGeometry.java \
  vidcrypt-core/src/main/java/com/vidcrypt/calibration/CalParams.java \
  vidcrypt-core/src/main/java/com/vidcrypt/hash/Crc16.java

# Verify classes exist
ls out/com/vidcrypt/frame/JniTileDecoder.class
cd ..
```

### 4. Configure CMake

Pick one:

```bash
# Minimal (CPU only, C SIMD)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# CPU + Java tiles
cmake -B build_jni -DCMAKE_BUILD_TYPE=Release -DUSE_JNI_TILES=ON

# GPU + Java tiles (Linux, requires CUDA)
cmake -B build_full -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=ON -DUSE_JNI_TILES=ON

# Windows MSVC GPU + Java
cmake -B build_full -G "Visual Studio 17 2022" -A x64 \
  -DUSE_CUDA=ON -DUSE_JNI_TILES=ON \
  -DJAVA_HOME="C:/Program Files/Java/jdk-21"
```

### 5. Build

```bash
cmake --build build --parallel      # or build_jni, build_full
```

### 6. Verify binaries

```bash
ls build/vidcrypt-encoder*
ls build/vidcrypt-decoder*
./build/vidcrypt-encoder --help
./build/vidcrypt-decoder --help
```

### 7. Run tests

```bash
# C tests
cd build && ctest --output-on-failure && cd ..

# Python tests (Java module)
cd vidcrypt-java && python -m pytest tests/ -v && cd ..
```

## Platform-specific notes

### Windows (MSVC)
- Install Visual Studio 2022 with "Desktop development with C++"
- Install CUDA Toolkit from NVIDIA (adds nvcc to PATH)
- Copy BtbN FFmpeg DLLs to `ffmpeg_shared/` for NVDEC support
- pthreads: use vcpkg or conan (`vcpkg install pthreads`)

### Windows (MinGW/MSYS2)
```bash
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-ffmpeg
```
- No CUDA support with MinGW — use MSVC for GPU builds

### Linux
```bash
# Ubuntu/Debian
apt install build-essential cmake ffmpeg libavcodec-dev libavformat-dev libavutil-dev
# CUDA: install from NVIDIA .run or .deb

# Fedora
sudo dnf install gcc cmake ffmpeg-devel
```

### macOS
- No CUDA support (Apple Silicon)
- ARM64 uses scalar fallback for calibration (no SSE2/AVX2)
- USE_JNI_TILES works if JDK 21+ is installed

## Troubleshooting

### "jni.h not found"
Set JAVA_HOME: `export JAVA_HOME=/path/to/jdk-21` or `-DJAVA_HOME=...` in cmake.

### "JVM library not found"
USE_JNI_TILES requires JDK (not just JRE). Install JDK 21+.

### "JniTileDecoder class not found" (runtime)
Compile Java module first (step 3). Verify `vidcrypt-java/out/com/vidcrypt/frame/JniTileDecoder.class` exists.

### "pthread.h not found" (Windows MinGW)
Install winpthreads: `pacman -S mingw-w64-x86_64-winpthreads-git`

### "h264_cuvid not found" (NVDEC)
Copy BtbN FFmpeg shared DLLs to `ffmpeg_shared/bin/`. Set FFMPEG_SHARED_DIR if not at default location.

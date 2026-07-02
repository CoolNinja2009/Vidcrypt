@echo off
setlocal enabledelayedexpansion

echo ========================================
echo  VIDCRYPT-V8 Build Script - GPU + CUDA
echo ========================================
echo.

REM -- Prerequisites check -------------------------------------------
echo [*] Checking prerequisites...

REM Check CMake
where cmake >nul 2>nul
if errorlevel 1 (
    echo [FAIL] CMake not found in PATH.
    echo        Install CMake 3.16+ from https://cmake.org/download/
    pause
    exit /b 1
)

REM Check C compiler (cc/gcc/clang or MSVC)
where cc >nul 2>nul
if errorlevel 1 (
    where gcc >nul 2>nul
    if errorlevel 1 (
        where cl >nul 2>nul
        if errorlevel 1 (
            echo [FAIL] No C compiler found (cc, gcc, or cl).
            echo        Install MinGW-w64 or Visual Studio Build Tools.
            pause
            exit /b 1
        )
    )
)

REM Check CUDA toolkit (nvcc)
where nvcc >nul 2>nul
if errorlevel 1 (
    echo [FAIL] nvcc not found in PATH.
    echo        Install CUDA Toolkit 12.x from:
    echo        https://developer.nvidia.com/cuda-downloads
    echo.
    echo        After installation, ensure nvcc is in your PATH:
    echo        e.g. C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('where nvcc') do set "CUDA_PATH=%%~dpi"
echo        CUDA: !CUDA_PATH!..

REM Check NVIDIA Video Codec SDK (nvEncodeAPI.h — optional, NVENC only)
set "NVENC_FOUND="
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include\nvEncodeAPI.h" (
    set "NVENC_FOUND=1"
)
if not defined NVENC_FOUND (
    if exist "%CUDA_PATH%..\include\nvEncodeAPI.h" (
        set "NVENC_FOUND=1"
    )
)
if defined NVENC_FOUND (
    echo        NVENC SDK: found
) else (
    echo        NVENC SDK: not found (NVENC encoder will be disabled)
    echo        Download from: https://developer.nvidia.com/nvidia-video-codec-sdk
)

REM Check FFmpeg dev headers (for NVDEC)
set "FFMPEG_DIR="
if exist "..\ffmpeg_shared\ffmpeg-master-latest-win64-gpl-shared\include\libavformat\avformat.h" (
    set "FFMPEG_DIR=..\ffmpeg_shared\ffmpeg-master-latest-win64-gpl-shared"
    echo        FFmpeg dev: !FFMPEG_DIR!
) else if exist "..\ffmpeg_shared\include\libavformat\avformat.h" (
    set "FFMPEG_DIR=..\ffmpeg_shared"
    echo        FFmpeg dev: !FFMPEG_DIR!
) else (
    echo        FFmpeg dev: not found (NVDEC zero-copy will be disabled)
    echo        Download BtbN FFmpeg shared build to ffmpeg_shared/
)

echo.

REM -- Create build directory ----------------------------------------
if not exist build_cuda (
    echo [*] Creating build_cuda directory...
    mkdir build_cuda
)

cd /d build_cuda

echo.
echo [*] Configuring CMake (CUDA, AVX2)...
cmake .. -DUSE_CUDA=ON -DENABLE_AVX2=ON
if errorlevel 1 (
    echo.
    echo [FAIL] CMake configuration failed.
    echo        Ensure CUDA Toolkit 12.x is installed and in PATH.
    echo        Ensure FFmpeg dev headers are available (optional).
    echo        See docs\GPU_ACCELERATION.md for details.
    pause
    exit /b 1
)

echo.
echo [*] Building Release...
cmake --build . --config Release
if errorlevel 1 (
    echo.
    echo [FAIL] Build failed.
    pause
    exit /b 1
)

REM -- Copy FFmpeg DLLs to build output ------------------------------
if defined FFMPEG_DIR (
    if exist "!FFMPEG_DIR!\bin\avformat-62.dll" (
        echo.
        echo [*] Copying FFmpeg DLLs to build output...
        copy /y "!FFMPEG_DIR!\bin\*.dll" "Release\" >nul 2>&1
        echo [*] FFmpeg DLLs copied to build_cuda\Release\
    )
)

echo.
echo ========================================
echo  CUDA build completed successfully!
echo  Output: build_cuda\Release\
echo ========================================
echo.
echo  Usage (GPU decode):
echo    vidcrypt-decoder -b gpu -i video.mkv
echo.
echo  Usage (CPU decode):
echo    vidcrypt-decoder -b cpu -i video.mkv
echo.
echo  Expected performance (RTX 5070, 1080p H.264):
echo    GPU: ~2378 FPS    CPU: ~836 FPS
echo.

pause

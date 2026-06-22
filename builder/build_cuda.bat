@echo off
setlocal enabledelayedexpansion

echo ========================================
echo  VIDCRYPT-V8 Build Script — GPU + CUDA
echo ========================================
echo.

REM ── Auto-detect local FFmpeg path ────────────────────────────────
set FFMPEG_DIR=
if exist "..\ffmpeg_shared\ffmpeg-master-latest-win64-gpl-shared\include\libavformat\avformat.h" (
    set "FFMPEG_DIR=..\ffmpeg_shared\ffmpeg-master-latest-win64-gpl-shared"
    echo [*] Found local FFmpeg: %FFMPEG_DIR%
) else if exist "..\ffmpeg_shared\include\libavformat\avformat.h" (
    set "FFMPEG_DIR=..\ffmpeg_shared"
    echo [*] Found local FFmpeg: %FFMPEG_DIR%
) else (
    echo [!] No ffmpeg_shared directory found.
    echo     CUDA builds require FFmpeg dev headers (BtbN shared build).
    echo     Place them in: ffmpeg_shared/ffmpeg-master-latest-win64-gpl-shared/
    echo.
    echo     Continuing with hardcoded system path (may fail on other machines)...
)

REM ── Check CUDA toolkit availability ──────────────────────────────
echo [*] Checking for CUDA toolkit...
where nvcc >nul 2>nul
if errorlevel 1 (
    echo [!] nvcc not found in PATH.
    echo     CUDA toolkit may not be installed or not in PATH.
    echo     Continuing anyway (CMake will fail if CUDAToolkit not found)...
) else (
    for /f "tokens=*" %%i in ('where nvcc') do set CUDA_PATH=%%~dpi
    echo [*] Found nvcc in: %CUDA_PATH%
)

REM ── Create build directory ────────────────────────────────────────
if not exist build_cuda (
    echo [*] Creating build_cuda directory...
    mkdir build_cuda
)

cd build_cuda

echo.
echo [*] Configuring CMake (CUDA, AVX2)...
cmake .. -DUSE_CUDA=ON -DENABLE_AVX2=ON
if errorlevel 1 (
    echo.
    echo [FAIL] CMake configuration failed.
    echo     Ensure CUDA Toolkit 12.x is installed.
    echo     Ensure FFmpeg dev headers are available.
    echo     See docs/GPU_ACCELERATION.md for details.
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

REM ── Copy FFmpeg DLLs to build output ──────────────────────────────
if defined FFMPEG_DIR (
    echo.
    echo [*] Copying FFmpeg DLLs to build output...
    if exist "!FFMPEG_DIR!\bin\avformat-62.dll" (
        copy /y "!FFMPEG_DIR!\bin\*.dll" "Release\" >nul 2>&1
        echo [*] FFmpeg DLLs copied to build_cuda\Release\
    )
)

echo.
echo ========================================
echo  CUDA build completed successfully!
echo  Output: build_cuda\Release\vidcrypt-decoder.exe
echo  Output: build_cuda\Release\vidcrypt-encoder.exe
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

@echo off
setlocal enabledelayedexpansion

echo ========================================
echo  VIDCRYPT-V8 Clean Rebuild
echo ========================================
echo.
echo  Select build type:
echo    1) CPU-only (default)
echo    2) CUDA GPU-accelerated
echo    3) Profiling + logging + verbose
echo    4) Tests
echo.
set /p BUILD_TYPE="Enter choice [1-4]: "

if "%BUILD_TYPE%"=="" set BUILD_TYPE=1

set BUILD_DIR=build
set CMAKE_FLAGS=-DENABLE_AVX2=ON

if "%BUILD_TYPE%"=="1" (
    set BUILD_DIR=build
    set CMAKE_FLAGS=-DENABLE_AVX2=ON
    echo [*] Selected: CPU-only
) else if "%BUILD_TYPE%"=="2" (
    set BUILD_DIR=build_cuda
    set CMAKE_FLAGS=-DUSE_CUDA=ON -DENABLE_AVX2=ON
    echo [*] Selected: CUDA GPU-accelerated
) else if "%BUILD_TYPE%"=="3" (
    set BUILD_DIR=build_verbose
    set CMAKE_FLAGS=-DENABLE_PROFILING=ON -DENABLE_LOGGING=ON -DENABLE_AVX2=ON
    echo [*] Selected: Profiling + logging + verbose
) else if "%BUILD_TYPE%"=="4" (
    set BUILD_DIR=build_tests
    set CMAKE_FLAGS=-DENABLE_AVX2=ON
    echo [*] Selected: Tests
) else (
    echo [FAIL] Invalid choice. Defaulting to CPU-only.
    set BUILD_DIR=build
    set CMAKE_FLAGS=-DENABLE_AVX2=ON
)

echo.
echo [*] Removing old build directory: !BUILD_DIR!
if exist "!BUILD_DIR!" (
    rmdir /s /q "!BUILD_DIR!"
    if errorlevel 1 (
        echo [FAIL] Could not remove !BUILD_DIR!. Files may be in use.
        pause
        exit /b 1
    )
)

echo [*] Creating fresh build directory...
mkdir "!BUILD_DIR!"
cd "!BUILD_DIR!"

echo.
echo [*] Configuring CMake: !CMAKE_FLAGS!
cmake .. !CMAKE_FLAGS!
if errorlevel 1 (
    echo.
    echo [FAIL] CMake configuration failed.
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

REM ── Copy FFmpeg DLLs for CUDA builds ──────────────────────────────
if "%BUILD_TYPE%"=="2" (
    if exist "..\ffmpeg_shared\ffmpeg-master-latest-win64-gpl-shared\bin\avformat-62.dll" (
        echo.
        echo [*] Copying FFmpeg DLLs...
        copy /y "..\ffmpeg_shared\ffmpeg-master-latest-win64-gpl-shared\bin\*.dll" "Release\" >nul 2>&1
    )
)

REM ── Run tests if tests build ──────────────────────────────────────
if "%BUILD_TYPE%"=="4" (
    echo.
    echo [*] Running tests...
    ctest -C Release --output-on-failure
    if errorlevel 1 (
        echo.
        echo [FAIL] Some tests failed.
        pause
        exit /b 1
    )
)

echo.
echo ========================================
echo  Rebuild completed successfully!
echo  Output: !BUILD_DIR!\Release\
echo ========================================

pause

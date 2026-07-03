@echo off
setlocal enabledelayedexpansion

REM ── Platform detection ────────────────────────────────────────────
cd /d "%~dp0.."
set "PROJECT_DIR=%CD%"

echo ========================================
echo  VIDCRYPT-V8  --  Clean Rebuild
echo ========================================
echo.

REM ── Auto-detect build type ────────────────────────────────────────
set "HAS_NVCC="
set "HAS_CL="
where nvcc >nul 2>nul && set "HAS_NVCC=1"
where cl   >nul 2>nul && set "HAS_CL=1"

set "BUILD_DIR=build"
set "CMAKE_FLAGS=-DENABLE_AVX2=ON"

if defined HAS_NVCC if defined HAS_CL (
    set "BUILD_DIR=build_cuda"
    set "CMAKE_FLAGS=-DUSE_CUDA=ON -DENABLE_AVX2=ON"
    echo [*] Detected CUDA + MSVC — GPU-accelerated build
) else if defined HAS_CL (
    echo [*] Detected MSVC — CPU build ^(CUDA not found^)
) else (
    echo [*] Detected MinGW/GCC — CPU build
)

echo   Directory: !BUILD_DIR!
echo   Flags:     !CMAKE_FLAGS!
echo.

REM ── Clean old build ───────────────────────────────────────────────
if exist "!BUILD_DIR!" (
    echo [*] Removing old !BUILD_DIR!...
    rmdir /s /q "!BUILD_DIR!"
    if errorlevel 1 (
        echo [FAIL] Could not remove !BUILD_DIR!. Files may be in use.
        pause & exit /b 1
    )
)

REM ── Create fresh build dir ────────────────────────────────────────
echo [*] Creating !BUILD_DIR!...
mkdir "!BUILD_DIR!"
cd "!BUILD_DIR!"

REM ── Configure CMake ───────────────────────────────────────────────
echo.
echo [*] Configuring CMake: !CMAKE_FLAGS!
cmake "!PROJECT_DIR!" !CMAKE_FLAGS!
if errorlevel 1 (
    echo.
    echo [FAIL] CMake configuration failed.
    pause & exit /b 1
)

REM ── Build ─────────────────────────────────────────────────────────
echo.
echo [*] Building Release...
cmake --build . --config Release --parallel %NUMBER_OF_PROCESSORS%
if errorlevel 1 (
    echo.
    echo [FAIL] Build failed.
    pause & exit /b 1
)

REM ── Run tests ─────────────────────────────────────────────────────
echo.
echo [*] Running tests...
ctest -C Release --output-on-failure
if errorlevel 1 (
    echo.
    echo [FAIL] Some tests failed.
    pause & exit /b 1
)

echo.
echo ========================================
echo  Rebuild + tests passed!
echo  Output: !BUILD_DIR!\Release\
echo ========================================

pause

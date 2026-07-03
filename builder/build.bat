@echo off
setlocal enabledelayedexpansion

REM ── Platform info ─────────────────────────────────────────────────
echo ========================================
echo  VIDCRYPT-V8  --  Auto-Detect Build
echo ========================================
echo  Platform:  Windows
echo.

REM ── Parse command-line flags ─────────────────────────────────────
set "FORCE_CPU="
set "FORCE_GPU="
:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="--cpu"     set "FORCE_CPU=1"
if /i "%~1"=="--no-gpu"  set "FORCE_CPU=1"
if /i "%~1"=="--gpu"     set "FORCE_GPU=1"
if /i "%~1"=="--cuda"    set "FORCE_GPU=1"
shift
goto :parse_args
:args_done

REM ── Find project root (script is in builder/) ─────────────────────
cd /d "%~dp0.."
set "PROJECT_DIR=%CD%"

REM ── Check cmake ───────────────────────────────────────────────────
echo ── Prerequisites ──
where cmake >nul 2>nul
if errorlevel 1 (
    echo [FAIL] cmake not found in PATH.
    pause & exit /b 1
)
for /f "tokens=*" %%i in ('cmake --version 2^>^&1 ^| findstr /b "cmake"') do echo   [+] %%i

REM ── Check C compiler ──────────────────────────────────────────────
set "HAS_CC="
set "CC_LABEL="
where cl >nul 2>nul && set "HAS_CC=cl" && set "CC_LABEL=MSVC (cl.exe)"
if not defined HAS_CC where cc  >nul 2>nul && set "HAS_CC=cc"  && set "CC_LABEL=MinGW (cc)"
if not defined HAS_CC where gcc >nul 2>nul && set "HAS_CC=gcc" && set "CC_LABEL=MinGW (gcc)"
if not defined HAS_CC (
    echo [FAIL] No C compiler found.
    pause & exit /b 1
)
echo   [+] Compiler: !CC_LABEL!

REM ── CPU info ──────────────────────────────────────────────────────
set "NPROC=4"
if defined NUMBER_OF_PROCESSORS set "NPROC=%NUMBER_OF_PROCESSORS%"
echo   [*] CPU cores: %NPROC%

REM ── CUDA + VS generator probe ─────────────────────────────────────
set "USE_CUDA=OFF"
set "BUILD_DIR=%PROJECT_DIR%\build"
set "BUILD_LABEL=CPU-only"
set "VS_GEN="

if defined FORCE_CPU (
    echo   [*] --cpu flag set — skipping CUDA detection
    goto :cuda_done
)

where nvcc >nul 2>nul
if not errorlevel 1 (
    for /f "tokens=*" %%i in ('nvcc --version 2^>^&1 ^| findstr "release"') do set "CUDA_VER=%%i"
    echo   [+] nvcc: !CUDA_VER!

    where cl >nul 2>nul
    if not errorlevel 1 (
        REM Try known VS generators newest-first
        if not defined VS_GEN call :probe_vs "Visual Studio 18 2026"
        if not defined VS_GEN call :probe_vs "Visual Studio 17 2022"
        if not defined VS_GEN call :probe_vs "Visual Studio 16 2019"

        if defined VS_GEN (
            set "USE_CUDA=ON"
            set "BUILD_DIR=%PROJECT_DIR%\build_cuda"
            set "BUILD_LABEL=GPU + CUDA"
            echo   [+] VS generator: !VS_GEN!
        ) else (
            if defined FORCE_GPU (
                echo   [FAIL] --gpu flag set but no usable VS generator found
                echo          Run from a Visual Studio Developer Command Prompt.
                pause ^& exit /b 1
            )
            echo   [!!] nvcc + MSVC found, but no usable VS generator in this shell
            echo   [!!] Falling back to CPU-only build
        )
    ) else (
        if defined FORCE_GPU (
            echo   [FAIL] --gpu flag set but no MSVC compiler found
            pause ^& exit /b 1
        )
        echo   [!!] nvcc found but no MSVC compiler
        echo   [!!] Falling back to CPU-only build
    )
) else (
    if defined FORCE_GPU (
        echo   [FAIL] --gpu flag set but nvcc not found
        echo          Install CUDA Toolkit 12.x
        pause ^& exit /b 1
    )
    echo   [*] nvcc not found -- CPU-only build
)
:cuda_done

REM ── FFmpeg detection ──────────────────────────────────────────────
set "HAS_FFMPEG="
where ffmpeg >nul 2>nul && set "HAS_FFMPEG=1"
if "!USE_CUDA!"=="ON" (
    if exist "%PROJECT_DIR%\ffmpeg_shared\ffmpeg-master-latest-win64-gpl-shared\include\libavformat\avformat.h" (
        echo   [+] FFmpeg dev headers: found
    ) else (
        echo   [*] FFmpeg dev headers: not found ^(NVDEC disabled^)
    )
) else (
    if defined HAS_FFMPEG (echo   [+] ffmpeg binary: found) else (
        echo   [*] ffmpeg: not found
    )
)

echo.

REM ── Summary ───────────────────────────────────────────────────────
echo ── Build plan ──
echo   Type:      !BUILD_LABEL!
echo   Directory: !BUILD_DIR!
if "!USE_CUDA!"=="ON" echo   Generator: !VS_GEN! -A x64
echo   Flags:     USE_CUDA=!USE_CUDA!  ENABLE_AVX2=ON
echo   Parallel:  !NPROC! jobs
echo.

REM ── Create / enter build dir ──────────────────────────────────────
if not exist "!BUILD_DIR!" mkdir "!BUILD_DIR!"
cd /d "!BUILD_DIR!"

REM Clean stale cache from different generator
if exist CMakeCache.txt (
    del CMakeCache.txt >nul 2>&1
    rmdir /s /q CMakeFiles >nul 2>&1
)

REM ── Configure CMake ───────────────────────────────────────────────
echo [*] Configuring CMake...
if "!USE_CUDA!"=="ON" (
    cmake "!PROJECT_DIR!" -G "!VS_GEN!" -A x64 -DUSE_CUDA=ON -DENABLE_AVX2=ON
) else (
    cmake "!PROJECT_DIR!" -DUSE_CUDA=OFF -DENABLE_AVX2=ON
)
if errorlevel 1 (
    echo.
    echo [FAIL] CMake configuration failed.
    if "!USE_CUDA!"=="ON" (
        echo        Run from a Visual Studio Developer Command Prompt.
    )
    pause & exit /b 1
)

REM ── Build ─────────────────────────────────────────────────────────
echo.
echo [*] Building Release ^(!NPROC! parallel^)...
cmake --build . --config Release --parallel !NPROC!
if errorlevel 1 (
    echo.
    echo [FAIL] Build failed.
    pause & exit /b 1
)

REM ── Done ──────────────────────────────────────────────────────────
echo.
echo ========================================
echo  Build complete -- !BUILD_LABEL!
if "!USE_CUDA!"=="ON" (
    echo  Output: !BUILD_DIR!\Release\
) else (
    echo  Output: !BUILD_DIR!\
)
echo ========================================
echo.
if "!USE_CUDA!"=="ON" (
    echo  GPU decode: !BUILD_DIR!\Release\vidcrypt-decoder.exe -b gpu -i video.mkv
    echo  CPU decode: !BUILD_DIR!\Release\vidcrypt-decoder.exe -b cpu -i video.mkv
) else (
    echo  Encode: !BUILD_DIR!\vidcrypt-encoder.exe -i input.zip -o encoded.mkv
    echo  Decode: !BUILD_DIR!\vidcrypt-decoder.exe -i encoded.mkv
)
echo  Tests:    cd !BUILD_DIR! ^&^& ctest -C Release --output-on-failure
echo.

pause
goto :eof

REM ═══════════════════════════════════════════════════════════════════
REM Probe: test if a VS generator is usable in the current shell
REM Sets VS_GEN if the generator works.
REM ═══════════════════════════════════════════════════════════════════
:probe_vs
set "TEST_GEN=%~1"
REM Try a minimal configure in a temp dir to test the generator
set "TMP_PROBE=%TEMP%\vidcrypt_vs_probe"
if exist "%TMP_PROBE%" rmdir /s /q "%TMP_PROBE%" >nul 2>&1
mkdir "%TMP_PROBE%" >nul 2>&1
pushd "%TMP_PROBE%" >nul 2>&1
echo project^(test LANGUAGES C^) > CMakeLists.txt
echo int main^(void^){return 0;} > main.c
cmake -G "%TEST_GEN%" -A x64 . >nul 2>&1
if not errorlevel 1 (
    set "VS_GEN=%TEST_GEN%"
)
popd >nul 2>&1
rmdir /s /q "%TMP_PROBE%" >nul 2>&1
exit /b 0

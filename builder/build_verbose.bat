@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0.."
set "PROJECT_DIR=%CD%"

echo ========================================
echo  VIDCRYPT-V8  --  Verbose Build
echo  ^(Profiling + Logging + AVX2^)
echo ========================================
echo.

set "BUILD_DIR=%PROJECT_DIR%\build_verbose"
if not exist "!BUILD_DIR!" mkdir "!BUILD_DIR!"
cd /d "!BUILD_DIR!"

echo [*] Configuring CMake ^(profiling, logging, AVX2^)...
cmake "!PROJECT_DIR!" -DENABLE_PROFILING=ON -DENABLE_LOGGING=ON -DENABLE_AVX2=ON
if errorlevel 1 (
    echo [FAIL] CMake configuration failed.
    pause & exit /b 1
)

echo.
echo [*] Building Release ^(%NUMBER_OF_PROCESSORS% parallel^)...
cmake --build . --config Release --parallel %NUMBER_OF_PROCESSORS%
if errorlevel 1 (
    echo [FAIL] Build failed.
    pause & exit /b 1
)

echo.
echo ========================================
echo  Verbose build complete!
echo  Output: !BUILD_DIR!\Release\
echo ========================================
echo.
echo  Profiling + logging enabled.
echo  Run:  vidcrypt-decoder -i video.mkv
echo.

pause

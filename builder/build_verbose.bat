@echo off
setlocal enabledelayedexpansionecho ========================================
 echo  VIDCRYPT-V8 Build Script — Verbose
 echo  (Profiling + Logging + AVX2)
 echo ========================================
 echo.

 if not exist build_verbose (
     echo [*] Creating build_verbose directory...
     mkdir build_verbose
 )

 cd build_verbose

 echo [*] Configuring CMake (profiling, logging, AVX2)...
 cmake .. -DENABLE_PROFILING=ON -DENABLE_LOGGING=ON -DENABLE_AVX2=ON
if errorlevel 1 (
    echo.
    echo [FAIL] CMake configuration failed.
    pause
    exit /b 1
)

echo.
echo [*] Building Release with verbose output...
cmake --build . --config Release --verbose
if errorlevel 1 (
    echo.
    echo [FAIL] Build failed.
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Verbose build completed successfully!
echo  Output: build_verbose\Release\
echo ========================================
echo.
echo  Note: Profiling is enabled.
echo  Run decoder with profiling:
echo    vidcrypt-decoder -i video.mkv  (prints stage timings)
echo.

pause

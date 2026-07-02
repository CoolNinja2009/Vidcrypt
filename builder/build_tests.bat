@echo off
setlocal enabledelayedexpansion

echo ========================================
echo  VIDCRYPT-V8 Build + Test Script
echo ========================================
echo.

if not exist build_tests (
    echo [*] Creating build_tests directory...
    mkdir build_tests
)

cd build_tests

echo [*] Configuring CMake with tests...
cmake .. -DENABLE_AVX2=ON
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

echo.
echo [*] Running tests via CTest...
echo.
ctest -C Release --output-on-failure
if errorlevel 1 (
    echo.
    echo [FAIL] Some tests failed. See output above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo  All tests passed!
echo ========================================

pause

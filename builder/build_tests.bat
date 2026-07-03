@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0.."
set "PROJECT_DIR=%CD%"

echo ========================================
echo  VIDCRYPT-V8  --  Build + Test
echo ========================================
echo.

set "BUILD_DIR=%PROJECT_DIR%\build_tests"
if not exist "!BUILD_DIR!" mkdir "!BUILD_DIR!"
cd /d "!BUILD_DIR!"

echo [*] Configuring CMake with tests...
cmake "!PROJECT_DIR!" -DENABLE_AVX2=ON
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
echo [*] Running tests...
echo.
ctest -C Release --output-on-failure
if errorlevel 1 (
    echo [FAIL] Some tests failed.
    pause & exit /b 1
)

echo.
echo ========================================
echo  All tests passed!
echo ========================================

pause

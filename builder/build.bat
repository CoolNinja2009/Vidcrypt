@echo off
setlocal

echo =====================================
echo VIDCRYPT-V8 Build Script
echo =====================================
echo.

REM Check if build folder exists
if not exist build (
    echo Creating build directory...
    mkdir build
)

cd build

echo.
echo Configuring CMake...
cmake ..
if errorlevel 1 (
    echo.
    echo CMake configuration failed.
    pause
    exit /b 1
)

echo.
echo Building project...
cmake --build . --config Release
if errorlevel 1 (
    echo.
    echo Build failed.
    pause
    exit /b 1
)

echo.
echo =====================================
echo Build completed successfully!
echo =====================================

pause
@echo off
setlocal

echo =====================================
echo Vidcrypt CUDA Build
echo =====================================

set CMAKE_EXE=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe

if not exist "%CMAKE_EXE%" (
    echo ERROR: CMake not found:
    echo %CMAKE_EXE%
    pause
    exit /b 1
)

echo.
echo Removing old build directory...
if exist build_cuda rmdir /s /q build_cuda

echo.
echo Configuring project...
"%CMAKE_EXE%" ^
-B build_cuda ^
-G "Visual Studio 17 2022" ^
-A x64 ^
-DUSE_CUDA=ON

if errorlevel 1 (
    echo.
    echo CONFIGURATION FAILED
    pause
    exit /b 1
)

echo.
echo Building Release...
"%CMAKE_EXE%" --build build_cuda --config Release

if errorlevel 1 (
    echo.
    echo BUILD FAILED
    pause
    exit /b 1
)

echo.
echo =====================================
echo BUILD SUCCESSFUL
echo =====================================

pause
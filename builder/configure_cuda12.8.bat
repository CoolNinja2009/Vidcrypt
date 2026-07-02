@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

:: Set CUDA 12.8 environment for MSBuild CUDA integration
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set CUDA_PATH_V12_8=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set CudaToolkitDir=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;%PATH%

cd /d "%~dp0..\build"
if exist CMakeCache.txt del CMakeCache.txt
if exist CMakeFiles rmdir /s /q CMakeFiles

cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DUSE_CUDA=ON -DENABLE_AVX2=ON ^
    -DCMAKE_CUDA_ARCHITECTURES="75;86;89;90;120"
echo Exit code: %ERRORLEVEL%
pause

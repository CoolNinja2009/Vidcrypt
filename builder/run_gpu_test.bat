@echo off
echo === GPU Encode Test with CUDA 12.8 Blackwell ===
cd /d "%~dp0..\build\Release"
echo CWD: %CD%
echo.

echo Running GPU encoder...
vidcrypt-encoder.exe -B gpu -i ..\..\test_encode.bin -o test_gpu_blackwell.mkv 2>&1
set RET=%ERRORLEVEL%
echo EXIT: %RET%
echo.

if exist test_gpu_blackwell.mkv (
    echo Output file size:
    dir test_gpu_blackwell.mkv
) else (
    echo Output file NOT created
)

echo.
echo === Test Complete ===
echo %DATE% %TIME%

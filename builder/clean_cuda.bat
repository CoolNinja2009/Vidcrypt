@echo off

if exist build_cuda (
    rmdir /s /q build_cuda
)

echo Build directory removed.
pause
@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0.."

echo ========================================
echo  VIDCRYPT Release Build
echo ========================================

REM ── Check GraalVM ──────────────────────────────────────────────────
set "NATIVE_IMAGE="
where native-image >nul 2>nul && set "NATIVE_IMAGE=native-image"
if not defined NATIVE_IMAGE if defined GRAALVM_HOME (
    if exist "%GRAALVM_HOME%\bin\native-image.cmd" set "NATIVE_IMAGE=%GRAALVM_HOME%\bin\native-image.cmd"
)
if not defined NATIVE_IMAGE if defined JAVA_HOME (
    if exist "%JAVA_HOME%\bin\native-image.cmd" set "NATIVE_IMAGE=%JAVA_HOME%\bin\native-image.cmd"
)

if defined NATIVE_IMAGE (
    echo [+] GraalVM native-image found
) else (
    echo [*] GraalVM not found - using JNI ^(JRE required at runtime^)
    echo     Install: https://www.graalvm.org/downloads/
    echo     Then: gu install native-image
)

REM ── Compile Java ───────────────────────────────────────────────────
echo.
echo [*] Compiling Java classes...
set "JAVA_SRC=vidcrypt-java\vidcrypt-core\src\main\java\com\vidcrypt"
set "JAVA_OUT=vidcrypt-java\out"

if exist "%JAVA_OUT%" rmdir /s /q "%JAVA_OUT%"
mkdir "%JAVA_OUT%"

javac --enable-preview --release 21 -d "%JAVA_OUT%" --add-modules jdk.incubator.vector ^
    "%JAVA_SRC%\frame\NativeTileDecoder.java" ^
    "%JAVA_SRC%\frame\JniTileDecoder.java" ^
    "%JAVA_SRC%\frame\TileDecoder.java" ^
    "%JAVA_SRC%\frame\DecodeGeometry.java" ^
    "%JAVA_SRC%\calibration\CalParams.java" ^
    "%JAVA_SRC%\hash\Crc16.java"
if errorlevel 1 (
    echo [FAIL] Java compilation failed
    pause & exit /b 1
)
echo   [+] Classes compiled

REM ── Native lib or JAR ─────────────────────────────────────────────
set "USE_NATIVE_TILES=OFF"
set "USE_JNI_TILES=OFF"

if defined NATIVE_IMAGE (
    echo.
    echo [*] Building native shared library ^(GraalVM^)...
    
    "%NATIVE_IMAGE%" --shared -o vidcrypt_tiles ^
        -H:Name=vidcrypt_tiles ^
        -cp "%JAVA_OUT%" ^
        --no-fallback ^
        --enable-preview ^
        --add-modules jdk.incubator.vector ^
        com.vidcrypt.frame.NativeTileDecoder
    if errorlevel 1 (
        echo [FAIL] native-image failed
        pause & exit /b 1
    )
    
    set "USE_NATIVE_TILES=ON"
    echo   [+] vidcrypt_tiles.dll built
    echo   [+] Zero runtime Java dependency
) else (
    echo.
    echo [*] Building JAR ^(JNI fallback^)...
    jar cf vidcrypt-tiles.jar -C "%JAVA_OUT%" .
    set "USE_JNI_TILES=ON"
    echo   [+] vidcrypt-tiles.jar built
)

REM ── Build C ───────────────────────────────────────────────────────
echo.
echo [*] Building C engine...

set "BUILD_DIR=build_release"
if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
mkdir "%BUILD_DIR%"
cd "%BUILD_DIR%"

cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF -DENABLE_AVX2=ON ^
    -DUSE_NATIVE_TILES=%USE_NATIVE_TILES% -DUSE_JNI_TILES=%USE_JNI_TILES% ^
    -DJAVA_HOME="%JAVA_HOME%"
if errorlevel 1 (
    echo [FAIL] CMake configure failed
    pause & exit /b 1
)

cmake --build . --config Release --parallel
if errorlevel 1 (
    echo [FAIL] Build failed
    pause & exit /b 1
)

cd ..

REM ── Package ───────────────────────────────────────────────────────
echo.
echo [*] Packaging...

set "RELEASE_DIR=release"
if exist "%RELEASE_DIR%" rmdir /s /q "%RELEASE_DIR%"
mkdir "%RELEASE_DIR%"

copy /y "%BUILD_DIR%\Release\vidcrypt-encoder.exe" "%RELEASE_DIR%\" >nul
copy /y "%BUILD_DIR%\Release\vidcrypt-decoder.exe" "%RELEASE_DIR%\" >nul
copy /y "%BUILD_DIR%\vidcrypt-encoder.exe" "%RELEASE_DIR%\" >nul 2>nul
copy /y "%BUILD_DIR%\vidcrypt-decoder.exe" "%RELEASE_DIR%\" >nul 2>nul

if "%USE_NATIVE_TILES%"=="ON" (
    if exist vidcrypt_tiles.dll copy /y vidcrypt_tiles.dll "%RELEASE_DIR%\" >nul
    echo   [+] Native lib copied ^(no Java needed^)
) else (
    copy /y vidcrypt-tiles.jar "%RELEASE_DIR%\" >nul
    echo   [+] JAR copied ^(JRE required^)
)

echo.
echo ========================================
echo  Release built: %RELEASE_DIR%\
echo ========================================
dir /b "%RELEASE_DIR%"
echo.
echo Runtime: %RELEASE_DIR%\vidcrypt-decoder -b gpu -i video.mkv
if "%USE_NATIVE_TILES%"=="ON" (
    echo   Zero dependencies - native tile decoder embedded
) else (
    echo   Requires: Java 21+ JRE ^(for tile decoder^)
)
pause

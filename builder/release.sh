#!/usr/bin/env bash
# Release build: produces standalone binaries with zero runtime Java dependency.
# Requires GraalVM 21+ with native-image installed.
# Falls back to USE_JNI_TILES if GraalVM unavailable.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "========================================"
echo " VIDCRYPT Release Build"
echo "========================================"

# ── Check GraalVM ──────────────────────────────────────────────────
NATIVE_IMAGE=""
if command -v native-image &>/dev/null; then
    NATIVE_IMAGE="native-image"
    echo "[+] GraalVM native-image found"
elif [ -n "${GRAALVM_HOME:-}" ] && [ -x "$GRAALVM_HOME/bin/native-image" ]; then
    NATIVE_IMAGE="$GRAALVM_HOME/bin/native-image"
    echo "[+] GraalVM native-image: $GRAALVM_HOME"
elif [ -n "${JAVA_HOME:-}" ] && [ -x "$JAVA_HOME/bin/native-image" ]; then
    NATIVE_IMAGE="$JAVA_HOME/bin/native-image"
    echo "[+] GraalVM native-image: $JAVA_HOME"
else
    echo "[*] GraalVM native-image not found — using JNI (requires JVM at runtime)"
    echo "    Install: https://www.graalvm.org/downloads/"
    echo "    Then: gu install native-image"
fi

# ── Compile Java classes ───────────────────────────────────────────
JAVA_SRC="vidcrypt-java/vidcrypt-core/src/main/java/com/vidcrypt"
JAVA_OUT="vidcrypt-java/out"

echo ""
echo "[*] Compiling Java classes..."
rm -rf "$JAVA_OUT"
mkdir -p "$JAVA_OUT"

javac --enable-preview --release 21 -d "$JAVA_OUT" --add-modules jdk.incubator.vector \
    "$JAVA_SRC/frame/NativeTileDecoder.java" \
    "$JAVA_SRC/frame/JniTileDecoder.java" \
    "$JAVA_SRC/frame/TileDecoder.java" \
    "$JAVA_SRC/frame/DecodeGeometry.java" \
    "$JAVA_SRC/calibration/CalParams.java" \
    "$JAVA_SRC/hash/Crc16.java" \
    || { echo "[FAIL] Java compilation failed"; exit 1; }
echo "  [+] Classes compiled"

# ── Build native library (GraalVM) or JAR (fallback) ───────────────
USE_NATIVE_TILES="OFF"
USE_JNI_TILES="OFF"

if [ -n "$NATIVE_IMAGE" ]; then
    echo ""
    echo "[*] Building native shared library (GraalVM native-image)..."
    
    # Build shared library
    "$NATIVE_IMAGE" --shared -o vidcrypt_tiles \
        -H:Name=vidcrypt_tiles \
        -cp "$JAVA_OUT" \
        --no-fallback \
        --enable-preview \
        --add-modules jdk.incubator.vector \
        com.vidcrypt.frame.NativeTileDecoder \
        || { echo "[FAIL] native-image failed"; exit 1; }
    
    USE_NATIVE_TILES="ON"
    echo "  [+] vidcrypt_tiles.{so,dll} built"
    echo "  [+] Zero runtime Java dependency"
else
    echo ""
    echo "[*] Building JAR (JNI fallback)..."
    jar cf vidcrypt-tiles.jar -C "$JAVA_OUT" .
    USE_JNI_TILES="ON"
    echo "  [+] vidcrypt-tiles.jar built"
    echo "  [*] Requires JRE at runtime"
fi

# ── Build C ────────────────────────────────────────────────────────
echo ""
echo "[*] Building C engine..."

BUILD_DIR="build_release"
rm -rf "$BUILD_DIR"
mkdir "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$PROJECT_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA=OFF \
    -DENABLE_AVX2=ON \
    -DUSE_NATIVE_TILES="$USE_NATIVE_TILES" \
    -DUSE_JNI_TILES="$USE_JNI_TILES" \
    ${JAVA_HOME:+-DJAVA_HOME="$JAVA_HOME"} \
    || { echo "[FAIL] CMake configure failed"; exit 1; }

cmake --build . --parallel \
    || { echo "[FAIL] Build failed"; exit 1; }

cd "$PROJECT_DIR"

# ── Copy artifacts ─────────────────────────────────────────────────
echo ""
echo "[*] Packaging..."

RELEASE_DIR="release"
rm -rf "$RELEASE_DIR"
mkdir -p "$RELEASE_DIR"

cp "$BUILD_DIR/vidcrypt-encoder"* "$RELEASE_DIR/"
cp "$BUILD_DIR/vidcrypt-decoder"* "$RELEASE_DIR/"

if [ "$USE_NATIVE_TILES" = "ON" ]; then
    cp vidcrypt_tiles.* "$RELEASE_DIR/" 2>/dev/null || true
    echo "  [+] Native lib copied (no Java needed)"
elif [ "$USE_JNI_TILES" = "ON" ]; then
    cp vidcrypt-tiles.jar "$RELEASE_DIR/"
    echo "  [+] JAR copied (JRE required)"
fi

echo ""
echo "========================================"
echo " Release built: $RELEASE_DIR/"
echo "========================================"
ls -la "$RELEASE_DIR/"
echo ""
echo "Runtime: $RELEASE_DIR/vidcrypt-decoder -b gpu -i video.mkv"

if [ "$USE_NATIVE_TILES" = "ON" ]; then
    echo "  Zero dependencies — native tile decoder embedded"
else
    echo "  Requires: Java 21+ JRE (for tile decoder)"
    echo "  Or: install GraalVM and rebuild with native-image"
fi

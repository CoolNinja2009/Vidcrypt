#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Platform detection ──────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Darwin)  OS_NAME="macOS"   ; PKG_MGR="brew install"   ;;
    Linux)   OS_NAME="Linux"   ; PKG_MGR="apt install"    ;;
    MINGW*|MSYS*)  OS_NAME="Windows (MSYS2)" ; PKG_MGR="pacman -S" ;;
    *)       OS_NAME="$OS"     ; PKG_MGR="your package manager" ;;
esac

echo "========================================"
echo " VIDCRYPT-V8  --  Auto-Detect Build"
echo "========================================"
echo " Platform:  $OS_NAME  ($ARCH)"
echo ""

# ── Fail helpers ────────────────────────────────────────────────────
have()  { command -v "$1" &>/dev/null; }
die()   { echo "[FAIL] $*"; exit 1; }
warn()  { echo "  [!!] $*"; }
info()  { echo "  [*] $*"; }
found() { echo "  [+] $*"; }

# ── Check cmake ─────────────────────────────────────────────────────
echo "── Prerequisites ──"
have cmake || die "cmake not found. Install: $PKG_MGR cmake"
found "cmake $(cmake --version | head -1 | grep -oP '\d+\.\d+\.\d+')"

# ── Check C compiler ────────────────────────────────────────────────
CC_BIN=""
CC_VER=""
for cc in cc gcc clang; do
    if have "$cc"; then
        CC_BIN="$cc"
        CC_VER="$($cc --version 2>/dev/null | head -1)"
        break
    fi
done
[ -n "$CC_BIN" ] || die "No C compiler. Install: $PKG_MGR build-essential (or Xcode CLT on macOS)"
found "$CC_BIN  -  $CC_VER"

# ── CPU cores ───────────────────────────────────────────────────────
if have nproc; then
    NPROC=$(nproc)
elif [[ "$OS" == "Darwin" ]]; then
    NPROC=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
else
    NPROC=4
fi
info "CPU cores: $NPROC"

# ── AVX2 (x86-64 only) ─────────────────────────────────────────────
ENABLE_AVX2="ON"
case "$ARCH" in
    x86_64|amd64) ;;
    arm64|aarch64)
        ENABLE_AVX2="OFF"
        info "ARM64 detected — AVX2 disabled (scalar fallback active)"
        ;;
esac

# ── CUDA detection ──────────────────────────────────────────────────
USE_CUDA="OFF"
BUILD_DIR="$PROJECT_DIR/build"
BUILD_LABEL="CPU"

if [[ "$OS" == "Darwin" ]]; then
    info "macOS: CUDA not available (use Metal via FFmpeg VideoToolbox if needed)"
elif have nvcc; then
    CUDA_VER="$(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+' || echo '?')"
    if [[ "$ARCH" == "x86_64" || "$ARCH" == "amd64" ]]; then
        USE_CUDA="ON"
        BUILD_DIR="$PROJECT_DIR/build_cuda"
        BUILD_LABEL="GPU + CUDA $CUDA_VER"
        found "nvcc (CUDA $CUDA_VER)"
    else
        warn "CUDA found but arch is $ARCH (not x86_64) — skipping GPU build"
    fi
else
    info "nvcc not found — CPU-only build"
    if [[ "$OS" == "Linux" ]]; then
        info "  Install CUDA: https://developer.nvidia.com/cuda-downloads"
    fi
fi

# ── FFmpeg dev headers ──────────────────────────────────────────────
HAS_FFMPEG=""
if pkg-config --exists libavformat libavcodec libavutil 2>/dev/null; then
    HAS_FFMPEG=1
    found "FFmpeg dev (pkg-config)"
elif have ffmpeg; then
    found "ffmpeg binary (dev headers not found — NVDEC disabled)"
else
    info "ffmpeg not found — install: $PKG_MGR ffmpeg"
fi

echo ""

# ── Summary ─────────────────────────────────────────────────────────
echo "── Build plan ──"
echo "  Type:      $BUILD_LABEL"
echo "  Directory: $BUILD_DIR"
echo "  Flags:     USE_CUDA=$USE_CUDA  ENABLE_AVX2=$ENABLE_AVX2"
echo "  Parallel:  $NPROC jobs"
echo ""

# ── Create / enter build dir ────────────────────────────────────────
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# ── Configure CMake ─────────────────────────────────────────────────
echo "[*] Configuring CMake..."
cmake "$PROJECT_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA="$USE_CUDA" \
    -DENABLE_AVX2="$ENABLE_AVX2" \
    || die "CMake configuration failed. Check dependencies."

# ── Build ───────────────────────────────────────────────────────────
echo ""
echo "[*] Building ($NPROC parallel)..."
cmake --build . --config Release --parallel "$NPROC" \
    || die "Build failed."

# ── Done ────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo " Build complete — $BUILD_LABEL"
echo " Output: $BUILD_DIR/"
echo "========================================"
echo ""
echo " Encode:   $BUILD_DIR/vidcrypt-encoder -i input.zip -o encoded.mkv"
echo " Decode:   $BUILD_DIR/vidcrypt-decoder -i encoded.mkv"
if [[ "$USE_CUDA" == "ON" ]]; then
    echo " GPU dec:  $BUILD_DIR/vidcrypt-decoder -b gpu -i encoded.mkv"
fi
echo ""
echo " Tests:    cd $BUILD_DIR && ctest -C Release --output-on-failure"
echo ""

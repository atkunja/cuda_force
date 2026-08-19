#!/usr/bin/env bash
# Configure and build the C++ targets, and the CUDA targets when nvcc exists.
#
#   ./scripts/build.sh                     portable build
#   ./scripts/build.sh --cuda              add the CUDA targets
#   ./scripts/build.sh --sanitizer thread  build with ThreadSanitizer
#
# The portable targets never depend on CUDA, so this works unchanged on a host
# with no NVIDIA hardware.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BUILD_TYPE="RelWithDebInfo"
SANITIZER="OFF"
ENABLE_CUDA="OFF"
BUILD_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda) ENABLE_CUDA="ON"; shift ;;
    --debug) BUILD_TYPE="Debug"; shift ;;
    --sanitizer) SANITIZER="$2"; BUILD_TYPE="Debug"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ "$ENABLE_CUDA" == "ON" ]] && ! command -v nvcc >/dev/null 2>&1; then
  echo "error: --cuda requested but nvcc is not on PATH" >&2
  exit 1
fi

# Separate directories per configuration so switching sanitizers does not force
# a full rebuild of the others.
if [[ -z "$BUILD_DIR" ]]; then
  BUILD_DIR="build"
  [[ "$SANITIZER" != "OFF" ]] && BUILD_DIR="build-${SANITIZER}"
  [[ "$ENABLE_CUDA" == "ON" ]] && BUILD_DIR="${BUILD_DIR}-cuda"
fi

GENERATOR=()
command -v ninja >/dev/null 2>&1 && GENERATOR=(-G Ninja)

echo "==> configuring $BUILD_DIR (type=$BUILD_TYPE cuda=$ENABLE_CUDA sanitizer=$SANITIZER)"
cmake -S . -B "$BUILD_DIR" "${GENERATOR[@]}" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DCUDAFORGE_ENABLE_CUDA="$ENABLE_CUDA" \
  -DCUDAFORGE_SANITIZER="$SANITIZER"

echo "==> building"
cmake --build "$BUILD_DIR" --parallel

echo
echo "built into $BUILD_DIR"

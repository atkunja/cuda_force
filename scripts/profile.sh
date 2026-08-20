#!/usr/bin/env bash
# Collect Nsight Systems and Nsight Compute profiles.
#
# Requires an NVIDIA GPU and the Nsight tools. Not executable on the
# development host; see docs/profiling.md for how to read the output.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT="${OUTPUT:-profiles}"
mkdir -p "$OUTPUT"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "error: no NVIDIA GPU detected" >&2
  exit 1
fi

# Located rather than hardcoded: the target is declared in cuda/CMakeLists.txt,
# so it lands in build-cuda/cuda/. Naming build-cuda/benchmarks/ meant the check
# below never found it, the rebuild ran every time, and the profilers were then
# pointed at a path that still did not exist — which they reported as an error
# and this script swallowed, exiting 0. The validation harness duly recorded a
# PASS for a stage that profiled nothing.
find_bench_kernels() {
  local candidate
  for candidate in \
    build-cuda/cuda/bench_kernels \
    build-cuda/benchmarks/bench_kernels \
    build-cuda/bench_kernels
  do
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  find build-cuda -name bench_kernels -type f -perm -u+x 2>/dev/null | head -1
}

BINARY="$(find_bench_kernels)"
if [[ -z "$BINARY" ]]; then
  # Built with NVTX so the timeline is labelled by phase. Without it, a gap
  # between kernels cannot be attributed to formation, transfer or the host.
  echo "==> building CUDA benchmarks with NVTX"
  cmake -S . -B build-cuda -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCUDAFORGE_ENABLE_CUDA=ON \
    -DCUDAFORGE_ENABLE_NVTX=ON >/dev/null
  cmake --build build-cuda --parallel >/dev/null
  BINARY="$(find_bench_kernels)"
fi

# A profile of a binary that does not exist is not a profile. Failing here is
# the point: the alternative is a green stage and an empty report.
if [[ -z "$BINARY" ]]; then
  echo "error: bench_kernels not found under build-cuda after building" >&2
  exit 1
fi

# --- Nsight Systems: timeline ----------------------------------------------
# System-wide view. This is what shows whether copies and kernels actually
# overlap — the question the stream scheduler exists to answer. If the timeline
# shows serialised copy/compute, the usual causes are pageable host memory or a
# stray device-wide synchronisation.
if command -v nsys >/dev/null 2>&1; then
  echo "==> nsys profile"
  nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=cpu \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    --output="$OUTPUT/timeline-$STAMP" \
    "$BINARY" > /dev/null
  echo "    wrote $OUTPUT/timeline-$STAMP.nsys-rep"
  nsys stats --report cuda_gpu_kern_sum "$OUTPUT/timeline-$STAMP.nsys-rep" \
    | tee "$OUTPUT/kernel-summary-$STAMP.txt"
else
  echo "==> nsys not found; skipping the timeline profile"
fi

# --- Nsight Compute: per-kernel counters ------------------------------------
# Kernel-level hardware counters. Replays each kernel many times, so it is far
# slower than nsys and is pointed at a small set of kernels rather than a whole
# run.
#
# The sections requested here answer the questions that actually change a
# kernel: is it memory- or compute-bound (SpeedOfLight), are accesses coalesced
# (MemoryWorkloadAnalysis), is there enough parallelism to hide latency
# (Occupancy), and do warps diverge (WarpStateStats).
if command -v ncu >/dev/null 2>&1; then
  echo "==> ncu profile"
  ncu \
    --set full \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section Occupancy \
    --section WarpStateStats \
    --kernel-name-base demangled \
    --launch-count 3 \
    --force-overwrite \
    --export "$OUTPUT/kernels-$STAMP" \
    "$BINARY" > "$OUTPUT/ncu-$STAMP.txt" 2>&1 || true

  # ncu exits non-zero for many reasons and this script deliberately continues,
  # so the output is inspected rather than the status. Counter access is the
  # one failure worth naming: it is not a bug in anything here, and it is not
  # fixable from inside a container.
  if grep -q "ERR_NVGPUCTRPERM" "$OUTPUT/ncu-$STAMP.txt" 2>/dev/null; then
    echo "    ncu could not read performance counters (ERR_NVGPUCTRPERM)."
    echo "    NVIDIA restricts them to admin users by default. Enabling it needs"
    echo "    a host kernel-module flag (NVreg_RestrictProfilingToAdminUsers=0),"
    echo "    so on a rented container this is generally not obtainable — the"
    echo "    host would have to set it. Timings from ncu are also heavily"
    echo "    instrumented and are not benchmark numbers."
  else
    echo "    wrote $OUTPUT/kernels-$STAMP.ncu-rep"
  fi
else
  echo "==> ncu not found; skipping the kernel profile"
fi

echo
echo "profiles in $OUTPUT — see docs/profiling.md for what to look at"

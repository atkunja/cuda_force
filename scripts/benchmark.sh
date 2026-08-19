#!/usr/bin/env bash
# Run every benchmark this machine can run, writing JSON into benchmarks/results.
#
# Results are gitignored. Committed numbers would be numbers from someone else's
# machine, which is worse than no numbers at all.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

RESULTS="benchmarks/results"
mkdir -p "$RESULTS"

PYTHON="${PYTHON:-python}"
[[ -x .venv/bin/python ]] && PYTHON=".venv/bin/python"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
echo "==> results go to $RESULTS/*-$STAMP.json"

# --- C++ concurrency -------------------------------------------------------
if [[ ! -x build/benchmarks/bench_queue ]]; then
  echo "==> building benchmarks"
  ./scripts/build.sh >/dev/null
fi

for bench in queue scheduler memory; do
  binary="build/benchmarks/bench_${bench}"
  if [[ -x "$binary" ]]; then
    echo "==> $bench"
    "$binary" > "$RESULTS/cpp-${bench}-${STAMP}.json"
  else
    echo "==> skipping $bench (not built)"
  fi
done

# --- Python ----------------------------------------------------------------
echo "==> operator comparison"
"$PYTHON" benchmarks/benchmark_kernels.py \
  --output "$RESULTS/operators-${STAMP}.json"

echo "==> batching sweep"
"$PYTHON" benchmarks/benchmark_batching.py \
  --output "$RESULTS/batching-${STAMP}.json"

# --- CUDA ------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  if [[ ! -x build-cuda/benchmarks/bench_kernels ]]; then
    echo "==> building CUDA benchmarks"
    ./scripts/build.sh --cuda >/dev/null
  fi
  if [[ -x build-cuda/bench_kernels ]]; then
    echo "==> CUDA kernels"
    ./build-cuda/bench_kernels > "$RESULTS/cuda-kernels-${STAMP}.json"
  elif [[ -x build-cuda/benchmarks/bench_kernels ]]; then
    echo "==> CUDA kernels"
    ./build-cuda/benchmarks/bench_kernels > "$RESULTS/cuda-kernels-${STAMP}.json"
  fi
else
  cat <<'NOTE'

==> CUDA kernel benchmarks skipped

    No NVIDIA GPU was detected on this host, so no GPU numbers were produced.
    The harness is complete and runs unchanged on CUDA hardware; see
    docs/benchmarking.md for the exact commands.
NOTE
fi

echo
echo "done. results in $RESULTS"

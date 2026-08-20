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

for bench in queue scheduler memory histogram kv_cache; do
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

echo "==> metrics overhead"
"$PYTHON" benchmarks/benchmark_metrics.py \
  --output "$RESULTS/metrics-${STAMP}.json"

echo "==> continuous vs static batching"
"$PYTHON" benchmarks/benchmark_continuous.py \
  --output "$RESULTS/continuous-${STAMP}.json"

echo "==> batching sweep"
"$PYTHON" benchmarks/benchmark_batching.py \
  --output "$RESULTS/batching-${STAMP}.json"

# --- HTTP ------------------------------------------------------------------
# Only run when a server is already listening. Starting one here would make the
# script responsible for a process lifecycle it cannot supervise well.
if curl -sf "${CUDAFORGE_URL:-http://127.0.0.1:8000}/health" >/dev/null 2>&1; then
  echo "==> http server"
  "$PYTHON" benchmarks/benchmark_server.py \
    --url "${CUDAFORGE_URL:-http://127.0.0.1:8000}" --json \
    > "$RESULTS/http-${STAMP}.json"
else
  echo "==> skipping the HTTP benchmark (no server listening; start one with cudaforge-serve)"
fi

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

# A Markdown summary alongside the JSON. The JSON is what a script consumes;
# this is what goes into an issue or a report without pasting hundreds of lines.
SUMMARY="$RESULTS/summary-${STAMP}.md"
"$PYTHON" benchmarks/summarize_results.py "$RESULTS"/*-"${STAMP}".json > "$SUMMARY"

echo
echo "done. results in $RESULTS"
echo "summary: $SUMMARY"

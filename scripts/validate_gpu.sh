#!/usr/bin/env bash
# Everything that needs an NVIDIA GPU, in one command.
#
#   ./scripts/validate_gpu.sh
#
# Written for a rented GPU session, where the clock is running and improvising
# is expensive. It is deliberately opinionated about that:
#
#   * preflight first — a missing toolkit is reported in seconds, not after a
#     ten-minute build;
#   * every stage runs even if an earlier one failed, except the build, so one
#     session produces a complete picture rather than the first error;
#   * everything is logged and a Markdown report is written at the end, so the
#     results survive the machine being torn down;
#   * skips are reported explicitly. A suite that silently skips looks like a
#     suite that passed.
#
# Expect failures the first time. The CUDA sources compile in CI but have never
# executed; this is the run that finds out whether they are correct.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-validation-$STAMP}"
mkdir -p "$OUT"
LOG="$OUT/full.log"
REPORT="$OUT/report.md"

PYTHON="${PYTHON:-python3}"
[[ -x .venv/bin/python ]] && PYTHON=".venv/bin/python"

FAILED=0
declare -a RESULTS=()

say() { printf '\n\033[1m==> %s\033[0m\n' "$1" | tee -a "$LOG"; }

# Runs a stage, tees its output, times it, and records the verdict without
# aborting the session.
stage() {
  local name="$1"; shift
  local started elapsed status
  say "$name"
  started=$SECONDS
  if "$@" >>"$LOG" 2>&1; then
    status="PASS"
  else
    status="FAIL"
    FAILED=$((FAILED + 1))
  fi
  elapsed=$((SECONDS - started))
  RESULTS+=("$status|$name|${elapsed}s")
  printf '    [%s] %s (%ss)\n' "$status" "$name" "$elapsed" | tee -a "$LOG"
}

skip() {
  RESULTS+=("SKIP|$1|$2")
  printf '    [SKIP] %s — %s\n' "$1" "$2" | tee -a "$LOG"
}

# --- preflight --------------------------------------------------------------
# Cheap checks first: discovering there is no nvcc after a long build is the
# most expensive way to learn it.

say "preflight"
{
  echo "date: $(date -u)"
  echo "host: $(uname -a)"
} >>"$LOG" 2>&1

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "error: nvidia-smi not found — this script needs an NVIDIA GPU." >&2
  echo "       Everything runnable without one is in ./scripts/test.sh." >&2
  exit 1
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "error: nvcc not found — install the CUDA toolkit, or use the container:" >&2
  echo "       docker compose run --rm test" >&2
  exit 1
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
NVCC_VERSION="$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | head -1)"
CAPABILITY="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)"

printf '    GPU        %s\n    driver     %s\n    nvcc       %s\n    capability %s\n' \
  "$GPU_NAME" "$DRIVER" "$NVCC_VERSION" "${CAPABILITY:-unknown}" | tee -a "$LOG"

# The BF16 kernels need 8.0. Warn rather than refuse: everything else still runs.
if [[ -n "${CAPABILITY:-}" ]] && awk "BEGIN{exit !($CAPABILITY < 8.0)}"; then
  printf '    note: compute capability %s is below 8.0; the BF16 paths need 8.0\n' \
    "$CAPABILITY" | tee -a "$LOG"
fi

# Torch's CUDA must match nvcc's, or the extension build fails after several
# minutes with a message buried in a setuptools traceback. Checking it here
# costs milliseconds. It is a warning rather than an error: the C++ CUDA tests
# do not involve torch and are worth running either way.
TORCH_CUDA="$("$PYTHON" -c 'import torch; print(torch.version.cuda or "none")' 2>/dev/null || echo "no-torch")"
NVCC_SHORT="${NVCC_VERSION#release }"
printf '    python     %s\n    torch cuda %s\n' "$PYTHON" "$TORCH_CUDA" | tee -a "$LOG"
if [[ "$TORCH_CUDA" != "no-torch" && "$TORCH_CUDA" != "none" && "$TORCH_CUDA" != "$NVCC_SHORT" ]]; then
  printf '    warning: torch was built against CUDA %s but nvcc is %s.\n' \
    "$TORCH_CUDA" "$NVCC_SHORT" | tee -a "$LOG"
  printf '             The extension build will fail. Point PYTHON at an\n' | tee -a "$LOG"
  printf '             interpreter whose torch matches, e.g.\n' | tee -a "$LOG"
  printf '             PYTHON=/venv/main/bin/python %s\n' "$0" | tee -a "$LOG"
fi

nvidia-smi >>"$LOG" 2>&1

# --- build ------------------------------------------------------------------
# The one fatal stage: nothing downstream means anything without it.

say "building with CUDA"
if ! ./scripts/build.sh --cuda >>"$LOG" 2>&1; then
  echo "    [FAIL] CUDA build — see $LOG" | tee -a "$LOG"
  echo "    Nothing downstream can run. Stopping." | tee -a "$LOG"
  exit 1
fi
RESULTS+=("PASS|CUDA build|-")
echo "    [PASS] CUDA build" | tee -a "$LOG"

# --- correctness ------------------------------------------------------------
# The stage that matters. If these pass, the kernels compute what they claim;
# everything after this is performance.

stage "CUDA correctness tests" ./build-cuda/tests/cuda/cudaforge_cuda_tests
stage "portable C++ tests" ./build-cuda/tests/cpp/cudaforge_tests

# --- the PyTorch extension --------------------------------------------------

stage "build the CUDA PyTorch extension" \
  "$PYTHON" -m pip install -e . --no-build-isolation

say "active backend"
"$PYTHON" -c "from cudaforge.ops import backend_report; print(backend_report())" \
  | tee -a "$LOG" | sed 's/^/    /'

if "$PYTHON" -c "
import sys
from cudaforge.ops import backend_report
sys.exit(0 if backend_report().using_custom_kernels else 1)
" >>"$LOG" 2>&1; then
  RESULTS+=("PASS|custom CUDA kernels active|-")
  echo "    [PASS] custom CUDA kernels active" | tee -a "$LOG"
else
  RESULTS+=("FAIL|custom CUDA kernels active|-")
  FAILED=$((FAILED + 1))
  echo "    [FAIL] the extension is not dispatching to CUDA — later numbers are" \
    | tee -a "$LOG"
  echo "           the reference path, not the kernels." | tee -a "$LOG"
fi

stage "operator parity" "$PYTHON" examples/kernel_parity.py
stage "transformer block parity" "$PYTHON" examples/transformer_block.py
stage "Python tests (cuda-marked now run)" "$PYTHON" -m pytest tests/python -q

# --- measurement ------------------------------------------------------------

stage "benchmarks" ./scripts/benchmark.sh

if [[ -x build-cuda/benchmarks/bench_kernels ]]; then
  stage "CUDA kernel benchmarks" bash -c \
    "./build-cuda/benchmarks/bench_kernels > '$OUT/cuda-kernels.json'"
else
  skip "CUDA kernel benchmarks" "bench_kernels was not built"
fi

if command -v nsys >/dev/null 2>&1 || command -v ncu >/dev/null 2>&1; then
  stage "Nsight profiles" env OUTPUT="$OUT/profiles" ./scripts/profile.sh
else
  skip "Nsight profiles" "neither nsys nor ncu is installed"
fi

# --- optional extras --------------------------------------------------------

if "$PYTHON" -c "import bitsandbytes" >>"$LOG" 2>&1; then
  stage "QLoRA fine-tune" "$PYTHON" -m training.train \
    --config training/configs/tiny.yaml --load-in-4bit --max-steps 2
else
  skip "QLoRA" "bitsandbytes is not installed (pip install -e '.[quantize]')"
fi

GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
if [[ "$GPU_COUNT" -gt 1 ]] && command -v torchrun >/dev/null 2>&1; then
  stage "distributed training ($GPU_COUNT GPUs)" \
    torchrun --nproc_per_node="$GPU_COUNT" examples/distributed_train.py --epochs 1
else
  skip "distributed training" "needs more than one GPU (found $GPU_COUNT)"
fi

# --- report -----------------------------------------------------------------

collect_results() {
  local status name detail
  for row in "${RESULTS[@]}"; do
    IFS='|' read -r status name detail <<< "$row"
    printf '| %s | %s | %s |\n' "$name" "$status" "$detail"
  done
}

{
  echo "# GPU validation — $STAMP"
  echo
  echo "| Property | Value |"
  echo "| --- | --- |"
  echo "| GPU | $GPU_NAME |"
  echo "| Driver | $DRIVER |"
  echo "| nvcc | $NVCC_VERSION |"
  echo "| Compute capability | ${CAPABILITY:-unknown} |"
  echo
  echo "| Stage | Result | Time |"
  echo "| --- | --- | --- |"
  collect_results
  echo
  if [[ $FAILED -eq 0 ]]; then
    echo "All executed stages passed."
  else
    echo "**$FAILED stage(s) failed.** See \`full.log\`."
  fi
  echo
  echo "## Benchmark results"
  echo
} > "$REPORT"

if [[ -d benchmarks/results ]]; then
  # shellcheck disable=SC2046
  "$PYTHON" benchmarks/summarize_results.py benchmarks/results \
    >> "$REPORT" 2>>"$LOG" || true
fi
if [[ -f "$OUT/cuda-kernels.json" ]]; then
  "$PYTHON" benchmarks/summarize_results.py "$OUT/cuda-kernels.json" \
    >> "$REPORT" 2>>"$LOG" || true
fi

say "summary"
for row in "${RESULTS[@]}"; do
  IFS='|' read -r status name detail <<< "$row"
  printf '    [%s] %s\n' "$status" "$name"
done

echo
echo "report: $REPORT"
echo "log:    $LOG"
if [[ $FAILED -eq 0 ]]; then
  echo
  echo "All executed stages passed. Copy $OUT off this machine before it is"
  echo "destroyed, then fold the numbers into docs/performance.md and"
  echo "PROJECT_STATUS.md."
else
  echo
  echo "$FAILED stage(s) failed — which is the expected outcome of a first run,"
  echo "and the reason for doing it. Details are in $LOG."
fi
exit $FAILED

#!/usr/bin/env bash
# Run everything this machine is capable of running.
#
# Each stage reports whether it ran or was skipped and why. A suite that
# silently skips is worse than one that fails, because it looks like a pass.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

FAILURES=0
run_stage() {
  local name="$1"; shift
  echo
  echo "=============================================================="
  echo "  $name"
  echo "=============================================================="
  if "$@"; then
    echo "  [PASS] $name"
  else
    echo "  [FAIL] $name"
    FAILURES=$((FAILURES + 1))
  fi
}

skip_stage() {
  echo
  echo "  [SKIP] $1 — $2"
}

PYTHON="${PYTHON:-python}"
[[ -x .venv/bin/python ]] && PYTHON=".venv/bin/python"

# Printed first so the skips below are interpretable rather than mysterious.
echo
"$PYTHON" scripts/environment_report.py || true

# --- C++ -------------------------------------------------------------------
if command -v cmake >/dev/null 2>&1; then
  run_stage "C++ build" ./scripts/build.sh
  run_stage "C++ tests" ./build/tests/cpp/cudaforge_tests

  # CI builds with -Werror, and Linux compilers warn about things Apple clang
  # does not — an unused object with a non-trivial constructor, for one. Not
  # reproducing that locally is how a green local run and a red CI happen.
  run_stage "C++ build (warnings as errors)" bash -c \
    "cmake -S . -B build-werror -G Ninja -DCUDAFORGE_WARNINGS_AS_ERRORS=ON >/dev/null &&
     cmake --build build-werror"

# The batcher conformance tests need this harness; they skip without it, so
# building it here is what makes them actually run.
run_stage "C++ scenario harness" test -x ./build/tests/cpp/batcher_scenario

  # Sanitizers are mutually exclusive at the ABI level, so each needs its own
  # build directory and its own run.
  for sanitizer in address undefined thread; do
    run_stage "C++ tests (${sanitizer}sanitizer)" bash -c \
      "./scripts/build.sh --sanitizer $sanitizer >/dev/null && \
       ./build-${sanitizer}/tests/cpp/cudaforge_tests"
  done
else
  skip_stage "C++ tests" "cmake not found"
fi

# --- CUDA ------------------------------------------------------------------
run_stage "CUDA structural checks" "$PYTHON" scripts/check_cuda_sources.py cuda tests/cuda
run_stage "Documentation links" "$PYTHON" scripts/check_docs.py .
run_stage "Documented file paths" "$PYTHON" scripts/check_references.py .

if command -v nvcc >/dev/null 2>&1 && command -v nvidia-smi >/dev/null 2>&1; then
  run_stage "CUDA build" ./scripts/build.sh --cuda
  run_stage "CUDA tests" ./build-cuda/tests/cuda/cudaforge_cuda_tests
else
  skip_stage "CUDA build and tests" "no nvcc or no NVIDIA GPU on this host"
fi

# --- Python ----------------------------------------------------------------
# Invoked as a bare `pytest`, the way CI does. `python -m pytest` puts the
# current directory on sys.path and hides an import that only works from a
# checkout — which is exactly the failure this missed once.
PYTEST="pytest"
[[ -x .venv/bin/pytest ]] && PYTEST=".venv/bin/pytest"
run_stage "Python tests" "$PYTEST" tests/python -q --cov --cov-report=term:skip-covered

# Runs the examples as smoke tests. An example that no longer works is a
# documentation bug with a working-code disguise.
run_stage "Operator parity" "$PYTHON" examples/kernel_parity.py
run_stage "Transformer block parity" "$PYTHON" examples/transformer_block.py
run_stage "Example: single request" "$PYTHON" examples/simple_inference.py --echo-runner

if "$PYTHON" -c "import ruff" >/dev/null 2>&1 || command -v ruff >/dev/null 2>&1; then
  run_stage "ruff check" "$PYTHON" -m ruff check .
  run_stage "ruff format check" "$PYTHON" -m ruff format --check .
else
  skip_stage "ruff" "not installed"
fi

if "$PYTHON" -c "import mypy" >/dev/null 2>&1; then
  run_stage "mypy" "$PYTHON" -m mypy
else
  skip_stage "mypy" "not installed"
fi

echo
if [[ $FAILURES -eq 0 ]]; then
  echo "all executed stages passed"
else
  echo "$FAILURES stage(s) failed"
fi
exit $FAILURES

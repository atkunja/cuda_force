#!/usr/bin/env bash
# Formatting and static analysis across every language in the repository.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FIX=0
[[ "${1:-}" == "--fix" ]] && FIX=1

FAILURES=0
check() {
  local name="$1"; shift
  echo "==> $name"
  if ! "$@"; then
    echo "    FAILED"
    FAILURES=$((FAILURES + 1))
  fi
}

PYTHON="${PYTHON:-python}"
[[ -x .venv/bin/python ]] && PYTHON=".venv/bin/python"

# find -exec rather than mapfile: mapfile needs bash 4, and macOS still ships
# bash 3.2, so this script would fail on the platform it is most often run on.
CPP_SOURCES=(-name '*.cpp' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh')

if command -v clang-format >/dev/null 2>&1; then
  if [[ $FIX -eq 1 ]]; then
    check "clang-format (fixing)" \
      find cpp cuda tests benchmarks \( "${CPP_SOURCES[@]}" \) -exec clang-format -i {} +
  else
    check "clang-format" \
      find cpp cuda tests benchmarks \( "${CPP_SOURCES[@]}" \) \
        -exec clang-format --dry-run --Werror {} +
  fi
else
  echo "==> clang-format not found; skipping C++ formatting"
fi

if [[ $FIX -eq 1 ]]; then
  check "ruff (fixing)" "$PYTHON" -m ruff check --fix .
  check "ruff format" "$PYTHON" -m ruff format .
else
  check "ruff" "$PYTHON" -m ruff check .
  check "ruff format" "$PYTHON" -m ruff format --check .
fi

check "mypy" "$PYTHON" -m mypy
check "CUDA structural checks" "$PYTHON" scripts/check_cuda_sources.py cuda tests/cuda

echo
if [[ $FAILURES -eq 0 ]]; then
  echo "clean"
else
  echo "$FAILURES check(s) failed"
fi
exit $FAILURES

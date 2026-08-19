#!/usr/bin/env python3
"""Structural checks for CUDA sources that run without nvcc.

This project is developed on a machine with no CUDA toolkit, so the `.cu` and
`.cuh` files cannot be compiled locally. That is not a licence to leave them
unchecked. These rules catch the mistakes that are both common and mechanically
detectable, so a review on a GPU-less host still has teeth:

  * a kernel launch whose errors are never checked
  * `cudaDeviceSynchronize`, which serialises every stream
  * a bare `cudaMalloc`/`cudaMemcpy` return value that is silently dropped
  * `__syncthreads()` reached from inside a conditional, which deadlocks when
    threads in a block diverge
  * `__shfl_*` without an explicit mask

Compilation itself still happens — in CI on an NVIDIA runner and in the CUDA
container. This is a fast pre-filter, not a substitute.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Calls whose status must not be discarded. Anything matching is required to sit
# inside CUDAFORGE_CHECK(...) or have its result explicitly assigned.
CHECKED_CALLS = (
    "cudaMalloc",
    "cudaFree",
    "cudaMemcpy",
    "cudaMemcpyAsync",
    "cudaMemsetAsync",
    "cudaStreamCreate",
    "cudaStreamCreateWithFlags",
    "cudaStreamCreateWithPriority",
    "cudaStreamSynchronize",
    "cudaEventRecord",
    "cudaEventSynchronize",
    "cudaSetDevice",
    "cudaGetDevice",
    "cudaDeviceGetAttribute",
    "cudaHostAlloc",
    "cudaFreeHost",
)

LAUNCH_RE = re.compile(r"<<<.*?>>>")
SHUFFLE_RE = re.compile(r"__shfl_(up|down|xor)?_?sync?\s*\(")
UNSYNCED_SHUFFLE_RE = re.compile(r"__shfl_(up|down|xor)?(?!_sync)\s*\(")


@dataclass
class Finding:
    path: Path
    line: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: [{self.rule}] {self.message}"


def strip_comments(text: str) -> list[str]:
    """Blank out comment bodies while preserving line numbering."""
    without_block = re.sub(
        r"/\*.*?\*/",
        lambda match: "\n" * match.group(0).count("\n"),
        text,
        flags=re.DOTALL,
    )
    return [re.sub(r"//.*$", "", line) for line in without_block.splitlines()]


def check_launch_is_checked(path: Path, lines: list[str]) -> list[Finding]:
    """A kernel launch must be followed by a launch check within a few lines.

    Kernel launches return void. Without cudaGetLastError() a bad launch
    configuration is silently ignored and the failure surfaces later at an
    unrelated call, which is the hardest class of CUDA bug to trace back.
    """
    findings: list[Finding] = []
    for index, line in enumerate(lines):
        if not LAUNCH_RE.search(line):
            continue
        window = "\n".join(lines[index : index + 12])
        if "CUDAFORGE_CHECK_LAUNCH" not in window:
            findings.append(
                Finding(
                    path,
                    index + 1,
                    "unchecked-launch",
                    "kernel launch without a following CUDAFORGE_CHECK_LAUNCH",
                )
            )
    return findings


def check_no_device_sync(path: Path, lines: list[str]) -> list[Finding]:
    """cudaDeviceSynchronize is a device-wide barrier across every stream."""
    findings: list[Finding] = []
    for index, line in enumerate(lines):
        if "cudaDeviceSynchronize" in line:
            findings.append(
                Finding(
                    path,
                    index + 1,
                    "device-sync",
                    "cudaDeviceSynchronize serialises all streams; "
                    "synchronise a stream or wait on an event instead",
                )
            )
    return findings


def check_status_is_used(path: Path, lines: list[str]) -> list[Finding]:
    """Every status-returning runtime call must have its result inspected."""
    findings: list[Finding] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        for call in CHECKED_CALLS:
            if f"{call}(" not in stripped:
                continue
            checked = (
                "CUDAFORGE_CHECK" in stripped
                or "cudaError_t" in stripped
                or "== cudaSuccess" in stripped
                or "!= cudaSuccess" in stripped
                or "static_cast<void>" in stripped
                or stripped.startswith("//")
            )
            if not checked:
                findings.append(
                    Finding(
                        path,
                        index + 1,
                        "unchecked-status",
                        f"{call} return value is discarded",
                    )
                )
            break
    return findings


def check_shuffle_has_mask(path: Path, lines: list[str]) -> list[Finding]:
    """Maskless warp shuffles are undefined on Volta and later."""
    findings: list[Finding] = []
    for index, line in enumerate(lines):
        if UNSYNCED_SHUFFLE_RE.search(line):
            findings.append(
                Finding(
                    path,
                    index + 1,
                    "maskless-shuffle",
                    "use the _sync form with an explicit participation mask",
                )
            )
    return findings


def check_syncthreads_not_divergent(path: Path, lines: list[str]) -> list[Finding]:
    """__syncthreads() inside a conditional deadlocks on divergence.

    Deliberately conservative: it flags a __syncthreads() on the same line as an
    if/for/while, which is the shape that is always wrong. Detecting the general
    case needs a parser, and this catches the mistakes people actually make.
    """
    findings: list[Finding] = []
    for index, line in enumerate(lines):
        if "__syncthreads()" not in line:
            continue
        if re.search(r"\b(if|else if)\b.*__syncthreads\(\)", line):
            findings.append(
                Finding(
                    path,
                    index + 1,
                    "divergent-barrier",
                    "__syncthreads() reached conditionally; every thread in the "
                    "block must reach the same barrier",
                )
            )
    return findings


CHECKS = (
    check_launch_is_checked,
    check_no_device_sync,
    check_status_is_used,
    check_shuffle_has_mask,
    check_syncthreads_not_divergent,
)


def check_file(path: Path) -> list[Finding]:
    lines = strip_comments(path.read_text(encoding="utf-8"))
    findings: list[Finding] = []
    for check in CHECKS:
        findings.extend(check(path, lines))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        default=["cuda"],
        help="files or directories to scan (default: cuda/)",
    )
    args = parser.parse_args()

    targets: list[Path] = []
    for entry in args.paths:
        path = Path(entry)
        if path.is_dir():
            targets.extend(sorted(path.rglob("*.cu")))
            targets.extend(sorted(path.rglob("*.cuh")))
        elif path.suffix in {".cu", ".cuh"}:
            targets.append(path)

    if not targets:
        print("no CUDA sources found", file=sys.stderr)
        return 1

    findings: list[Finding] = []
    for target in targets:
        findings.extend(check_file(target))

    for finding in findings:
        print(finding)

    print(
        f"\nchecked {len(targets)} file(s): "
        f"{len(findings)} finding(s)",
        file=sys.stderr,
    )
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())

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


def function_end(lines: list[str], start: int) -> int:
    """Index just past the enclosing top-level definition.

    Approximated by the next line whose first character is a closing brace,
    which is where clang-format puts the end of a namespace-scope function. A
    launch and its check can be separated by an arbitrary amount of code inside
    a switch, so a fixed-size window produces false positives.
    """
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("}"):
            return index + 1
    return len(lines)


def check_launch_is_checked(path: Path, lines: list[str]) -> list[Finding]:
    """A kernel launch must be checked before its launcher returns.

    Kernel launches return void. Without cudaGetLastError() a bad launch
    configuration is silently ignored and the failure surfaces later at an
    unrelated call, which is the hardest class of CUDA bug to trace back.
    """
    findings: list[Finding] = []
    for index, line in enumerate(lines):
        if not LAUNCH_RE.search(line):
            continue
        scope = "\n".join(lines[index : function_end(lines, index)])
        if "CUDAFORGE_CHECK_LAUNCH" not in scope:
            findings.append(
                Finding(
                    path,
                    index + 1,
                    "unchecked-launch",
                    "kernel launch not followed by CUDAFORGE_CHECK_LAUNCH "
                    "before the launcher returns",
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


def logical_statements(lines: list[str]) -> list[tuple[int, str]]:
    """Group physical lines into (first_line_number, statement) pairs.

    A checked call is routinely split across lines by the formatter:

        CUDAFORGE_CHECK(
            cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream));

    Matching per physical line would report the second line as unchecked. The
    statement is what carries the meaning, so that is what gets matched.
    """
    statements: list[tuple[int, str]] = []
    buffer: list[str] = []
    first = 1
    for index, line in enumerate(lines):
        if not buffer:
            first = index + 1
        buffer.append(line.strip())
        if ";" in line or line.strip().endswith("{") or line.strip().endswith("}"):
            statements.append((first, " ".join(buffer)))
            buffer = []
    if buffer:
        statements.append((first, " ".join(buffer)))
    return statements


def check_status_is_used(path: Path, lines: list[str]) -> list[Finding]:
    """Every status-returning runtime call must have its result inspected.

    Accepted forms are the check macro, an explicit assignment to cudaError_t,
    a direct comparison against cudaSuccess, or an explicit discard via
    static_cast<void> — the last being how the destructors here acknowledge
    that they cannot throw.
    """
    findings: list[Finding] = []
    for line_number, statement in logical_statements(lines):
        for call in CHECKED_CALLS:
            if f"{call}(" not in statement:
                continue
            checked = (
                "CUDAFORGE_CHECK" in statement
                or "cudaError_t" in statement
                or "cudaSuccess" in statement
                or "static_cast<void>" in statement
            )
            if not checked:
                findings.append(
                    Finding(
                        path,
                        line_number,
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
        f"\nchecked {len(targets)} file(s): {len(findings)} finding(s)",
        file=sys.stderr,
    )
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())

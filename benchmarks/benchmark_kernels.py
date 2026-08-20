#!/usr/bin/env python3
"""Compare the custom operators against their PyTorch equivalents.

On a machine without an NVIDIA GPU this measures the reference implementations
against PyTorch's — which is a tautology for the ops that delegate, and is
reported as such rather than dressed up as a speedup. The comparison only means
something when ``backend.using_custom_kernels`` is true in the output.

Timing rules:

* CUDA work is timed with ``torch.cuda.Event``. A host timer around an
  asynchronous launch measures the launch, not the execution.
* Every case is warmed up. The first call pays for context creation, autotuning
  and lazy module loading.
* The median of N runs is reported. The mean is dragged by occasional
  scheduling interference; the median is not.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial

import torch

from cudaforge import ops


@dataclass
class Measurement:
    name: str
    variant: str
    shape: str
    median_ms: float
    min_ms: float
    p95_ms: float


def _time_cuda(fn: Callable[[], object], runs: int) -> list[float]:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(runs):
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))
    return samples


def _time_host(fn: Callable[[], object], runs: int) -> list[float]:
    samples = []
    for _ in range(runs):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1e3)
    return samples


def measure(
    name: str, variant: str, shape: str, fn: Callable[[], object], warmup: int, runs: int
) -> Measurement:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        samples = _time_cuda(fn, runs)
    else:
        samples = _time_host(fn, runs)

    ordered = sorted(samples)
    return Measurement(
        name=name,
        variant=variant,
        shape=shape,
        median_ms=statistics.median(ordered),
        min_ms=ordered[0],
        p95_ms=ordered[min(int(0.95 * len(ordered)), len(ordered) - 1)],
    )


def _torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The reference RMSNorm, as a function so it can be bound with `partial`."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * weight


def _torch_lora_linear(
    x: torch.Tensor, weight: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor
) -> torch.Tensor:
    """The reference LoRA projection, matching `ops.lora_linear`'s scale of 2.0."""
    return x @ weight + 2.0 * ((x @ lora_a) @ lora_b)


def run(device: torch.device, warmup: int, runs: int) -> list[Measurement]:
    """Build every case and time it.

    Each case binds its tensors with `functools.partial`. Python closes over the
    *variable*, not its value, so a bare `lambda: ops.rmsnorm(x, weight)` inside
    a loop would capture whichever tensors the loop had reached by the time it
    ran — the classic late-binding trap, and one that would silently benchmark
    the same shape repeatedly.

    `partial` rather than the `lambda x=x` idiom, which binds correctly but
    gives the callable parameters it does not want: it no longer matches the
    `Callable[[], object]` that `measure` takes, and the type checker cannot see
    that the defaults make it callable with no arguments.
    """
    results: list[Measurement] = []

    for rows, cols in [(1024, 512), (2048, 4096), (512, 8192)]:
        x = torch.randn(rows, cols, device=device)
        weight = torch.randn(cols, device=device)
        shape = f"{rows}x{cols}"

        results.append(
            measure(
                "rmsnorm",
                "cudaforge",
                shape,
                partial(ops.rmsnorm, x, weight),
                warmup,
                runs,
            )
        )
        results.append(
            measure(
                "rmsnorm",
                "torch",
                shape,
                partial(_torch_rmsnorm, x, weight),
                warmup,
                runs,
            )
        )

        results.append(
            measure("softmax", "cudaforge", shape, partial(ops.softmax, x), warmup, runs)
        )
        results.append(
            measure("softmax", "torch", shape, partial(torch.softmax, x, dim=-1), warmup, runs)
        )

    for batch, in_features, out_features, rank in [(32, 1024, 1024, 8), (128, 4096, 4096, 16)]:
        x = torch.randn(batch, in_features, device=device)
        weight = torch.randn(in_features, out_features, device=device)
        lora_a = torch.randn(in_features, rank, device=device)
        lora_b = torch.randn(rank, out_features, device=device)
        shape = f"{batch}x{in_features}x{out_features}r{rank}"

        results.append(
            measure(
                "lora_linear",
                "cudaforge",
                shape,
                partial(ops.lora_linear, x, weight, lora_a, lora_b, 2.0),
                warmup,
                runs,
            )
        )
        results.append(
            measure(
                "lora_linear",
                "torch",
                shape,
                partial(_torch_lora_linear, x, weight, lora_a, lora_b),
                warmup,
                runs,
            )
        )

    for count in [1 << 20, 1 << 24]:
        x = torch.randn(count, device=device)
        results.append(
            measure(
                "quantize_int8",
                "cudaforge",
                str(count),
                partial(ops.quantize_int8, x),
                warmup,
                runs,
            )
        )
        results.append(
            measure("sum", "cudaforge", str(count), partial(ops.sum_reduce, x), warmup, runs)
        )
        results.append(measure("sum", "torch", str(count), partial(torch.sum, x), warmup, runs))

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--output", help="write JSON here instead of stdout")
    args = parser.parse_args(argv)

    report = ops.backend_report()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = {
        "benchmark": "operators",
        "backend": {
            "description": str(report),
            "using_custom_kernels": report.using_custom_kernels,
            "extension_loaded": report.extension_loaded,
            "cuda_compiled": report.cuda_compiled,
            "device": report.device_name or platform.processor() or platform.machine(),
        },
        "torch_version": torch.__version__,
        "warmup": args.warmup,
        "runs": args.runs,
        "results": [asdict(measurement) for measurement in run(device, args.warmup, args.runs)],
    }

    if not report.using_custom_kernels:
        # Stated in the output itself, so a results file cannot be mistaken for
        # a custom-kernel comparison later.
        payload["caveat"] = (
            "Custom CUDA kernels were not used. Both columns are PyTorch "
            "reference implementations, so the comparison shows dispatch "
            "overhead only, not kernel performance."
        )

    text = json.dumps(payload, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

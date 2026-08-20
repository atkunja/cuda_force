#!/usr/bin/env python3
"""Check every custom kernel against its reference on your own GPU.

    python examples/kernel_parity.py

This is the check worth running first on new hardware. Performance is a
question you can defer; whether the kernels compute the right thing is not, and
a kernel that is wrong on your architecture will otherwise surface as a subtly
degraded model rather than as an error.

On a machine with no GPU this compares the reference implementations against
themselves, reports that it did so, and exits successfully — there is nothing
to verify, and pretending otherwise would be worse than saying so.

Shapes deliberately include non-power-of-two, very small and large cases, plus
widths that are not multiples of four so the vectorised RMSNorm's fallback path
is exercised rather than assumed.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import torch

from cudaforge import ops


@dataclass
class Check:
    kernel: str
    shape: str
    max_abs_error: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return self.max_abs_error <= self.tolerance

    def __str__(self) -> str:
        mark = "ok  " if self.passed else "FAIL"
        return (
            f"  [{mark}] {self.kernel:<16} {self.shape:<22} "
            f"max |Δ| = {self.max_abs_error:.3e}  (tol {self.tolerance:.1e})"
        )


def compare(
    kernel: str, shape: str, actual: torch.Tensor, expected: torch.Tensor, tolerance: float
) -> Check:
    error = (actual.float() - expected.float()).abs().max().item()
    return Check(kernel, shape, error, tolerance)


def run(device: torch.device) -> list[Check]:
    checks: list[Check] = []

    # Widths mix multiples of four with sizes that are not, so both the
    # vectorised path and its scalar fallback are covered.
    for rows, cols in [(1, 1), (4, 17), (8, 128), (33, 1023), (16, 4096)]:
        x = torch.randn(rows, cols, device=device)
        weight = torch.randn(cols, device=device)
        shape = f"{rows}x{cols}"

        expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * weight
        checks.append(compare("rmsnorm", shape, ops.rmsnorm(x, weight), expected, 1e-4))

        checks.append(compare("softmax", shape, ops.softmax(x), torch.softmax(x, dim=-1), 1e-5))

    # Logits that overflow exp() without the max-subtraction.
    extreme = torch.tensor([[1000.0, 1000.0, 1000.0], [-1000.0, 0.0, 1000.0]], device=device)
    checks.append(
        compare("softmax(±1e3)", "2x3", ops.softmax(extreme), torch.softmax(extreme, dim=-1), 1e-6)
    )

    lora_shapes = [(1, 8, 4, 2), (7, 129, 65, 3), (64, 1024, 1024, 16)]
    for batch, in_features, out_features, rank in lora_shapes:
        x = torch.randn(batch, in_features, device=device)
        weight = torch.randn(in_features, out_features, device=device)
        lora_a = torch.randn(in_features, rank, device=device)
        lora_b = torch.randn(rank, out_features, device=device)
        expected = x @ weight + 2.0 * ((x @ lora_a) @ lora_b)
        checks.append(
            compare(
                "lora_linear",
                f"{batch}x{in_features}x{out_features}r{rank}",
                ops.lora_linear(x, weight, lora_a, lora_b, 2.0),
                expected,
                1e-2,
            )
        )

    for dims in [(64,), (4, 17), (8, 4096)]:
        gate = torch.randn(*dims, device=device)
        up = torch.randn(*dims, device=device)
        label = "x".join(str(dim) for dim in dims)

        checks.append(compare("silu", label, ops.silu(gate), torch.nn.functional.silu(gate), 1e-5))
        checks.append(
            compare(
                "gelu",
                label,
                ops.gelu(gate),
                torch.nn.functional.gelu(gate, approximate="tanh"),
                1e-5,
            )
        )
        checks.append(
            compare(
                "swiglu",
                label,
                ops.swiglu(gate, up),
                torch.nn.functional.silu(gate) * up,
                1e-5,
            )
        )

    for count in [1, 63, 64, 65, 1 << 20]:
        x = torch.randn(count, device=device)
        checks.append(compare("sum", str(count), ops.sum_reduce(x), x.sum(), 1e-2))

        # The bound is a property of the scheme, not of the implementation:
        # symmetric absmax rounding cannot err by more than half a step.
        quantised, scales = ops.quantize_int8(x)
        restored = ops.dequantize_int8(quantised, scales)
        bound = scales.max().item() / 2 + 1e-6
        checks.append(
            Check("quant round trip", str(count), (x - restored).abs().max().item(), bound)
        )

    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)

    report = ops.backend_report()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(report)
    print(f"device: {device}\n")

    if not report.using_custom_kernels:
        print(
            "No custom CUDA kernels are active, so this run compares the reference\n"
            "implementations against themselves. It verifies nothing about kernel\n"
            "correctness. Run it on an NVIDIA GPU with the extension built.\n"
        )

    checks = run(device)
    for check in checks:
        print(check)

    failures = [check for check in checks if not check.passed]
    print(f"\n{len(checks) - len(failures)}/{len(checks)} checks passed")
    if failures:
        print("\nFAILURES:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

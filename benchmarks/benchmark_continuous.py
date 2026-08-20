#!/usr/bin/env python3
"""Continuous batching against static batching, on the same workload.

    python benchmarks/benchmark_continuous.py

Static batching holds a batch until its *longest* member finishes, so rows freed
by short sequences sit idle. Continuous batching refills them at the next decode
step. The size of that difference is entirely a property of the generation-length
distribution, which is why this sweeps several rather than reporting one number.

Both sides drive the identical step-wise runner, so the difference is
attributable to scheduling alone and not to anything about the model.

The runner is simulated. These figures describe the **scheduler** — how many
decode steps a workload costs, and what fraction of the batch was occupied — not
the throughput of any real model. That separation is deliberate: it isolates the
variable and keeps the benchmark runnable without a GPU.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
from dataclasses import asdict, dataclass

from cudaforge.config import GenerationConfig
from cudaforge.continuous import ContinuousBatcher, ContinuousStats, run_static
from cudaforge.scheduler import Request
from cudaforge.stepwise import SequenceState
from cudaforge.stepwise import EchoStepwiseRunner

#: A decode step must cost something, or the producer becomes the bottleneck and
#: the comparison measures submission speed instead of scheduling. On real
#: hardware a step is milliseconds; this is enough to keep the queue backed up.
STEP_SECONDS = 0.0002


@dataclass
class Comparison:
    workload: str
    description: str
    sequences: int
    total_tokens: int
    median_length: int
    max_length: int
    max_batch_size: int
    ideal_steps: int
    static_steps: int
    continuous_steps: int
    static_utilisation: float
    continuous_utilisation: float

    @property
    def step_reduction(self) -> float:
        if self.static_steps == 0:
            return 0.0
        return 1.0 - self.continuous_steps / self.static_steps


def workloads(count: int, seed: int) -> list[tuple[str, str, list[int]]]:
    rng = random.Random(seed)
    return [
        (
            "lognormal",
            "long tail, as real chat traffic looks",
            [max(4, int(rng.lognormvariate(2.4, 0.9))) for _ in range(count)],
        ),
        (
            "bimodal",
            "short answers with an occasional long generation",
            [4 if rng.random() < 0.9 else 200 for _ in range(count)],
        ),
        (
            "uniform",
            "uniform over a wide range",
            [rng.randint(4, 120) for _ in range(count)],
        ),
        (
            "constant",
            "every sequence the same length — nothing for refilling to recover",
            [32] * count,
        ),
    ]


def run_continuous(lengths: list[int], max_batch_size: int) -> ContinuousStats:
    completed = threading.Event()
    seen = 0
    lock = threading.Lock()

    def on_complete(_request: Request, _state: SequenceState) -> None:
        nonlocal seen
        with lock:
            seen += 1
            if seen >= len(lengths):
                completed.set()

    runner = EchoStepwiseRunner(per_step_seconds=STEP_SECONDS)
    batcher = ContinuousBatcher(
        runner,
        on_complete,
        max_batch_size=max_batch_size,
        queue_capacity=max(1024, len(lengths) * 2),
    )
    for index, length in enumerate(lengths):
        batcher.submit(
            Request(prompt=f"p{index}", generation=GenerationConfig(max_new_tokens=length))
        )
    completed.wait(timeout=600)
    stats = batcher.stats()
    batcher.shutdown()
    return stats


def compare(name: str, description: str, lengths: list[int], max_batch: int) -> Comparison:
    work = [
        (f"p{index}", GenerationConfig(max_new_tokens=length))
        for index, length in enumerate(lengths)
    ]
    static = run_static(EchoStepwiseRunner(per_step_seconds=STEP_SECONDS), work, max_batch)
    continuous = run_continuous(lengths, max_batch)

    total = sum(lengths)
    return Comparison(
        workload=name,
        description=description,
        sequences=len(lengths),
        total_tokens=total,
        median_length=int(statistics.median(lengths)),
        max_length=max(lengths),
        max_batch_size=max_batch,
        # No schedule can beat both the total work spread across the batch and
        # the single longest sequence. Reporting it keeps the comparison honest:
        # continuous batching approaches this, it does not beat it.
        ideal_steps=max((total + max_batch - 1) // max_batch, max(lengths)),
        static_steps=static.decode_steps,
        continuous_steps=continuous.decode_steps,
        static_utilisation=static.utilisation,
        continuous_utilisation=continuous.utilisation,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=int, default=200)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    results = [
        compare(name, description, lengths, args.max_batch_size)
        for name, description, lengths in workloads(args.sequences, args.seed)
    ]

    payload = {
        "benchmark": "continuous_batching",
        "note": (
            "the runner is simulated; these measure the scheduler in decode "
            "steps and batch occupancy, not model throughput"
        ),
        "max_batch_size": args.max_batch_size,
        "sequences": args.sequences,
        "comparisons": [asdict(result) for result in results],
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print(f"wrote {args.output}", file=sys.stderr)
        return 0

    print(payload["note"])
    print(f"\n{args.sequences} sequences, max_batch_size {args.max_batch_size}\n")
    header = (
        f"{'workload':<12}{'median':>8}{'max':>6}{'ideal':>8}"
        f"{'static':>8}{'contin.':>9}{'fewer':>8}{'util':>16}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        util = (
            f"{result.static_utilisation * 100:.0f}% -> {result.continuous_utilisation * 100:.0f}%"
        )
        print(
            f"{result.workload:<12}{result.median_length:>8}{result.max_length:>6}"
            f"{result.ideal_steps:>8}{result.static_steps:>8}"
            f"{result.continuous_steps:>9}{result.step_reduction * 100:>7.0f}%{util:>16}"
        )

    print(
        "\n`ideal` is the lower bound: the total work spread across the batch, or "
        "the\nlongest single sequence, whichever is larger. Continuous batching "
        "approaches it;\nnothing beats it. The constant workload shows the "
        "honest floor — when every\nsequence is the same length, no row is ever "
        "freed early and there is nothing to\nrecover."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Sweep the batching parameters and report the throughput/latency tradeoff.

The two knobs pull in opposite directions:

* ``max_batch_size`` raises throughput by amortising the fixed per-batch cost
  over more rows, and raises latency because a bigger batch takes longer to
  fill and longer to run.
* ``max_wait_us`` bounds how long an early arrival is held waiting for company.
  It is the direct upper bound on batching-induced queue delay, so it is the
  main p99 lever.

The output is what shows where each stops paying: throughput that has flattened
while p99 keeps climbing means the batch is already large enough, and a
``timeout_closure_fraction`` near 1.0 means the arrival rate never fills a
batch, so the configured wait is pure added latency.

Run against the deterministic runner by default, so the numbers describe the
*scheduler* rather than a particular model. Point it at a real model with
``--model`` when you want end-to-end figures.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import asdict, dataclass

from cudaforge.config import EngineConfig, GenerationConfig
from cudaforge.engine import EngineClosedError, InferenceEngine
from cudaforge.ops import backend_report
from cudaforge.runners import EchoRunner, ModelRunner, TransformersRunner


@dataclass
class CaseResult:
    clients: int
    max_batch_size: int
    max_wait_us: int
    completed: int
    rejected: int
    wall_seconds: float
    requests_per_second: float
    tokens_per_second: float
    average_batch_size: float
    timeout_closure_fraction: float
    queue_delay_p50_ms: float
    queue_delay_p99_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float


def drive(engine: InferenceEngine, clients: int, per_client: int, tokens: int) -> tuple[int, int]:
    """Independent client threads, so the arrival pattern resembles real traffic."""
    completed = 0
    rejected = 0
    lock = threading.Lock()
    generation = GenerationConfig(max_new_tokens=tokens)

    def worker(index: int) -> None:
        nonlocal completed, rejected
        local_ok = local_reject = 0
        for i in range(per_client):
            try:
                if engine.generate(f"client-{index}-req-{i}", generation).ok:
                    local_ok += 1
            except EngineClosedError:
                local_reject += 1
        with lock:
            completed += local_ok
            rejected += local_reject

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(clients)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return completed, rejected


def run_case(
    clients: int,
    per_client: int,
    max_batch_size: int,
    max_wait_us: int,
    tokens: int,
    model: str | None,
    per_token_seconds: float,
    fixed_overhead: float,
) -> CaseResult:
    config = EngineConfig(
        model_name=model or "sshleifer/tiny-gpt2",
        max_batch_size=max_batch_size,
        max_wait_us=max_wait_us,
        queue_capacity=max(1024, max_batch_size * 4),
        worker_threads=4,
        warmup_iterations=2,
    )
    runner: ModelRunner = (
        TransformersRunner(config)
        if model
        else EchoRunner(per_token_seconds=per_token_seconds, fixed_overhead=fixed_overhead)
    )

    with InferenceEngine(config=config, runner=runner) as engine:
        started = time.monotonic()
        completed, rejected = drive(engine, clients, per_client, tokens)
        wall = time.monotonic() - started
        snapshot = engine.snapshot()

    closures = snapshot.batches_processed or 1
    return CaseResult(
        clients=clients,
        max_batch_size=max_batch_size,
        max_wait_us=max_wait_us,
        completed=completed,
        rejected=rejected,
        wall_seconds=round(wall, 4),
        requests_per_second=snapshot.requests_per_second,
        tokens_per_second=snapshot.tokens_per_second,
        average_batch_size=snapshot.average_batch_size,
        timeout_closure_fraction=snapshot.batches_closed_by_timeout / closures,
        queue_delay_p50_ms=snapshot.queue_delay_p50_ms,
        queue_delay_p99_ms=snapshot.queue_delay_p99_ms,
        latency_p50_ms=snapshot.latency_p50_ms,
        latency_p95_ms=snapshot.latency_p95_ms,
        latency_p99_ms=snapshot.latency_p99_ms,
    )


def print_table(results: list[CaseResult]) -> None:
    header = (
        f"{'clients':>8} {'batch':>6} {'wait_us':>8} {'req/s':>10} {'avg_batch':>10} "
        f"{'timeout%':>9} {'p50_ms':>9} {'p99_ms':>9}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.clients:>8} {result.max_batch_size:>6} {result.max_wait_us:>8} "
            f"{result.requests_per_second:>10.1f} {result.average_batch_size:>10.2f} "
            f"{result.timeout_closure_fraction * 100:>8.0f}% "
            f"{result.latency_p50_ms:>9.2f} {result.latency_p99_ms:>9.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 32])
    parser.add_argument("--waits-us", type=int, nargs="+", default=[500, 2000, 5000])
    parser.add_argument("--requests-per-client", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--model",
        help="load a real model instead of the deterministic runner (much slower)",
    )
    parser.add_argument(
        "--per-token-seconds",
        type=float,
        default=0.0002,
        help="simulated per-token cost for the deterministic runner",
    )
    parser.add_argument(
        "--fixed-overhead",
        type=float,
        default=0.002,
        help="simulated per-batch fixed cost; this is what batching amortises",
    )
    parser.add_argument("--output", help="write JSON here instead of a table")
    args = parser.parse_args(argv)

    results = [
        run_case(
            clients,
            args.requests_per_client,
            batch_size,
            wait,
            args.max_new_tokens,
            args.model,
            args.per_token_seconds,
            args.fixed_overhead,
        )
        for clients in args.clients
        for batch_size in args.batch_sizes
        for wait in args.waits_us
    ]

    payload = {
        "benchmark": "dynamic_batching",
        "backend": str(backend_report()),
        "runner": "transformers" if args.model else "deterministic",
        "note": (
            "Execution is simulated unless --model is given; these numbers "
            "describe the scheduler, not model throughput."
        ),
        "requests_per_client": args.requests_per_client,
        "max_new_tokens": args.max_new_tokens,
        "cases": [asdict(result) for result in results],
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(payload["backend"])
        print(payload["note"])
        print()
        print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

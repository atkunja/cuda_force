#!/usr/bin/env python3
"""Show what dynamic batching actually buys.

    python examples/concurrent_requests.py

Runs the same workload twice against the same runner, changing only
``max_batch_size``. The first pass has batching effectively disabled
(``max_batch_size=1``), the second allows aggregation. Everything else is held
constant, so the difference is attributable to batching alone.

The simulated runner has a fixed per-batch cost and a per-token cost, mirroring
real inference where a batch reads the weights once regardless of how many rows
it contains. That fixed cost is what batching amortises; without it there would
be nothing to gain and the comparison would show nothing.
"""

from __future__ import annotations

import argparse
import threading
import time

from cudaforge.config import EngineConfig, GenerationConfig
from cudaforge.engine import InferenceEngine
from cudaforge.metrics import MetricsSnapshot
from cudaforge.runners import EchoRunner


def run(clients: int, per_client: int, max_batch_size: int, wait_us: int) -> MetricsSnapshot:
    config = EngineConfig(
        max_batch_size=max_batch_size,
        max_wait_us=wait_us,
        queue_capacity=max(1024, max_batch_size * 4),
        worker_threads=4,
        warmup_iterations=1,
    )
    runner = EchoRunner(per_token_seconds=0.0002, fixed_overhead=0.004)
    generation = GenerationConfig(max_new_tokens=16)

    with InferenceEngine(config=config, runner=runner) as engine:

        def worker(index: int) -> None:
            for i in range(per_client):
                engine.generate(f"client-{index}-request-{i}", generation, timeout=120)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(clients)]
        started = time.monotonic()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        elapsed = time.monotonic() - started
        snapshot = engine.snapshot()

    snapshot.extra["wall_seconds"] = elapsed
    return snapshot


def report(label: str, snapshot: MetricsSnapshot) -> None:
    print(f"\n{label}")
    print(f"  wall time         {snapshot.extra['wall_seconds']:.3f} s")
    print(f"  throughput        {snapshot.requests_per_second:.1f} req/s")
    print(f"  average batch     {snapshot.average_batch_size:.2f}")
    print(f"  batches           {snapshot.batches_processed}")
    print(f"  queue delay p99   {snapshot.queue_delay_p99_ms:.2f} ms")
    print(f"  latency p50       {snapshot.latency_p50_ms:.2f} ms")
    print(f"  latency p99       {snapshot.latency_p99_ms:.2f} ms")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", type=int, default=16)
    parser.add_argument("--requests-per-client", type=int, default=20)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--wait-us", type=int, default=5_000)
    args = parser.parse_args(argv)

    unbatched = run(args.clients, args.requests_per_client, 1, args.wait_us)
    batched = run(args.clients, args.requests_per_client, args.max_batch_size, args.wait_us)

    print(f"{args.clients} concurrent clients x {args.requests_per_client} requests")
    report("max_batch_size = 1 (batching disabled)", unbatched)
    report(f"max_batch_size = {args.max_batch_size}", batched)

    if unbatched.requests_per_second > 0:
        ratio = batched.requests_per_second / unbatched.requests_per_second
        print(f"\nthroughput ratio    {ratio:.2f}x")
        print(
            "\nMeasured against a simulated runner on this machine. The ratio "
            "depends entirely on\nthe fixed-to-variable cost split of the model "
            "being served; treat it as a demonstration\nof the mechanism, not as "
            "a performance claim about any real model."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Load-test the HTTP server end to end.

    cudaforge-serve --echo-runner &
    python benchmarks/benchmark_server.py --requests 500 --concurrency 32

`cudaforge-bench` drives the engine in-process, which is the right way to
measure the *scheduler* — it excludes HTTP entirely. This measures what a client
actually experiences: connection handling, serialisation, the event loop, and
the queue behind them.

The difference between the two is the server's overhead, and it is worth knowing
which of the two a latency number came from.

Client-side latency is measured per request, so the percentiles here include
queueing *and* transport. The server's own `/metrics` are fetched afterwards for
comparison; a large gap between the two p99s is HTTP overhead or client-side
contention, not the runtime.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field

try:
    import httpx
except ImportError:  # pragma: no cover - exercised only without the extra
    print(
        "httpx is required: pip install httpx (or install the 'serve' extra)",
        file=sys.stderr,
    )
    raise SystemExit(1) from None


@dataclass
class Outcome:
    latencies: list[float] = field(default_factory=list)
    completed: int = 0
    shed: int = 0
    failed: int = 0


async def one_request(
    client: httpx.AsyncClient, url: str, prompt: str, tokens: int, deadline: float | None
) -> tuple[float, int]:
    body: dict[str, object] = {"prompt": prompt, "max_new_tokens": tokens}
    if deadline is not None:
        body["deadline_seconds"] = deadline

    started = time.perf_counter()
    response = await client.post(url, json=body)
    return time.perf_counter() - started, response.status_code


async def drive(
    base_url: str, total: int, concurrency: int, tokens: int, deadline: float | None
) -> tuple[Outcome, float]:
    outcome = Outcome()
    # A semaphore rather than batched gathers: it keeps exactly `concurrency`
    # requests in flight for the whole run, which is what a steady offered load
    # looks like. Batched gathers produce a sawtooth as each batch drains.
    gate = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(300.0),
        limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
    ) as client:

        async def worker(index: int) -> None:
            async with gate:
                try:
                    latency, status = await one_request(
                        client, f"{base_url}/generate", f"request-{index}", tokens, deadline
                    )
                except httpx.HTTPError:
                    outcome.failed += 1
                    return

                outcome.latencies.append(latency)
                if status == 200:
                    outcome.completed += 1
                elif status == 503:
                    # Deliberate load shedding, not an error.
                    outcome.shed += 1
                else:
                    outcome.failed += 1

        started = time.perf_counter()
        await asyncio.gather(*(worker(i) for i in range(total)))
        elapsed = time.perf_counter() - started

    return outcome, elapsed


def percentile(samples: list[float], quantile: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    index = min(int(quantile * (len(ordered) - 1)), len(ordered) - 1)
    return ordered[index]


async def fetch_metrics(base_url: str) -> dict[str, object]:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{base_url}/metrics")
            response.raise_for_status()
            return dict(response.json())
    except httpx.HTTPError as error:
        return {"error": str(error)}


async def run(args: argparse.Namespace) -> int:
    base_url = args.url.rstrip("/")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            health = (await client.get(f"{base_url}/health")).json()
    except httpx.HTTPError as error:
        print(f"cannot reach {base_url}: {error}", file=sys.stderr)
        print("start the server first: cudaforge-serve --echo-runner", file=sys.stderr)
        return 1

    outcome, elapsed = await drive(
        base_url, args.requests, args.concurrency, args.max_new_tokens, args.deadline_seconds
    )
    metrics = await fetch_metrics(base_url)

    payload = {
        "benchmark": "http_server",
        "url": base_url,
        "server": health,
        "load": {
            "requests": args.requests,
            "concurrency": args.concurrency,
            "max_new_tokens": args.max_new_tokens,
            "deadline_seconds": args.deadline_seconds,
        },
        "client_side": {
            "completed": outcome.completed,
            "shed_503": outcome.shed,
            "failed": outcome.failed,
            "wall_seconds": round(elapsed, 4),
            "requests_per_second": round(outcome.completed / elapsed, 2) if elapsed else 0.0,
            "latency_ms": {
                "mean": round(statistics.mean(outcome.latencies) * 1e3, 3)
                if outcome.latencies
                else 0.0,
                "p50": round(percentile(outcome.latencies, 0.50) * 1e3, 3),
                "p95": round(percentile(outcome.latencies, 0.95) * 1e3, 3),
                "p99": round(percentile(outcome.latencies, 0.99) * 1e3, 3),
                "max": round(max(outcome.latencies) * 1e3, 3) if outcome.latencies else 0.0,
            },
        },
        "server_side": metrics,
    }

    if args.json:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    client_side = payload["client_side"]
    latency = client_side["latency_ms"]
    print(f"{base_url}  model={health.get('model')}  device={health.get('device')}")
    print(f"custom CUDA kernels: {health.get('custom_cuda_kernels')}")
    print(f"\n{args.requests} requests at concurrency {args.concurrency}\n")
    print(f"  completed        {client_side['completed']}")
    print(f"  shed (503)       {client_side['shed_503']}")
    print(f"  failed           {client_side['failed']}")
    print(f"  wall time        {client_side['wall_seconds']:.3f} s")
    print(f"  throughput       {client_side['requests_per_second']:.1f} req/s")
    print(
        f"\n  client latency p50/p95/p99   "
        f"{latency['p50']:.2f} / {latency['p95']:.2f} / {latency['p99']:.2f} ms"
    )

    if isinstance(metrics, dict) and "latency_p99_ms" in metrics:
        print(
            f"  server latency p50/p95/p99   "
            f"{metrics['latency_p50_ms']:.2f} / {metrics['latency_p95_ms']:.2f} / "
            f"{metrics['latency_p99_ms']:.2f} ms"
        )
        print(f"  server avg batch size        {metrics['average_batch_size']:.2f}")
        print(
            "\n  The gap between the two p99s is HTTP overhead and client-side\n"
            "  contention, not the runtime."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--deadline-seconds",
        type=float,
        help="ask the server to drop requests still queued past this point",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())

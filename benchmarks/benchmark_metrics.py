#!/usr/bin/env python3
"""Measure the cost of recording a metric on the request path.

    python benchmarks/benchmark_metrics.py

A metrics system that degrades the thing it measures is not useful, and the
Python histogram has a real cost: `bisect.insort` into a sorted list is O(n),
because the insertion memmoves the tail. That cost scales with the window size,
which is why the window has a measured default rather than a generous one.

Two histograms are recorded per request — latency and queue delay — so the
per-request figure is what matters, not the per-record one.

The window is filled before timing so that eviction and insertion are both on
the measured path; an empty histogram measures the cheap case.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass

from cudaforge.metrics import LatencyHistogram, MetricsRegistry


@dataclass
class WindowResult:
    capacity: int
    microseconds_per_record: float
    records_per_second: float


@dataclass
class RegistryResult:
    microseconds_per_request: float
    requests_per_second: float


def time_window(capacity: int, samples: int) -> WindowResult:
    histogram = LatencyHistogram(capacity=capacity)
    # Fill it, so eviction is on the path as well as insertion.
    for i in range(capacity):
        histogram.record((i % 997) / 1000)

    started = time.perf_counter()
    for i in range(samples):
        histogram.record((i % 997) / 1000)
    elapsed = time.perf_counter() - started

    return WindowResult(
        capacity=capacity,
        microseconds_per_record=elapsed / samples * 1e6,
        records_per_second=samples / elapsed,
    )


def time_registry(samples: int) -> RegistryResult:
    registry = MetricsRegistry()
    for i in range(20_000):
        registry.record_completion(i / 100_000, tokens=1)
        registry.record_queue_delay(i / 200_000)

    started = time.perf_counter()
    for i in range(samples):
        registry.record_completion(i / 100_000, tokens=1)
        registry.record_queue_delay(i / 200_000)
    elapsed = time.perf_counter() - started

    return RegistryResult(
        microseconds_per_request=elapsed / samples * 1e6,
        requests_per_second=samples / elapsed,
    )


def time_snapshot(samples: int) -> float:
    """Snapshots read every percentile, so they are far from free."""
    registry = MetricsRegistry()
    for i in range(10_000):
        registry.record_completion(i / 100_000, tokens=1)

    durations = []
    for _ in range(samples):
        started = time.perf_counter()
        registry.snapshot()
        durations.append(time.perf_counter() - started)
    return statistics.median(durations) * 1e6


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--windows", type=int, nargs="+", default=[1_000, 10_000, 100_000])
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    payload = {
        "benchmark": "metrics_overhead",
        "note": (
            "recording cost on the request path; two histograms are recorded "
            "per request, so the registry figure is the one that matters"
        ),
        "samples": args.samples,
        "windows": [asdict(time_window(window, args.samples)) for window in args.windows],
        "registry": asdict(time_registry(args.samples)),
        "snapshot_median_us": time_snapshot(200),
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print(f"wrote {args.output}", file=sys.stderr)
        return 0

    print(payload["note"] + "\n")
    print(f"{'window':>10} {'µs/record':>12} {'records/s':>14}")
    print("-" * 38)
    for window in payload["windows"]:
        print(
            f"{window['capacity']:>10,} {window['microseconds_per_record']:>12.2f} "
            f"{window['records_per_second']:>14,.0f}"
        )

    registry = payload["registry"]
    print(
        f"\nregistry: {registry['microseconds_per_request']:.2f} µs per request "
        f"({registry['requests_per_second']:,.0f} req/s of pure metrics overhead)"
    )
    print(f"snapshot: {payload['snapshot_median_us']:.2f} µs median")
    print(
        "\nThe default window is 10,000. A larger one costs proportionally more "
        "on every\nrequest, which is the thing being measured."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

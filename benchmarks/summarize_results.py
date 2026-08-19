#!/usr/bin/env python3
"""Turn benchmark result JSON into readable Markdown tables.

    ./scripts/benchmark.sh
    python benchmarks/summarize_results.py benchmarks/results/*.json

The harnesses emit JSON because it is what a script should consume. This is
what a person should read, and it exists so that pasting results into an issue
or a report does not mean pasting three hundred lines of JSON.

Dependency-free on purpose: the machine that produced an interesting result is
often not the machine with a plotting stack installed, and requiring one is a
good way to end up with no summary at all.

Speedups are computed only where the harness measured both sides. Nothing is
inferred, and a missing baseline is reported as missing rather than filled in.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def table(headers: list[str], rows: Iterable[list[str]]) -> str:
    rows = list(rows)
    if not rows:
        return "_no rows_\n"

    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def line(cells: list[str]) -> str:
        padded = (cell.ljust(widths[i]) for i, cell in enumerate(cells))
        return "| " + " | ".join(padded) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([line(headers), separator, *(line(row) for row in rows)]) + "\n"


def summarize_cuda_kernels(payload: dict[str, Any]) -> str:
    out = [f"### CUDA kernels — {payload.get('device', 'unknown device')}\n"]
    peak = payload.get("theoretical_bandwidth_gb_s")
    if peak:
        out.append(
            f"Compute capability {payload.get('compute_capability', '?')}, "
            f"{payload.get('multiprocessors', '?')} SMs, "
            f"theoretical bandwidth {peak:.0f} GB/s.\n"
        )

    rows = []
    for result in payload.get("results", []):
        bandwidth = result.get("effective_bandwidth_gb_s")
        share = f"{100 * bandwidth / peak:.0f}%" if bandwidth and peak else "—"
        rows.append(
            [
                result["kernel"],
                result["variant"],
                result["shape"],
                f"{result['median_ms']:.4f}",
                f"{bandwidth:.1f}" if bandwidth else "—",
                share,
            ]
        )

    out.append(
        table(
            ["kernel", "variant", "shape", "median ms", "GB/s", "of peak"],
            rows,
        )
    )
    out.append(
        "\n`of peak` is the number to read. A kernel near the device's "
        "theoretical bandwidth is finished; further work needs an algorithmic "
        "change, not more tuning.\n"
    )
    return "".join(out)


def summarize_batching(payload: dict[str, Any]) -> str:
    out = ["### Dynamic batching\n"]
    if note := payload.get("note"):
        out.append(f"{note}\n\n")

    rows = []
    for case in payload.get("cases", []):
        rows.append(
            [
                str(case["clients"]),
                str(case["max_batch_size"]),
                str(case["max_wait_us"]),
                f"{case['requests_per_second']:.1f}",
                f"{case['average_batch_size']:.2f}",
                f"{100 * case['timeout_closure_fraction']:.0f}%",
                f"{case['latency_p50_ms']:.2f}",
                f"{case['latency_p99_ms']:.2f}",
            ]
        )

    out.append(
        table(
            ["clients", "batch", "wait µs", "req/s", "avg batch", "timeout", "p50 ms", "p99 ms"],
            rows,
        )
    )
    out.append(
        "\nA `timeout` fraction near 100% with small batches means arrivals "
        "never fill a batch, so the configured wait is pure added latency.\n"
    )
    return "".join(out)


def summarize_operators(payload: dict[str, Any]) -> str:
    backend = payload.get("backend", {})
    out = [f"### Operators — {backend.get('description', 'unknown backend')}\n"]
    if caveat := payload.get("caveat"):
        out.append(f"\n> **{caveat}**\n\n")

    # Pair cudaforge against torch for the same kernel and shape. Only where
    # both exist — a missing side is reported, never inferred.
    measurements: dict[tuple[str, str], dict[str, float]] = {}
    for result in payload.get("results", []):
        key = (result["name"], result["shape"])
        measurements.setdefault(key, {})[result["variant"]] = result["median_ms"]

    rows = []
    for (name, shape), variants in measurements.items():
        ours = variants.get("cudaforge")
        theirs = variants.get("torch")
        ratio = f"{theirs / ours:.2f}x" if ours and theirs and ours > 0 else "—"
        rows.append(
            [
                name,
                shape,
                f"{ours:.4f}" if ours else "—",
                f"{theirs:.4f}" if theirs else "—",
                ratio,
            ]
        )

    out.append(table(["operator", "shape", "cudaforge ms", "torch ms", "ratio"], rows))
    if not backend.get("using_custom_kernels", False):
        out.append(
            "\nBoth columns are PyTorch reference implementations. The ratio "
            "measures dispatch overhead, not kernel performance.\n"
        )
    return "".join(out)


def summarize_histogram(payload: dict[str, Any]) -> str:
    bound = payload.get("documented_max_relative_error", 0.0)
    out = [f"### Latency histogram — documented bound {100 * bound:.2f}%\n"]
    rows = [
        [
            distribution["name"],
            f"{100 * distribution['worst_relative_error']:.2f}%",
            distribution["within_documented_bound"],
            f"{distribution['records_per_second'] / 1e6:.1f}M",
        ]
        for distribution in payload.get("distributions", [])
    ]
    out.append(table(["distribution", "worst error", "within bound", "records/s"], rows))
    return "".join(out)


def summarize_generic(payload: dict[str, Any], name: str) -> str:
    out = [f"### {name}\n"]
    cases = payload.get("cases", [])
    if not cases:
        return "".join(out) + "_no cases_\n"

    headers = list(cases[0])
    rows = [
        [f"{case[key]:.4f}" if isinstance(case[key], float) else str(case[key]) for key in headers]
        for case in cases
    ]
    out.append(table(headers, rows))
    return "".join(out)


def summarize_kv_cache(payload: dict[str, Any]) -> str:
    out = [
        f"### KV cache occupancy — {payload.get('cache_tokens', 0):,} token cache, "
        f"max sequence {payload.get('max_sequence_length', 0)}\n"
    ]
    if note := payload.get("note"):
        out.append(f"\n{note}\n\n")

    rows = []
    for workload in payload.get("workloads", []):
        # Block size 16 is the reported point; the others are in the JSON.
        paged = next(
            (entry for entry in workload.get("paged", []) if entry["block_size"] == 16),
            None,
        )
        if paged is None:
            continue
        rows.append(
            [
                workload["workload"],
                f"{workload['mean_length']:.0f}",
                str(workload["contiguous_sequences"]),
                f"{100 * workload['contiguous_waste']:.1f}%",
                str(paged["sequences"]),
                f"{100 * paged['waste']:.1f}%",
                f"{paged['sequences_ratio']:.1f}x",
            ]
        )

    out.append(
        table(
            [
                "workload",
                "mean len",
                "contiguous",
                "waste",
                "paged (16)",
                "waste",
                "ratio",
            ],
            rows,
        )
    )
    out.append(
        "\nA ratio of 1.0x means every sequence reached the permitted maximum, "
        "so there was no waste for paging to recover. The gain is the gap "
        "between the permitted maximum and the actual distribution.\n"
    )

    throughput = payload.get("allocator_throughput", [])
    if throughput:
        out.append(
            "\n"
            + table(
                ["pool blocks", "operations/s"],
                [
                    [str(entry["pool_blocks"]), f"{entry['operations_per_second'] / 1e6:.1f}M"]
                    for entry in throughput
                ],
            )
        )
    return "".join(out)


def summarize_metrics(payload: dict[str, Any]) -> str:
    out = ["### Metrics overhead\n"]
    if note := payload.get("note"):
        out.append(f"\n{note}\n\n")

    out.append(
        table(
            ["window", "µs/record", "records/s"],
            [
                [
                    f"{window['capacity']:,}",
                    f"{window['microseconds_per_record']:.2f}",
                    f"{window['records_per_second']:,.0f}",
                ]
                for window in payload.get("windows", [])
            ],
        )
    )

    registry = payload.get("registry", {})
    if registry:
        out.append(
            f"\nRegistry: **{registry['microseconds_per_request']:.2f} µs per request** "
            f"for both histograms. Snapshot: "
            f"{payload.get('snapshot_median_us', 0.0):.2f} µs median.\n"
        )
    return "".join(out)


def summarize_queue(payload: dict[str, Any]) -> str:
    """Producer/consumer scaling.

    The interesting quantity is not peak throughput but where it stops scaling:
    one mutex serialises every push and pop, so beyond some thread count the
    queue is the bottleneck and adding threads makes things worse.
    """
    out = [f"### Concurrent queue — {payload.get('hardware_threads', '?')} hardware threads\n"]

    by_capacity: dict[int, list[dict[str, Any]]] = {}
    for case in payload.get("cases", []):
        by_capacity.setdefault(case["capacity"], []).append(case)

    for capacity, cases in sorted(by_capacity.items()):
        out.append(f"\n**Capacity {capacity}**\n\n")
        best = max((case["items_per_second"] for case in cases), default=0.0)
        out.append(
            table(
                ["producers", "consumers", "items/s", "of best"],
                [
                    [
                        str(case["producers"]),
                        str(case["consumers"]),
                        f"{case['items_per_second']:,.0f}",
                        f"{100 * case['items_per_second'] / best:.0f}%" if best else "—",
                    ]
                    for case in cases
                ],
            )
        )

    out.append(
        "\nThroughput falling as threads are added means the queue's single "
        "mutex has become the bottleneck. The fix is sharding it, not removing "
        "the lock.\n"
    )
    return "".join(out)


def summarize_scheduler(payload: dict[str, Any]) -> str:
    out = ["### Batch scheduler\n"]
    if note := payload.get("note"):
        out.append(f"\n{note}\n\n")

    out.append(
        table(
            ["producers", "batch", "wait µs", "req/s", "avg batch", "timeout", "p99 ms"],
            [
                [
                    str(case["producers"]),
                    str(case["max_batch_size"]),
                    str(case["max_wait_us"]),
                    f"{case['requests_per_second']:,.0f}",
                    f"{case['average_batch_size']:.2f}",
                    f"{100 * case['timeout_closure_fraction']:.0f}%",
                    f"{case['queue_delay_p99_ms']:.2f}",
                ]
                for case in payload.get("cases", [])
            ],
        )
    )
    return "".join(out)


def summarize_memory(payload: dict[str, Any]) -> str:
    out = [f"### Memory pool — {payload.get('backend', '?')} backend\n"]
    if note := payload.get("note"):
        out.append(f"\n{note}\n\n")

    out.append(
        table(
            ["threads", "pool s", "raw s", "allocations", "backend calls", "reuse"],
            [
                [
                    str(case["threads"]),
                    f"{case['pool_seconds']:.4f}",
                    f"{case['raw_backend_seconds']:.4f}",
                    f"{case['pool_allocations']:,}",
                    str(case["backend_allocations"]),
                    f"{100 * case['reuse_rate']:.2f}%",
                ]
                for case in payload.get("cases", [])
            ],
        )
    )
    out.append(
        "\nThe column that transfers to the device backend is **backend calls**: "
        "each one avoided there is a `cudaMalloc` that would have synchronised "
        "the device. The wall-clock saving on the host backend is small because "
        "malloc already caches.\n"
    )
    return "".join(out)


SUMMARIES = {
    "cuda_kernels": summarize_cuda_kernels,
    "dynamic_batching": summarize_batching,
    "operators": summarize_operators,
    "latency_histogram": summarize_histogram,
    "kv_cache_occupancy": summarize_kv_cache,
    "metrics_overhead": summarize_metrics,
    "concurrent_queue": summarize_queue,
    "dynamic_batcher": summarize_scheduler,
    "memory_pool": summarize_memory,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="result JSON files or a directory")
    args = parser.parse_args(argv)

    targets: list[Path] = []
    for entry in args.paths or ["benchmarks/results"]:
        path = Path(entry)
        if path.is_dir():
            targets.extend(sorted(path.glob("*.json")))
        elif path.is_file():
            targets.append(path)

    if not targets:
        print(
            "no result files found — run ./scripts/benchmark.sh first",
            file=sys.stderr,
        )
        return 1

    print("# Benchmark results\n")
    for target in targets:
        try:
            payload = read(target)
        except json.JSONDecodeError as error:
            print(f"skipping {target}: {error}", file=sys.stderr)
            continue

        name = payload.get("benchmark", target.stem)
        summarize = SUMMARIES.get(name)
        print(summarize(payload) if summarize else summarize_generic(payload, name))
        print(f"<sub>from `{target.name}`</sub>\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

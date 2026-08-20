#!/usr/bin/env python3
"""Serve across every visible GPU, and show the requests actually spread.

    python examples/replicated_serving.py

One engine per device, requests routed to whichever has the shallowest queue.
This is data parallelism: every replica holds the whole model, so throughput
multiplies but the model must still fit on one GPU.

With no GPU, or one, this still runs — the replicas land on CPU and the routing
is unchanged. What it cannot demonstrate without several devices is that the
work went to different *devices*, so the distribution is printed either way and
the device column tells you which case you are looking at.
"""

from __future__ import annotations

import argparse
import time

from cudaforge.config import EngineConfig, GenerationConfig
from cudaforge.engine import InferenceEngine
from cudaforge.replicated import ReplicatedEngine
from cudaforge.runners import EchoRunner


def visible_devices() -> list[str]:
    try:
        import torch
    except ImportError:
        return ["cpu"]
    if not torch.cuda.is_available():
        return ["cpu"]
    return [f"cuda:{index}" for index in range(torch.cuda.device_count())]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-batch-size", type=int, default=8)
    args = parser.parse_args()

    devices = visible_devices()
    print(f"replicas: {len(devices)} on {', '.join(devices)}")

    def build(device: str) -> InferenceEngine:
        return InferenceEngine(
            config=EngineConfig(
                device=device,
                max_batch_size=args.max_batch_size,
                warmup_iterations=0,
                generation=GenerationConfig(max_new_tokens=args.max_new_tokens),
            ),
            # The deterministic runner: this example is about routing and
            # lifecycle, and a real model would measure the model instead.
            runner=EchoRunner(per_token_seconds=0.001),
        )

    engine = ReplicatedEngine.across_devices(devices, build)
    with engine:
        started = time.perf_counter()
        responses = engine.generate_many([f"prompt {i}" for i in range(args.requests)])
        elapsed = time.perf_counter() - started
        counts = engine.routed_counts()

    ok = sum(1 for response in responses if response.ok)
    print(f"completed {ok}/{len(responses)} in {elapsed:.2f}s")
    print("\nrequests per replica:")
    for device, count in zip(devices, counts, strict=True):
        share = count / max(sum(counts), 1)
        bar = "#" * int(share * 40)
        print(f"  {device:<10} {count:>4}  {bar}")

    if len(devices) == 1:
        print("\nOne device, so this shows routing but not parallelism.")
    else:
        spread = min(counts) / max(max(counts), 1)
        print(f"\nBalance: the least-loaded replica took {spread:.0%} of the busiest one's share.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

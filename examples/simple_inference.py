#!/usr/bin/env python3
"""Smallest useful example: one prompt through the engine.

    python examples/simple_inference.py --echo-runner

Start here. It confirms the package imports, reports which implementation path
is active, and shows the timing breakdown a response carries.
"""

from __future__ import annotations

import argparse
import logging

from cudaforge.config import EngineConfig, GenerationConfig
from cudaforge.engine import InferenceEngine
from cudaforge.ops import backend_report
from cudaforge.runners import EchoRunner, ModelRunner, TransformersRunner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="Explain what a CUDA warp is.")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--echo-runner",
        action="store_true",
        help="skip model loading and use the deterministic runner",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Worth printing before anything else: a silent fallback to the reference
    # path looks exactly like a very slow custom kernel.
    print(backend_report())

    config = EngineConfig(model_name=args.model, max_batch_size=4, max_wait_us=2_000)
    runner: ModelRunner = EchoRunner() if args.echo_runner else TransformersRunner(config)

    with InferenceEngine(config=config, runner=runner) as engine:
        response = engine.generate(
            args.prompt, GenerationConfig(max_new_tokens=args.max_new_tokens)
        )

    print(f"\nprompt : {args.prompt}")
    print(f"output : {response.text}")
    print(f"\nrequest id       {response.request_id}")
    print(f"queue time       {response.queue_time * 1e3:.2f} ms")
    print(f"inference time   {response.inference_time * 1e3:.2f} ms")
    print(f"total latency    {response.total_latency * 1e3:.2f} ms")
    print(f"batch size       {response.batch_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""End-to-end LoRA fine-tune on a tiny model.

    python examples/fine_tune.py

Runs on CPU in under a minute with no dataset download. The point is to show
the pipeline working and to make the parameter-efficiency argument concrete:
the printed trainable-parameter share is the number that explains why LoRA fits
where full fine-tuning does not.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from training.config import LoRAConfig, TrainingConfig
from training.train import train

CORPUS = [
    "CUDA threads are grouped into warps of 32 that execute the same instruction.",
    "Shared memory is on-chip and far faster to access than global memory.",
    "Coalesced access lets the hardware merge adjacent requests into one transaction.",
    "A condition variable parks a thread until another signals a state change.",
    "Dynamic batching amortises weight loading across concurrent requests.",
    "CUDA streams let copies and kernels overlap on separate hardware engines.",
    "Pinned host memory is required for a transfer to overlap with computation.",
    "LoRA freezes the base weights and trains a low-rank update alongside them.",
    "Quantisation trades numerical precision for memory and bandwidth.",
    "Occupancy measures how many warps are resident on a streaming multiprocessor.",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--output-dir", default="checkpoints/example")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rank", type=int, default=4)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    config = TrainingConfig(
        model_name=args.model,
        output_dir=args.output_dir,
        inline_texts=CORPUS,
        max_seq_length=64,
        batch_size=2,
        learning_rate=1e-3,
        epochs=args.epochs,
        mixed_precision=False,
        logging_steps=1,
        lora=LoRAConfig(rank=args.rank, alpha=args.rank * 2, target_modules=["c_attn"]),
    )

    state = train(config)

    print(f"\ncompleted {state.step} optimiser steps over {state.tokens_seen} tokens")
    print(f"checkpoint: {Path(args.output_dir) / 'final'}")
    print(
        "\nOnly the adapter tensors were written. The frozen base weights are "
        "not duplicated,\nwhich is why a LoRA checkpoint is megabytes rather "
        "than gigabytes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

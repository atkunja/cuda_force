"""Continuous versus static batching, driving a real transformer.

`benchmark_continuous.py` measures the same comparison against
`EchoStepwiseRunner`, which sleeps instead of computing. That isolates the
scheduler, but it means the per-step cost is a number chosen by the benchmark.
Here the step cost is whatever a forward pass actually costs: real attention
over a real KV cache, with the cache growing and being resized as sequences join
and leave.

By default the model is constructed locally with random weights and a synthetic
vocabulary, so the benchmark downloads nothing and runs anywhere. The generated
*text* is meaningless — but nothing measured here depends on it. What is
measured is decode steps, row occupancy and wall-clock, all of which come from
real forward passes. Pass `--model` to run against pretrained weights.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from cudaforge.config import EngineConfig, GenerationConfig
from cudaforge.continuous import ContinuousBatcher, ContinuousStats, run_static
from cudaforge.scheduler import Request
from cudaforge.stepwise import SequenceState
from cudaforge.stepwise_transformers import TransformersStepwiseRunner


class SyntheticTokenizer:
    """A whitespace tokenizer over a synthetic vocabulary.

    Present so the benchmark needs no download. Token *identity* is irrelevant
    to every quantity reported here; token *count* is not, and this preserves it.
    """

    pad_token_id = 0
    eos_token_id = None
    eos_token = "<eos>"
    padding_side = "right"

    def __init__(self, vocab_size: int = 64) -> None:
        self.vocab_size = vocab_size

    def __call__(
        self,
        prompts: list[str],
        return_tensors: str | None = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> dict[str, torch.Tensor]:
        encoded = [[len(word) % (self.vocab_size - 4) + 1 for word in p.split()] for p in prompts]
        width = max(len(ids) for ids in encoded)
        input_ids, attention_mask = [], []
        for ids in encoded:
            pad = width - len(ids)
            input_ids.append([self.pad_token_id] * pad + ids)
            attention_mask.append([0] * pad + [1] * len(ids))

        class Batch(dict):
            def to(self, _device: object) -> Batch:
                return self

        return Batch(
            input_ids=torch.tensor(input_ids),
            attention_mask=torch.tensor(attention_mask),
        )

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return f" t{int(ids[0])}"


class Timed:
    """Wraps a runner and separates prefill time from decode time.

    The two move in opposite directions between the policies, so a single
    wall-clock number hides the actual trade. Kept in the benchmark rather than
    the runner: production code should not pay for a clock read per token.
    """

    def __init__(self, inner: TransformersStepwiseRunner) -> None:
        self.inner = inner
        self.prefill_seconds = 0.0
        self.decode_seconds = 0.0

    def prefill(self, states: list[SequenceState]) -> None:
        start = time.perf_counter()
        self.inner.prefill(states)
        self.prefill_seconds += time.perf_counter() - start

    def decode_step(self, states: list[SequenceState]) -> None:
        start = time.perf_counter()
        self.inner.decode_step(states)
        self.decode_seconds += time.perf_counter() - start

    def evict(self, sequence_id: int) -> None:
        self.inner.evict(sequence_id)

    @property
    def description(self) -> str:
        return self.inner.description


def build_runner(args: argparse.Namespace) -> TransformersStepwiseRunner:
    config = EngineConfig(device=args.device, dtype="float32", model_name=args.model or "synthetic")
    if args.model:
        return TransformersStepwiseRunner(config)

    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(
            n_layer=args.layers,
            n_head=args.heads,
            n_embd=args.width,
            vocab_size=64,
            n_positions=1024,
        )
    )
    return TransformersStepwiseRunner(config, model=model, tokenizer=SyntheticTokenizer())


def workload(count: int, seed: int) -> list[tuple[str, GenerationConfig]]:
    """A skewed length mix, which is where the two policies diverge.

    Static batching holds a batch until its longest member finishes, so the cost
    of a skewed mix is paid by every short sequence sharing a batch with a long
    one. An even mix would understate the difference; real traffic is skewed.
    """
    generator = torch.Generator().manual_seed(seed)
    lengths = []
    for _ in range(count):
        # One in eight requests is an order of magnitude longer than the rest.
        long = torch.rand(1, generator=generator).item() < 0.125
        span = (60, 100) if long else (4, 12)
        lengths.append(int(torch.randint(span[0], span[1], (1,), generator=generator).item()))

    return [
        (f"prompt number {index} asks a question", GenerationConfig(max_new_tokens=length))
        for index, length in enumerate(lengths)
    ]


def measure_static(
    args: argparse.Namespace, prompts: list[tuple[str, GenerationConfig]]
) -> tuple[ContinuousStats, float, Timed]:
    runner = Timed(build_runner(args))
    start = time.perf_counter()
    stats = run_static(runner, prompts, max_batch_size=args.batch)
    return stats, time.perf_counter() - start, runner


def measure_continuous(
    args: argparse.Namespace, prompts: list[tuple[str, GenerationConfig]]
) -> tuple[ContinuousStats, float, Timed]:
    runner = Timed(build_runner(args))
    done: list[float] = []
    start = time.perf_counter()

    batcher = ContinuousBatcher(
        runner,
        on_complete=lambda *_: done.append(time.perf_counter() - start),
        max_batch_size=args.batch,
        queue_capacity=max(len(prompts), args.batch),
    )
    with batcher:
        for prompt, generation in prompts:
            batcher.submit(Request(prompt=prompt, generation=generation))
    elapsed = time.perf_counter() - start
    return batcher.stats(), elapsed, runner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default=None, help="pretrained model name; default is synthetic")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    prompts = workload(args.requests, args.seed)
    lengths = [generation.max_new_tokens for _, generation in prompts]

    print(f"requests {len(prompts)}  batch {args.batch}  device {args.device}")
    print(
        f"generation lengths: min {min(lengths)}  median "
        f"{int(statistics.median(lengths))}  max {max(lengths)}"
    )
    described = args.model or f"synthetic gpt2 ({args.layers}L/{args.heads}H/{args.width}d)"
    print(f"model: {described}\n")

    static, static_seconds, static_timed = measure_static(args, prompts)
    continuous, continuous_seconds, continuous_timed = measure_continuous(args, prompts)

    row = "{:<12}{:>8}{:>9}{:>11}{:>11}{:>11}{:>10}"
    print(
        row.format("policy", "steps", "prefills", "occupancy", "prefill s", "decode s", "total s")
    )
    for name, stats, seconds, timed in (
        ("static", static, static_seconds, static_timed),
        ("continuous", continuous, continuous_seconds, continuous_timed),
    ):
        print(
            row.format(
                name,
                stats.decode_steps,
                timed.inner.prefills,
                f"{stats.utilisation:.1%}",
                f"{timed.prefill_seconds:.3f}",
                f"{timed.decode_seconds:.3f}",
                f"{seconds:.3f}",
            )
        )

    saved = 1 - continuous.decode_steps / static.decode_steps if static.decode_steps else 0.0
    decode_saved = (
        1 - continuous_timed.decode_seconds / static_timed.decode_seconds
        if static_timed.decode_seconds
        else 0.0
    )
    print(f"\ncontinuous uses {saved:.1%} fewer decode steps, {decode_saved:.1%} less decode time")
    print(
        f"but {continuous_timed.inner.prefills} prefill calls against "
        f"{static_timed.inner.prefills}: rows free up in ones and twos, so admissions "
        f"trickle in rather than arriving as full batches."
    )
    print(f"overall {static_seconds / continuous_seconds:.2f}x" if continuous_seconds else "")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "requests": len(prompts),
                    "max_batch_size": args.batch,
                    "model": args.model or "synthetic-gpt2",
                    "static": {
                        "decode_steps": static.decode_steps,
                        "utilisation": static.utilisation,
                        "prefill_calls": static_timed.inner.prefills,
                        "prefill_seconds": static_timed.prefill_seconds,
                        "decode_seconds": static_timed.decode_seconds,
                        "seconds": static_seconds,
                    },
                    "continuous": {
                        "decode_steps": continuous.decode_steps,
                        "utilisation": continuous.utilisation,
                        "prefill_calls": continuous_timed.inner.prefills,
                        "prefill_seconds": continuous_timed.prefill_seconds,
                        "decode_seconds": continuous_timed.decode_seconds,
                        "seconds": continuous_seconds,
                    },
                },
                indent=2,
            )
            + "\n"
        )
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

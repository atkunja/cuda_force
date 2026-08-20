#!/usr/bin/env python3
"""Show that speculative decoding changes speed without changing output.

    python examples/speculative_decoding.py

The claim worth demonstrating is not that draft-and-verify is fast — it is that
it is *lossless*. So this decodes the same prompt twice: once the ordinary way,
one token per forward pass, and once with a draft model proposing ahead. The
two token sequences must be identical, and the second must use fewer target
calls to get there.

Both models are built locally with random weights, so nothing is downloaded and
this runs on any machine. The generated text is meaningless — nothing shown here
depends on it. What is shown is the target-call count and the token sequence,
and both are real.

Pass `--model`/`--draft` to run against pretrained weights instead. They must
share a tokenizer: the draft's token ids are handed to the target directly, so a
mismatched vocabulary would compare unrelated tokens rather than fail loudly.
"""

from __future__ import annotations

import argparse
import sys

import torch

from cudaforge.config import GenerationConfig
from cudaforge.speculative import SpeculativeDecoder, expected_tokens_per_call


def build(layers: int, width: int, heads: int, vocab: int, seed: int) -> object:
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    return GPT2LMHeadModel(
        GPT2Config(n_layer=layers, n_head=heads, n_embd=width, vocab_size=vocab, n_positions=512)
    ).eval()


def decode_one_at_a_time(
    target: object, prompt: torch.Tensor, tokens: int
) -> tuple[list[int], int]:
    """Ordinary greedy decoding: one target call per token."""
    produced: list[int] = []
    step, cache = prompt, None
    with torch.inference_mode():
        for _ in range(tokens):
            output = target(input_ids=step, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            token = int(output.logits[0, -1, :].argmax())
            produced.append(token)
            step = torch.tensor([[token]])
    return produced, tokens


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument("--model", default=None, help="pretrained target; default is synthetic")
    parser.add_argument("--draft", default=None, help="pretrained draft; required with --model")
    args = parser.parse_args()

    if bool(args.model) != bool(args.draft):
        print("--model and --draft must be given together", file=sys.stderr)
        return 2

    try:
        import transformers  # noqa: F401
    except ImportError:
        print("transformers is not installed; install the inference extra to run this example")
        return 0

    vocab = 64
    if args.model:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        target = AutoModelForCausalLM.from_pretrained(args.model).eval()
        draft = AutoModelForCausalLM.from_pretrained(args.draft).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        prompt = tokenizer("The capital of France is", return_tensors="pt")["input_ids"]
        described = f"{args.model} drafted by {args.draft}"
    else:
        target = build(4, 64, 4, vocab, seed=0)
        draft = build(1, 32, 2, vocab, seed=1)
        tokenizer = None
        prompt = torch.randint(1, vocab - 4, (1, 7), generator=torch.Generator().manual_seed(41))
        described = "synthetic gpt2 (4L/64d) drafted by (1L/32d), random weights"

    print(f"model: {described}")
    print(f"prompt: {prompt.shape[1]} tokens, generating {args.tokens}\n")

    baseline, baseline_calls = decode_one_at_a_time(target, prompt, args.tokens)
    speculative, stats = SpeculativeDecoder(target, draft, lookahead=args.lookahead).generate(
        prompt, GenerationConfig(max_new_tokens=args.tokens, temperature=0.0)
    )

    row = "{:<26}{:>14}{:>18}"
    print(row.format("", "target calls", "tokens per call"))
    print(row.format("one token at a time", baseline_calls, "1.00"))
    print(
        row.format(
            f"speculative (k={args.lookahead})",
            stats.target_calls,
            f"{stats.tokens_per_target_call:.2f}",
        )
    )

    identical = baseline == speculative
    print(f"\noutput identical: {identical}")
    if not identical:
        # A mismatch means the verification rule is broken, which matters far
        # more than any speedup. Fail loudly rather than reporting the timing.
        for index, (left, right) in enumerate(zip(baseline, speculative, strict=False)):
            if left != right:
                print(f"first divergence at token {index}: {left} vs {right}", file=sys.stderr)
                break
        return 1

    if tokenizer is not None:
        print(f"text: {tokenizer.decode(speculative, skip_special_tokens=True)!r}")

    saved = 1 - stats.target_calls / baseline_calls
    print(f"target calls saved: {saved:.1%}")
    print(
        f"draft calls spent: {stats.draft_calls} "
        f"({stats.draft_calls / max(stats.target_calls, 1):.1f} per target call)"
    )
    print(
        f"\nAt this acceptance the closed form predicts "
        f"{expected_tokens_per_call(stats.accepted / max(stats.proposed, 1), args.lookahead):.2f} "
        f"tokens per target call."
    )
    print(
        "Target calls saved is a ceiling, not a speedup: the draft calls above are\n"
        "real work. The gain is only realised when the draft is much cheaper than\n"
        "the target, which these equally-tiny models are not."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

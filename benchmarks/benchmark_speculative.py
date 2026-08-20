"""What speculative decoding buys, as a function of how good the draft is.

    python benchmarks/benchmark_speculative.py

The quantity that matters is **tokens per target call**. Target calls dominate
the cost — the draft is chosen to be cheap — so that ratio is the speedup
ceiling, and it is 1.0 without speculation.

## Why the acceptance rate is an input here, not a measurement

The obvious benchmark is "run a real draft against a real target and report the
acceptance rate". That cannot be done honestly on this machine. Randomly
initialised models agree with each other at a rate that reflects their shared
degeneracy rather than any real draft/target relationship, and pretrained
weights would need a download and a GPU to be worth timing.

So acceptance is *imposed*: a synthetic draft agrees with the target with a
chosen probability, independently per token. That turns the benchmark into
something more useful than one number — a curve over the parameter a real
deployment would actually have to measure — and it can be checked against theory.

For i.i.d. acceptance probability `a` and lookahead `k`, the number of accepted
proposals is a geometric variable capped at `k`, and every block emits one extra
token (the target's own on rejection, or the free trailing token on a full
accept). So

    E[tokens per target call] = 1 + sum(a^i for i in 1..k) = (1 - a^(k+1)) / (1 - a)

The table prints measurement against that closed form. Agreement is evidence the
implementation is right; it is not itself a performance claim.

## What this does not measure

Wall-clock. That depends on the draft-to-target cost ratio, which depends on the
two models and the hardware. The arithmetic to combine them is in the notes at
the end of the output, but no timing here comes from a GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from cudaforge.config import GenerationConfig
from cudaforge.speculative import SpeculativeDecoder, expected_tokens_per_call


class ListCache:
    """The whole cache interface `SpeculativeDecoder` requires: `crop`.

    The decoder only ever rolls a cache back, and by a count rather than to an
    absolute length. Writing that out keeps this benchmark independent of
    transformers and pins the contract — if the decoder starts calling something
    else, this fails loudly instead of silently binding to a library type.
    """

    def __init__(self) -> None:
        self.tokens: list[int] = []

    def extend(self, ids: list[int]) -> None:
        self.tokens.extend(ids)

    def crop(self, amount: int) -> None:
        # Negative-only, matching the non-deprecated transformers signature.
        # `crop(0)` would mean "keep nothing", so the decoder must never send it.
        assert amount < 0, f"crop expects a negative count, got {amount}"
        del self.tokens[amount:]


class ChainModel:
    """A causal model whose next token depends on its whole history.

    Deterministic and context-sensitive, so a mishandled KV cache changes the
    answer instead of being absorbed by a degenerate argmax.
    """

    def __init__(self, vocab: int = 64) -> None:
        self.vocab = vocab

    def __call__(
        self,
        input_ids: torch.Tensor,
        past_key_values: ListCache | None = None,
        use_cache: bool = True,
    ) -> SimpleNamespace:
        cache = past_key_values if past_key_values is not None else ListCache()
        added = input_ids[0].tolist()
        before = list(cache.tokens)
        cache.extend(added)

        rows = []
        for offset in range(len(added)):
            prefix = before + added[: offset + 1]
            value = int(sum(prefix) * 5 + len(prefix) * 13) % self.vocab
            row = torch.full((self.vocab,), -8.0)
            row[value] = 8.0
            rows.append(row)

        return SimpleNamespace(logits=torch.stack(rows).unsqueeze(0), past_key_values=cache)


class ImperfectDraft(ChainModel):
    """Agrees with the target with probability `rate`, independently per token."""

    def __init__(self, rate: float, seed: int, vocab: int = 64) -> None:
        super().__init__(vocab)
        self.rate = rate
        self._rng = torch.Generator().manual_seed(seed)

    def __call__(
        self,
        input_ids: torch.Tensor,
        past_key_values: ListCache | None = None,
        use_cache: bool = True,
    ) -> SimpleNamespace:
        output = super().__call__(input_ids, past_key_values, use_cache)
        logits = output.logits
        for position in range(logits.shape[1]):
            if float(torch.rand(1, generator=self._rng)) >= self.rate:
                # Move the peak one token over, so this proposal is rejected.
                best = int(logits[0, position].argmax())
                logits[0, position, best] = -8.0
                logits[0, position, (best + 1) % self.vocab] = 8.0
        return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rates = [0.3, 0.5, 0.7, 0.9]
    lookaheads = [1, 2, 4, 8]
    prompt = torch.tensor([[3, 1, 4, 1, 5]])
    settings = GenerationConfig(max_new_tokens=args.tokens, temperature=0.0)

    print(f"{args.tokens} tokens per run, acceptance imposed rather than measured\n")
    header = "{:>10}{:>4}{:>12}{:>14}{:>12}"
    print(header.format("imposed a", "k", "measured", "closed form", "accept/prop"))

    records = []
    for rate in rates:
        for lookahead in lookaheads:
            _, stats = SpeculativeDecoder(
                ChainModel(), ImperfectDraft(rate, args.seed), lookahead=lookahead
            ).generate(prompt, settings)
            predicted = expected_tokens_per_call(rate, lookahead)
            print(
                header.format(
                    f"{rate:.1f}",
                    lookahead,
                    f"{stats.tokens_per_target_call:.3f}",
                    f"{predicted:.3f}",
                    f"{stats.acceptance_rate:.3f}",
                )
            )
            records.append(
                {
                    "acceptance": rate,
                    "lookahead": lookahead,
                    "tokens_per_target_call": stats.tokens_per_target_call,
                    "closed_form": predicted,
                    "observed_acceptance": stats.acceptance_rate,
                    "target_calls": stats.target_calls,
                    "draft_calls": stats.draft_calls,
                }
            )

    print(
        "\nTokens per target call is the speedup ceiling, not the speedup. Real gain is\n"
        "  ceiling / (1 + k * draft_cost / target_cost)\n"
        "so a draft costing a tenth of the target at k=4 keeps about 5/7 of the ceiling.\n"
        "Both cost terms are hardware-dependent and are not measured here."
    )
    print(
        "Note also that raising k past the point where a^k is small buys little: at\n"
        "acceptance 0.5 the ceiling is already 1.94 by k=4 and cannot exceed 2.0."
    )
    print(
        "The last column is accepted/proposed, which is not the imposed rate: a block\n"
        "stops at its first rejection while the draft has already produced the whole\n"
        "lookahead, so unused proposals count against it. It falls with k by\n"
        "construction and measures wasted draft work, not draft quality."
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"runs": records}, indent=2) + "\n")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

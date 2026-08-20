"""Speculative decoding: a cheap model proposes, an expensive one verifies.

Autoregressive decoding is latency-bound rather than compute-bound. Generating
one token needs one pass over the whole model, and at batch size 1 that pass
moves far more weight through memory than it does arithmetic — the hardware is
idle waiting on bandwidth. Running the same pass over *several* candidate tokens
costs almost the same wall-clock as running it over one.

Speculative decoding spends that headroom. A small draft model proposes `k`
tokens, and the target model checks all `k` in a single pass. Every proposal the
target agrees with is a token generated without a target call of its own.

## Why the output is not an approximation

The obvious worry is that this trades quality for speed. It does not. The
acceptance rule is built so the accepted tokens are distributed exactly as if
they had been sampled from the target alone:

* **Greedy** (`temperature == 0`): a proposal is kept only when it equals the
  target's own argmax. Identical by construction — `test_speculative_greedy_
  matches_the_target_alone` asserts token-for-token equality.
* **Sampling**: a proposal `x` drawn from the draft's `q` is kept with
  probability `min(1, p(x)/q(x))`. On rejection the replacement is drawn from
  the normalised residual `max(0, p - q)`. Those two cases compose to exactly
  `p` — the standard rejection-sampling identity, which is what makes the
  method lossless rather than a heuristic.

The draft model's quality therefore affects *speed only*. A bad draft is
rejected more often and saves less; it cannot corrupt the output.

## Progress is guaranteed

On rejection at position `i`, the target's own token for that position is
emitted. So every target call yields at least one token, and the loop cannot
stall no matter how badly the draft performs. When all `k` proposals are
accepted the pass also yields a free "bonus" token from the trailing logits,
giving `k + 1` tokens from one target call.

## Scope

Batch size 1. Batched speculation needs per-row acceptance lengths, which makes
the KV cache ragged in a way the contiguous cache here cannot express — the same
limitation described in `stepwise_transformers`. Stated rather than worked
around.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from cudaforge.config import GenerationConfig


@dataclass
class SpeculativeStats:
    """What the speculation actually bought.

    `acceptance_rate` is the quality signal — it measures how well the draft
    tracks the target. `tokens_per_target_call` is the speed signal, and is the
    one that translates into wall-clock, since target calls dominate the cost.
    """

    target_calls: int = 0
    draft_calls: int = 0
    proposed: int = 0
    accepted: int = 0
    generated: int = 0

    @property
    def acceptance_rate(self) -> float:
        """Accepted proposals over proposals *made*.

        Not the per-token probability that the draft agrees with the target. A
        block stops at its first rejection, but the draft has already generated
        the whole lookahead, and those unused proposals still count here — they
        cost draft time. So this figure falls as `lookahead` rises even when the
        draft is exactly as good: at a true 0.9 agreement it reads 0.92 at k=1
        and 0.62 at k=8.

        That makes it the right number for "what fraction of draft work was
        wasted" and the wrong one for "how well does the draft track the
        target". Use `tokens_per_target_call` to compare lookaheads.
        """
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tokens_per_target_call(self) -> float:
        """The speedup ceiling: without speculation this is exactly 1.0."""
        return self.generated / self.target_calls if self.target_calls else 0.0


def _distribution(logits: torch.Tensor, generation: GenerationConfig) -> torch.Tensor:
    """Turn one row of logits into the sampling distribution.

    Filtering must happen *before* the acceptance test rather than after, because
    the rule compares the two models' probabilities. Comparing an unfiltered
    draft probability against a filtered target one is not the same distribution
    the tokens would have come from, and the losslessness argument no longer
    holds.
    """
    scaled = logits / max(generation.temperature, 1e-6)

    if generation.top_k > 0:
        limit = min(generation.top_k, scaled.shape[-1])
        threshold = scaled.topk(limit, dim=-1).values[..., -1:]
        scaled = scaled.masked_fill(scaled < threshold, float("-inf"))

    probabilities = scaled.softmax(dim=-1)

    if generation.top_p < 1.0:
        ordered, order = probabilities.sort(dim=-1, descending=True)
        cumulative = ordered.cumsum(dim=-1)
        drop = cumulative - ordered > generation.top_p
        drop[..., 0] = False
        ordered = ordered.masked_fill(drop, 0.0)
        probabilities = ordered.scatter(-1, order, ordered)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)

    return probabilities


def expected_tokens_per_call(acceptance: float, lookahead: int) -> float:
    """Tokens per target call predicted for i.i.d. acceptance probability.

    The accepted count is geometric capped at `lookahead`, and every block emits
    one further token — the target's own on rejection, or the free trailing one
    on a full accept. So the expectation is `1 + sum(a**i for i in 1..k)`, which
    telescopes to the closed form below.

    Useful for choosing a lookahead without running anything: returns saturate
    once `a**k` is small, so raising `k` past that point buys nothing but wasted
    draft work. At `a = 0.5` the value is already 1.94 by `k = 4` against a
    ceiling of 2.0.

    This is an upper bound on realisable speedup, not the speedup: it ignores
    what the draft costs. With a draft costing `c` times the target, the gain is
    roughly `expected_tokens_per_call(a, k) / (1 + k * c)`.
    """
    if not 0.0 <= acceptance <= 1.0:
        raise ValueError(f"acceptance must be a probability, got {acceptance}")
    if lookahead < 1:
        raise ValueError(f"lookahead must be at least 1, got {lookahead}")
    if acceptance >= 1.0:
        return float(lookahead + 1)
    return (1.0 - acceptance ** (lookahead + 1)) / (1.0 - acceptance)


class SpeculativeDecoder:
    """Runs a target model with a draft model proposing ahead of it.

    Both models must share a tokenizer — the draft's token ids are fed to the
    target directly, so a mismatched vocabulary would silently compare unrelated
    tokens rather than fail.
    """

    def __init__(self, target: Any, draft: Any, lookahead: int = 4) -> None:
        if lookahead < 1:
            raise ValueError(f"lookahead must be at least 1, got {lookahead}")
        self._target = target
        self._draft = draft
        self._lookahead = lookahead

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        generation: GenerationConfig,
        seed: int | None = None,
    ) -> tuple[list[int], SpeculativeStats]:
        """Generate from a single prompt of shape `(1, prompt_length)`."""
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"expected a single prompt of shape (1, n), got {tuple(input_ids.shape)}"
            )

        rng = None
        if seed is not None:
            rng = torch.Generator(device=input_ids.device).manual_seed(seed)

        stats = SpeculativeStats()
        greedy = generation.temperature <= 0.0
        budget = generation.max_new_tokens

        # Every token settled so far, prompt included. Both models consume from
        # this list at their own pace: they ingest different amounts per block,
        # so tracking one shared "pending" tensor silently desynchronises them.
        context: list[int] = input_ids[0].tolist()
        prompt_length = len(context)
        target_ingested = 0
        draft_ingested = 0

        target_cache: Any = None
        draft_cache: Any = None

        while len(context) - prompt_length < budget:
            base = len(context)
            remaining = budget - (base - prompt_length)
            span = min(self._lookahead, remaining)

            proposals, draft_probabilities, draft_cache = self._propose(
                self._tensor(context[draft_ingested:], input_ids),
                draft_cache,
                span,
                generation,
                greedy,
                rng,
                stats,
            )
            # The draft ingests its pending tokens plus every proposal except the
            # last, which it generates but never feeds back.
            draft_covered = base + span - 1

            # One target pass covers the pending tokens plus every proposal,
            # yielding span + 1 predictions. This is the whole trick: checking
            # `span` tokens costs about what generating one costs.
            block = torch.cat(
                [self._tensor(context[target_ingested:], input_ids), proposals], dim=1
            )
            output = self._target(input_ids=block, past_key_values=target_cache, use_cache=True)
            target_cache = output.past_key_values
            stats.target_calls += 1

            # The last span + 1 logits align with the proposals; position -1 is
            # the prediction after the final proposal, hence the bonus token.
            verdict = output.logits[0, -(span + 1) :, :]
            taken, replacement = self._verify(
                proposals[0].tolist(), verdict, draft_probabilities, generation, greedy, rng
            )
            stats.proposed += span
            stats.accepted += taken

            # A block yields taken + 1 tokens, which can overshoot the budget.
            # Clipping only ever bites on the final block — producing `remaining`
            # tokens ends the loop — so the caches below are left describing
            # slightly more history than `context` holds, and nothing reads them
            # again.
            settled = [*proposals[0, :taken].tolist(), replacement][:remaining]
            context.extend(settled)
            stats.generated += len(settled)

            # Roll both caches back to what was accepted. The replacement token
            # is deliberately left out: no model has seen it, so it becomes the
            # next block's input rather than history.
            settled_position = base + taken
            self._crop(target_cache, base + span, settled_position)
            target_ingested = settled_position
            if draft_cache is not None:
                draft_ingested = min(draft_covered, settled_position)
                self._crop(draft_cache, draft_covered, draft_ingested)

        generated = context[prompt_length:]
        return generated, stats

    @staticmethod
    def _tensor(tokens: list[int], like: torch.Tensor) -> torch.Tensor:
        return torch.tensor([tokens], device=like.device, dtype=like.dtype)

    @staticmethod
    def _crop(cache: Any, current: int, wanted: int) -> None:
        """Drop `current - wanted` tokens from the end of a cache.

        Expressed as a count rather than an absolute length: the absolute form is
        deprecated, and `crop(0)` means "keep nothing" rather than "keep
        everything", so the no-op case must be skipped explicitly.
        """
        excess = current - wanted
        if excess > 0:
            cache.crop(-excess)

    def _propose(
        self,
        pending: torch.Tensor,
        cache: Any,
        span: int,
        generation: GenerationConfig,
        greedy: bool,
        rng: torch.Generator | None,
        stats: SpeculativeStats,
    ) -> tuple[torch.Tensor, list[torch.Tensor], Any]:
        """Run the draft model `span` times, one token per call."""
        proposals: list[int] = []
        distributions: list[torch.Tensor] = []
        step_input = pending

        for _ in range(span):
            output = self._draft(input_ids=step_input, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            stats.draft_calls += 1

            logits = output.logits[0, -1, :]
            if greedy:
                # The distribution is still recorded: rejection needs the
                # residual, and for greedy that is a point mass.
                token = int(logits.argmax())
                probabilities = torch.zeros_like(logits)
                probabilities[token] = 1.0
            else:
                probabilities = _distribution(logits, generation)
                token = int(torch.multinomial(probabilities, 1, generator=rng))

            proposals.append(token)
            distributions.append(probabilities)
            step_input = torch.tensor([[token]], device=pending.device)

        return torch.tensor([proposals], device=pending.device), distributions, cache

    def _verify(
        self,
        proposals: list[int],
        verdict: torch.Tensor,
        draft_probabilities: list[torch.Tensor],
        generation: GenerationConfig,
        greedy: bool,
        rng: torch.Generator | None,
    ) -> tuple[int, int]:
        """Return how many proposals survive, and the token that follows them."""
        for index, proposed in enumerate(proposals):
            row = verdict[index]

            if greedy:
                if int(row.argmax()) == proposed:
                    continue
                # Rejected: emit the target's own choice, so the call still
                # advances by a token and the result matches greedy decoding.
                return index, int(row.argmax())

            target = _distribution(row, generation)
            draft = draft_probabilities[index]
            ratio = (target[proposed] / draft[proposed]).clamp(max=1.0)
            if float(torch.rand(1, generator=rng, device=row.device)) < float(ratio):
                continue

            # The residual is what the target wanted but the draft under-proposed.
            # Sampling it here is what keeps the composite distribution exactly p.
            residual = (target - draft).clamp(min=0.0)
            # Both distributions are normalised, so the residual sums to zero
            # only when p equals q — and then the acceptance ratio is 1 and this
            # branch is never reached. Asserted rather than guarded: a guard here
            # would be a line no test can reach.
            total = float(residual.sum())
            assert total > 0.0, "a rejected proposal implies a non-empty residual"
            return index, int(torch.multinomial(residual / total, 1, generator=rng))

        # Every proposal held, so the trailing logits give a token for free.
        bonus = verdict[len(proposals)]
        if greedy:
            return len(proposals), int(bonus.argmax())
        return len(proposals), int(
            torch.multinomial(_distribution(bonus, generation), 1, generator=rng)
        )

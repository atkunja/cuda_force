"""Tests for speculative decoding.

The claim that needs proving is not that this is fast — it is that it is
*lossless*. Speed is measured elsewhere; here the tests establish that
speculation does not change what the target model would have produced.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cudaforge.config import GenerationConfig
from cudaforge.speculative import (
    SpeculativeDecoder,
    SpeculativeStats,
    _distribution,
    expected_tokens_per_call,
)


def model(layers: int, width: int, heads: int, vocab: int = 64, seed: int = 0):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(seed)
    return transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            n_layer=layers, n_head=heads, n_embd=width, vocab_size=vocab, n_positions=256
        )
    ).eval()


def greedy_reference(target, input_ids: torch.Tensor, tokens: int) -> list[int]:
    """Plain one-token-at-a-time greedy decoding: the thing to be matched."""
    produced, step, cache = [], input_ids, None
    with torch.inference_mode():
        for _ in range(tokens):
            output = target(input_ids=step, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            token = int(output.logits[0, -1, :].argmax())
            produced.append(token)
            step = torch.tensor([[token]])
    return produced


class ListCache:
    """The whole cache interface `SpeculativeDecoder` requires: `crop`.

    Everything else about a KV cache is the model's business — the decoder only
    ever rolls one back, and by a count rather than to an absolute length. Two
    things follow from writing it out. These tests stop depending on
    transformers, so the algorithm is checked even where it is not installed.
    And the contract is pinned: if the decoder starts calling something else,
    this fails loudly instead of silently binding to a library type.
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
    """A causal model whose next token depends on its entire history.

    Randomly initialised transformers are useless for testing cache handling:
    their argmax barely moves with context, so a corrupted KV cache yields the
    same tokens and the bug passes unnoticed. That is not hypothetical — it hid
    a real desynchronisation between the draft's cache and its own proposals,
    which only surfaced once the fixture actually depended on history.

    Here the history *is* the cache, so any mis-rollback changes the prediction.
    `stumble` makes the model disagree with an otherwise identical one every nth
    call, which simulates a partially accurate draft deterministically.
    """

    def __init__(self, vocab: int = 31, stumble: int = 0) -> None:
        self.vocab = vocab
        self.stumble = stumble
        self.calls = 0

    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        cache = past_key_values if past_key_values is not None else ListCache()
        added = input_ids[0].tolist()
        before = list(cache.tokens)
        cache.extend(added)

        self.calls += 1
        drift = 1 if self.stumble and self.calls % self.stumble == 0 else 0

        rows = []
        for offset in range(len(added)):
            prefix = before + added[: offset + 1]
            value = int(sum(prefix) * 5 + len(prefix) * 13 + drift) % self.vocab
            row = torch.full((self.vocab,), -8.0)
            row[value] = 8.0
            rows.append(row)

        return SimpleNamespace(logits=torch.stack(rows).unsqueeze(0), past_key_values=cache)


CHAIN_PROMPT = torch.tensor([[3, 1, 4, 1, 5]])


def test_a_history_dependent_model_is_decoded_exactly(pair_unused=None):
    """Greedy equality on a model that actually reads its cache.

    The transformer fixtures above accept every proposal, so they never reach
    the rejection or rollback paths. This draft is deliberately wrong every
    third call, forcing both.
    """
    target = ChainModel()
    draft = ChainModel(stumble=3)

    produced, stats = SpeculativeDecoder(target, draft, lookahead=4).generate(
        CHAIN_PROMPT, GenerationConfig(max_new_tokens=20, temperature=0.0)
    )

    assert produced == greedy_reference(ChainModel(), CHAIN_PROMPT, 20)

    # Pinned rather than bounded. Nothing here is random, so the exact counts are
    # reproducible — and acceptance is the only signal that catches a draft
    # decoding from a corrupted cache, which costs speed without ever changing
    # the output. A loose `0 < rate < 1` bound lets that through.
    assert stats.proposed == 38
    assert stats.accepted == 10
    assert stats.target_calls == 11


def test_an_identical_draft_is_accepted_in_full_on_a_history_dependent_model():
    """Catches the draft cache falling out of step with its own proposals.

    A draft that *is* the target must agree every time. When the draft failed to
    ingest the last proposal of each block, its history developed a gap and this
    fell to 7% — while every transformer-based test stayed green.
    """
    target = ChainModel()
    produced, stats = SpeculativeDecoder(target, ChainModel(), lookahead=4).generate(
        CHAIN_PROMPT, GenerationConfig(max_new_tokens=20, temperature=0.0)
    )

    assert produced == greedy_reference(ChainModel(), CHAIN_PROMPT, 20)
    assert stats.acceptance_rate == 1.0
    assert stats.tokens_per_target_call == 5.0


@pytest.fixture(scope="module")
def pair():
    return model(4, 64, 4, seed=0), model(1, 32, 2, seed=1)


@pytest.mark.parametrize("lookahead", [1, 2, 4, 8])
def test_greedy_speculation_matches_the_target_alone(pair, lookahead):
    """The correctness anchor.

    Greedy speculation keeps a proposal only when it equals the target's own
    argmax, so the output must be identical token for token — at every
    lookahead, including one longer than the budget.
    """
    target, draft = pair
    prompt = torch.randint(1, 60, (1, 6), generator=torch.Generator().manual_seed(3))
    expected = greedy_reference(target, prompt, 24)

    produced, _ = SpeculativeDecoder(target, draft, lookahead=lookahead).generate(
        prompt, GenerationConfig(max_new_tokens=24, temperature=0.0)
    )
    assert produced == expected


def test_a_draft_that_is_the_target_is_never_rejected(pair):
    target, _ = pair
    prompt = torch.randint(1, 60, (1, 6), generator=torch.Generator().manual_seed(3))

    produced, stats = SpeculativeDecoder(target, target, lookahead=4).generate(
        prompt, GenerationConfig(max_new_tokens=20, temperature=0.0)
    )
    assert produced == greedy_reference(target, prompt, 20)
    assert stats.acceptance_rate == 1.0
    # Every call yields its proposals plus the free trailing token.
    assert stats.tokens_per_target_call > 4.0


def test_a_useless_draft_still_makes_progress(pair):
    """The guarantee that the loop cannot stall.

    On rejection the target's own token is emitted, so even a draft that is
    wrong every single time produces exactly one token per target call — never
    zero.
    """
    target, _ = pair
    prompt = torch.randint(1, 60, (1, 5), generator=torch.Generator().manual_seed(4))

    class AlwaysWrong:
        """Proposes a token the target's distribution ranks last."""

        def __call__(self, input_ids, past_key_values=None, use_cache=True):
            with torch.inference_mode():
                real = target(
                    input_ids=input_ids, past_key_values=past_key_values, use_cache=use_cache
                )
            # Invert the ordering so the argmax becomes the least likely token.
            return type(real)(logits=-real.logits, past_key_values=real.past_key_values)

    produced, stats = SpeculativeDecoder(target, AlwaysWrong(), lookahead=4).generate(
        prompt, GenerationConfig(max_new_tokens=12, temperature=0.0)
    )

    assert produced == greedy_reference(target, prompt, 12)
    assert stats.acceptance_rate == 0.0
    assert stats.tokens_per_target_call == 1.0


def test_the_token_budget_is_exact(pair):
    """Blocks yield up to lookahead + 1 tokens, so the last one must be clipped."""
    target, draft = pair
    prompt = torch.randint(1, 60, (1, 5), generator=torch.Generator().manual_seed(5))

    for budget in (1, 2, 3, 7, 10):
        produced, stats = SpeculativeDecoder(target, draft, lookahead=4).generate(
            prompt, GenerationConfig(max_new_tokens=budget, temperature=0.0)
        )
        assert len(produced) == budget
        assert stats.generated == budget


def test_speculative_sampling_reproduces_the_target_distribution():
    """The losslessness claim, tested where it could actually fail.

    Greedy equality is close to true by construction. The sampling path is not:
    a proposal is accepted with probability `p/q` and, on rejection, replaced by
    a draw from the normalised residual `max(0, p - q)`. Those branches compose
    back to `p` only if the rule is exactly right, and an error would produce
    plausible text with a quietly wrong distribution — which no equality
    assertion catches.

    So the empirical distribution over many single-token generations is compared
    against the target's own softmax. The draft's distribution is *imposed*
    rather than learned: two randomly initialised models both come out close to
    uniform, agree with each other, and would leave the rejection branch
    carrying almost no traffic. Skewing the draft onto one token forces the
    residual path to do most of the work, which is where a mistake would live.
    """
    vocabulary = 16
    target = model(2, 32, 2, vocab=vocabulary, seed=0)
    small = model(1, 16, 2, vocab=vocabulary, seed=1)

    class Skewed:
        """Borrows the small model's KV cache but imposes the distribution."""

        def __call__(self, input_ids, past_key_values=None, use_cache=True):
            real = small(input_ids=input_ids, past_key_values=past_key_values, use_cache=use_cache)
            logits = torch.zeros_like(real.logits)
            logits[..., 0] = 4.0
            return type(real)(logits=logits, past_key_values=real.past_key_values)

    prompt = torch.randint(1, vocabulary - 1, (1, 4), generator=torch.Generator().manual_seed(2))
    settings = GenerationConfig(max_new_tokens=1, temperature=1.0, top_p=1.0, top_k=0)

    with torch.inference_mode():
        expected = target(input_ids=prompt).logits[0, -1, :].softmax(-1)

    decoder = SpeculativeDecoder(target, Skewed(), lookahead=3)
    trials = 3000
    counts = torch.zeros(vocabulary)
    rejected = 0
    for trial in range(trials):
        token, stats = decoder.generate(prompt, settings, seed=trial)
        counts[token[0]] += 1
        rejected += stats.proposed - stats.accepted

    # Both branches must carry real traffic, or the test proves nothing about
    # whichever one went unused.
    assert 0.4 < rejected / trials < 0.95, f"rejection rate {rejected / trials:.1%}"

    observed = counts / trials

    # The sharpest coordinate: the draft proposes token 0 almost every time, so
    # that is where a broken acceptance rule leaks mass. Replacing the residual
    # draw with a plain target draw — the most plausible way to get this wrong —
    # nearly doubles this one entry (0.068 -> 0.118) while moving total
    # variation only from 0.023 to 0.055. An aggregate bound loose enough not to
    # flake would let that through; this does not.
    #
    # One-coordinate noise at these settings is sqrt(p(1-p)/n) ~ 0.005, so the
    # bound sits at roughly three standard deviations.
    drafted = abs(float(observed[0] - expected[0]))
    assert drafted < 0.015, f"mass on the drafted token is off by {drafted:.4f}"

    distance = 0.5 * (observed - expected).abs().sum()
    # Sampling noise alone is about sqrt(V / (2 pi n)) ~ 0.029 at these settings.
    assert distance < 0.045, f"total variation distance {distance:.4f} is too large"


def test_greedy_and_sampling_at_temperature_zero_agree(pair):
    """`temperature == 0` must take the greedy path, not divide by zero."""
    target, draft = pair
    prompt = torch.randint(1, 60, (1, 5), generator=torch.Generator().manual_seed(6))
    produced, _ = SpeculativeDecoder(target, draft, lookahead=3).generate(
        prompt, GenerationConfig(max_new_tokens=8, temperature=0.0)
    )
    assert produced == greedy_reference(target, prompt, 8)


def test_sampling_is_reproducible_from_a_seed(pair):
    target, draft = pair
    prompt = torch.randint(1, 60, (1, 5), generator=torch.Generator().manual_seed(7))
    settings = GenerationConfig(max_new_tokens=10, temperature=1.0)
    decoder = SpeculativeDecoder(target, draft, lookahead=3)

    first, _ = decoder.generate(prompt, settings, seed=11)
    again, _ = decoder.generate(prompt, settings, seed=11)
    assert first == again


def test_lookahead_must_be_positive():
    # ChainModel rather than the transformer fixture: argument validation has
    # nothing to do with the model, and should not be skipped without one.
    with pytest.raises(ValueError, match="lookahead"):
        SpeculativeDecoder(ChainModel(), ChainModel(), lookahead=0)


def test_only_one_prompt_at_a_time():
    """Batched speculation is out of scope; the refusal must be explicit."""
    decoder = SpeculativeDecoder(ChainModel(), ChainModel())
    with pytest.raises(ValueError, match="single prompt"):
        decoder.generate(torch.randint(1, 60, (2, 5)), GenerationConfig(max_new_tokens=4))
    with pytest.raises(ValueError, match="single prompt"):
        decoder.generate(torch.randint(1, 60, (5,)), GenerationConfig(max_new_tokens=4))


class ImperfectDraft(ChainModel):
    """Agrees with an identical target with probability `rate`, per token.

    Defined here rather than imported from `benchmarks/`: `scripts/test.sh` runs
    the `pytest` console script, which does not put the checkout on `sys.path`,
    so a test that imports a benchmark passes locally and fails in the suite.
    """

    def __init__(self, rate: float, seed: int, vocab: int = 64) -> None:
        super().__init__(vocab)
        self.rate = rate
        self._rng = torch.Generator().manual_seed(seed)

    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        output = super().__call__(input_ids, past_key_values, use_cache)
        logits = output.logits
        for position in range(logits.shape[1]):
            if float(torch.rand(1, generator=self._rng)) >= self.rate:
                # Shift the peak one token over, so this proposal is rejected.
                best = int(logits[0, position].argmax())
                logits[0, position, best] = -8.0
                logits[0, position, (best + 1) % self.vocab] = 8.0
        return output


def test_throughput_matches_the_closed_form():
    """Validates the accept-and-bonus arithmetic against theory.

    With acceptance probability `a` per token the accepted count is geometric
    capped at `k`, and every block emits one further token — the target's own on
    rejection, or the free trailing one on a full accept. So

        E[tokens per target call] = (1 - a^(k+1)) / (1 - a)

    This catches off-by-one errors in the bonus token and in the verification
    window that token-equality tests cannot see, because those change *how many*
    tokens a call yields without making any individual token wrong. Dropping the
    bonus fails this while leaving the generated sequence identical.
    """
    prompt = torch.tensor([[3, 1, 4, 1, 5]])
    settings = GenerationConfig(max_new_tokens=600, temperature=0.0)

    for rate in (0.5, 0.7, 0.9):
        for lookahead in (1, 2, 4):
            _, stats = SpeculativeDecoder(
                ChainModel(vocab=64), ImperfectDraft(rate, seed=7), lookahead=lookahead
            ).generate(prompt, settings)

            predicted = expected_tokens_per_call(rate, lookahead)
            measured = stats.tokens_per_target_call
            # 600 tokens keeps sampling noise well under 8%.
            assert abs(measured - predicted) / predicted < 0.08, (
                f"a={rate} k={lookahead}: measured {measured:.3f}, expected {predicted:.3f}"
            )


def test_a_perfect_draft_reaches_the_ceiling():
    for lookahead in (1, 3, 5):
        assert expected_tokens_per_call(1.0, lookahead) == lookahead + 1


def test_the_closed_form_saturates_with_lookahead():
    """The property that makes a large lookahead pointless at low acceptance.

    The limit as `k` grows is `1 / (1 - a)`, and it is approached fast: at
    `a = 0.5` a lookahead of 4 already captures 97% of what an infinite one
    would. Every proposal past that point is draft work thrown away.
    """
    limit = 1 / (1 - 0.5)
    assert expected_tokens_per_call(0.5, 4) / limit > 0.96
    assert expected_tokens_per_call(0.5, 4) < limit
    # Far enough out the double-precision result reaches the limit exactly,
    # which is the saturation being asserted rather than a defect.
    assert expected_tokens_per_call(0.5, 64) == limit

    # A good draft keeps paying off much longer: at a = 0.9 the limit is 10 and
    # a lookahead of 8 has reached only two thirds of it.
    assert expected_tokens_per_call(0.9, 8) / (1 / (1 - 0.9)) < 0.7


def test_the_closed_form_rejects_nonsense():
    with pytest.raises(ValueError, match="probability"):
        expected_tokens_per_call(1.5, 4)
    with pytest.raises(ValueError, match="lookahead"):
        expected_tokens_per_call(0.5, 0)


# --- statistics and filtering ----------------------------------------------


def test_stats_are_zero_safe():
    empty = SpeculativeStats()
    assert empty.acceptance_rate == 0.0
    assert empty.tokens_per_target_call == 0.0


def test_stats_report_the_two_signals():
    stats = SpeculativeStats(target_calls=4, proposed=16, accepted=12, generated=16)
    assert stats.acceptance_rate == 0.75
    assert stats.tokens_per_target_call == 4.0


def test_top_k_zeroes_everything_outside_the_k_best():
    logits = torch.tensor([4.0, 3.0, 2.0, 1.0])
    probabilities = _distribution(logits, GenerationConfig(temperature=1.0, top_k=2, top_p=1.0))
    assert probabilities[2] == 0.0 and probabilities[3] == 0.0
    assert pytest.approx(float(probabilities.sum()), abs=1e-6) == 1.0


def test_top_p_keeps_the_leading_token_and_renormalises():
    logits = torch.tensor([9.0, 0.0, 0.0, 0.0])
    probabilities = _distribution(logits, GenerationConfig(temperature=1.0, top_p=0.01, top_k=0))
    assert probabilities[0] > 0.99
    assert pytest.approx(float(probabilities.sum()), abs=1e-6) == 1.0


def test_a_filtered_distribution_still_sums_to_one():
    logits = torch.randn(32, generator=torch.Generator().manual_seed(1))
    probabilities = _distribution(logits, GenerationConfig(temperature=0.8, top_k=8, top_p=0.9))
    assert pytest.approx(float(probabilities.sum()), abs=1e-6) == 1.0
    assert int((probabilities > 0).sum()) <= 8

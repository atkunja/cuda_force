"""Tests for speculative decoding.

The claim that needs proving is not that this is fast — it is that it is
*lossless*. Speed is measured elsewhere; here the tests establish that
speculation does not change what the target model would have produced.
"""

from __future__ import annotations

import pytest
import torch
from cudaforge.config import GenerationConfig
from cudaforge.speculative import SpeculativeDecoder, SpeculativeStats, _distribution


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


def test_lookahead_must_be_positive(pair):
    target, draft = pair
    with pytest.raises(ValueError, match="lookahead"):
        SpeculativeDecoder(target, draft, lookahead=0)


def test_only_one_prompt_at_a_time(pair):
    """Batched speculation is out of scope; the refusal must be explicit."""
    target, draft = pair
    decoder = SpeculativeDecoder(target, draft)
    with pytest.raises(ValueError, match="single prompt"):
        decoder.generate(torch.randint(1, 60, (2, 5)), GenerationConfig(max_new_tokens=4))
    with pytest.raises(ValueError, match="single prompt"):
        decoder.generate(torch.randint(1, 60, (5,)), GenerationConfig(max_new_tokens=4))


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

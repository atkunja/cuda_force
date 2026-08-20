"""Tests for the step-wise runner over a real causal model.

The model is built locally rather than downloaded: the batch-composition logic
is what is worth testing, and it does not depend on which model is loaded.
"""

from __future__ import annotations

import pytest
import torch

from cudaforge.config import EngineConfig, GenerationConfig
from cudaforge.stepwise import SequenceState, StepwiseRunner


class StubTokenizer:
    """A whitespace tokenizer, so no vocabulary is downloaded."""

    pad_token_id = 0
    eos_token_id: int | None = None
    eos_token = "<eos>"
    padding_side = "right"

    def __call__(self, prompts, return_tensors=None, padding=False, truncation=False):
        encoded = [[len(word) % 60 + 1 for word in prompt.split()] for prompt in prompts]
        width = max(len(ids) for ids in encoded)

        input_ids, attention_mask = [], []
        for ids in encoded:
            pad = width - len(ids)
            input_ids.append([self.pad_token_id] * pad + ids)
            attention_mask.append([0] * pad + [1] * len(ids))

        class Batch(dict):
            def to(self, _device):
                return self

        return Batch(
            input_ids=torch.tensor(input_ids),
            attention_mask=torch.tensor(attention_mask),
        )

    def decode(self, ids, skip_special_tokens=False):
        return f"|{int(ids[0])}"


def tiny_lm():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=64, n_positions=128)
    )
    model.eval()
    return model


def make_runner(model=None, tokenizer=None):
    pytest.importorskip("transformers")
    from cudaforge.stepwise_transformers import TransformersStepwiseRunner

    return TransformersStepwiseRunner(
        EngineConfig(device="cpu", dtype="float32"),
        model=model if model is not None else tiny_lm(),
        tokenizer=tokenizer if tokenizer is not None else StubTokenizer(),
    )


def greedy(prompt, tokens=4):
    return SequenceState(1, prompt, GenerationConfig(max_new_tokens=tokens, temperature=0.0))


def drain(runner, states):
    while any(not state.finished for state in states):
        runner.decode_step(states)


def test_the_runner_satisfies_the_stepwise_protocol():
    assert isinstance(make_runner(), StepwiseRunner)


def test_cached_decoding_matches_an_uncached_greedy_loop():
    """The correctness anchor.

    Everything else here is about batch bookkeeping; this is the check that the
    bookkeeping has not changed what the model would have said. The reference
    re-runs the full prompt each step with no cache at all, which is slow and
    obviously correct.
    """
    model, tokenizer = tiny_lm(), StubTokenizer()

    encoded = tokenizer(["alpha beta gamma"])
    ids, mask = encoded["input_ids"], encoded["attention_mask"]
    reference = []
    with torch.inference_mode():
        for _ in range(6):
            positions = (mask.cumsum(dim=-1) - 1).clamp(min=0)
            logits = model(input_ids=ids, attention_mask=mask, position_ids=positions).logits[
                :, -1, :
            ]
            token = logits.argmax(dim=-1, keepdim=True)
            reference.append(int(token.item()))
            ids = torch.cat([ids, token], dim=1)
            mask = torch.cat([mask, torch.ones_like(token)], dim=1)

    runner = make_runner(model, tokenizer)
    states = [greedy("alpha beta gamma", tokens=6)]
    runner.prefill(states)
    drain(runner, states)

    assert [int(token.lstrip("|")) for token in states[0].tokens] == reference


def test_a_short_prompt_is_unaffected_by_a_long_neighbour():
    """Left padding and mask-derived positions, tested by their consequence.

    If the padded row's positions were taken from its index rather than its
    mask, or if the padding were on the right, this row would produce different
    tokens in a batch than it does alone. That is the failure that looks like a
    bad model instead of a bad harness.
    """
    model, tokenizer = tiny_lm(), StubTokenizer()

    alone = [greedy("aa")]
    solo = make_runner(model, tokenizer)
    solo.prefill(alone)
    drain(solo, alone)

    batched = [greedy("aa"), SequenceState(2, "bb ccc dddd eeeee", alone[0].generation)]
    together = make_runner(model, tokenizer)
    together.prefill(batched)
    drain(together, batched)

    assert batched[0].tokens == alone[0].tokens


def test_admitting_a_sequence_mid_flight_grows_the_cache():
    runner = make_runner()
    states = [greedy("aa bb"), SequenceState(2, "cc", greedy("x").generation)]
    runner.prefill(states)
    assert runner.active_rows == 2
    assert runner.cache_length == 2

    runner.decode_step(states)
    latecomer = SequenceState(3, "dd ee ff gg", greedy("x").generation)
    runner.prefill([latecomer])

    assert runner.active_rows == 3
    # The newcomer's four prompt tokens outrun the three the incumbents hold, so
    # the incumbents are the ones padded.
    assert runner.cache_length == 4


def run_alone(model, tokenizer, prompt, tokens):
    """What a sequence produces with no neighbours at all."""
    runner = make_runner(model, tokenizer)
    states = [SequenceState(1, prompt, GenerationConfig(max_new_tokens=tokens, temperature=0.0))]
    runner.prefill(states)
    drain(runner, states)
    return states[0].tokens


def test_a_latecomer_is_unaffected_by_being_merged_into_a_longer_cache():
    """The merge path, checked by its output rather than its shapes.

    A sequence admitted mid-flight has its cache concatenated onto rows that
    already hold more history. If the shorter side were padded on the right, its
    newest token would no longer sit at the position the next decode step reads
    from, and it would generate from the wrong place. Shape assertions do not
    notice that; token equality does.
    """
    model, tokenizer = tiny_lm(), StubTokenizer()
    expected = run_alone(model, tokenizer, "aa bb", tokens=4)

    runner = make_runner(model, tokenizer)
    incumbent = SequenceState(1, "cc dd ee ff", GenerationConfig(max_new_tokens=8, temperature=0.0))
    states = [incumbent]
    runner.prefill(states)
    runner.decode_step(states)

    latecomer = SequenceState(2, "aa bb", GenerationConfig(max_new_tokens=4, temperature=0.0))
    runner.prefill([latecomer])
    states.append(latecomer)
    drain(runner, states)

    assert latecomer.tokens == expected


def test_an_incumbent_is_unaffected_by_a_longer_sequence_joining_it():
    """The mirror image: when the newcomer holds more history, the padding lands
    on the rows already in the cache. Those must survive it untouched."""
    model, tokenizer = tiny_lm(), StubTokenizer()
    expected = run_alone(model, tokenizer, "aa", tokens=5)

    runner = make_runner(model, tokenizer)
    incumbent = SequenceState(1, "aa", GenerationConfig(max_new_tokens=5, temperature=0.0))
    states = [incumbent]
    runner.prefill(states)

    latecomer = SequenceState(
        2, "bb cc dd ee ff gg", GenerationConfig(max_new_tokens=5, temperature=0.0)
    )
    runner.prefill([latecomer])
    states.append(latecomer)
    drain(runner, states)

    assert incumbent.tokens == expected


def test_padding_goes_on_the_left():
    from cudaforge.stepwise_transformers import TransformersStepwiseRunner

    pad = TransformersStepwiseRunner._left_pad
    tensor = torch.ones(1, 1, 2, 1)
    assert pad(tensor, 4, 2).squeeze().tolist() == [0, 0, 1, 1]
    # Already long enough, and longer than asked for, are both left alone.
    assert pad(tensor, 2, 2).shape == tensor.shape
    assert pad(tensor, 1, 2).shape == tensor.shape


def test_every_row_stays_right_aligned_after_a_merge():
    """The invariant left padding exists to maintain.

    A single decode step feeds one token per row and reads the position at the
    right edge of each. That is only well-defined if every row's newest token is
    at that edge — which means the zeros must all sit at the front. Expressed as
    a mask, no row may contain a 1 before a 0.

    Asserted structurally rather than through generated tokens: padding the
    wrong side keeps the cache and mask consistent with *each other* and only
    corrupts the positions, which a small model can absorb without changing its
    argmax. The invariant does not depend on the model noticing.
    """
    runner = make_runner()
    states = [SequenceState(1, "aa bb cc dd", greedy("x").generation)]
    runner.prefill(states)
    runner.decode_step(states)
    runner.prefill([SequenceState(2, "ee", greedy("x").generation)])

    mask = runner._mask
    assert mask.shape[0] == 2
    for row in mask.tolist():
        assert row == sorted(row), f"padding is not on the left: {row}"
        assert row[-1] == 1, f"row does not end in a real token: {row}"


def test_eviction_removes_the_row_and_leaves_the_others_intact():
    model, tokenizer = tiny_lm(), StubTokenizer()

    runner = make_runner(model, tokenizer)
    keep = greedy("aa bb")
    drop = SequenceState(2, "cc dd", keep.generation)
    states = [keep, drop]
    runner.prefill(states)
    runner.decode_step(states)
    runner.evict(2)
    assert runner.active_rows == 1

    states = [keep]
    drain(runner, states)

    undisturbed = make_runner(model, tokenizer)
    reference = [greedy("aa bb"), SequenceState(2, "cc dd", keep.generation)]
    undisturbed.prefill(reference)
    drain(undisturbed, reference)

    assert keep.tokens == reference[0].tokens


def test_evicting_an_unknown_sequence_is_a_no_op():
    runner = make_runner()
    states = [greedy("aa")]
    runner.prefill(states)
    runner.evict(999)
    assert runner.active_rows == 1


def test_evicting_every_row_empties_the_cache():
    runner = make_runner()
    states = [greedy("aa"), SequenceState(2, "bb", greedy("x").generation)]
    runner.prefill(states)
    runner.evict(1)
    runner.evict(2)
    assert runner.active_rows == 0
    assert runner.cache_length == 0
    # And a decode step against an empty cache does nothing rather than raising.
    runner.decode_step(states)


def test_a_retired_sequence_is_dropped_from_the_cache_on_the_next_step():
    """The scheduler retires sequences; the runner must notice.

    A finished row left in the cache would keep consuming a slot and, worse,
    shift every other row's index out of alignment with `_rows`.
    """
    runner = make_runner()
    short = SequenceState(1, "aa", GenerationConfig(max_new_tokens=1, temperature=0.0))
    long = SequenceState(2, "bb", GenerationConfig(max_new_tokens=6, temperature=0.0))
    states = [short, long]

    runner.prefill(states)
    assert short.finished and runner.active_rows == 2

    runner.decode_step(states)
    assert runner.active_rows == 1


def test_the_end_of_sequence_token_stops_a_sequence_early():
    model = tiny_lm()
    tokenizer = StubTokenizer()

    # Learn which token this model produces first, then declare it the eos.
    probe = make_runner(model, tokenizer)
    states = [greedy("aa bb", tokens=5)]
    probe.prefill(states)
    first = int(states[0].tokens[0].lstrip("|"))

    tokenizer = StubTokenizer()
    tokenizer.eos_token_id = first
    runner = make_runner(model, tokenizer)
    stopped = [greedy("aa bb", tokens=5)]
    runner.prefill(stopped)

    assert stopped[0].stopped_early
    assert stopped[0].finished
    assert stopped[0].generated == 0


def test_prefilling_nothing_is_harmless():
    runner = make_runner()
    runner.prefill([])
    assert runner.active_rows == 0
    assert runner.cache_length == 0


def test_a_tokenizer_without_a_pad_token_borrows_the_end_of_sequence_one():
    """Batching needs a pad id, and many causal tokenizers ship without one."""

    class Unpadded(StubTokenizer):
        pad_token_id = None
        eos_token = "<end>"

    tokenizer = Unpadded()
    make_runner(tiny_lm(), tokenizer)
    assert tokenizer.pad_token == "<end>"
    assert tokenizer.padding_side == "left"


def test_stepping_sequences_the_runner_never_saw_does_nothing():
    """The scheduler and the cache can disagree; the runner must not guess.

    Reached when every state handed to a step belongs to some other runner, so
    reconciliation empties the row list. Returning is right — inventing rows for
    them would generate from a cache that holds nobody's history.
    """
    runner = make_runner()
    known = [greedy("aa")]
    runner.prefill(known)

    stranger = [SequenceState(99, "zz", greedy("x").generation)]
    runner.decode_step(stranger)

    assert runner.active_rows == 0
    assert stranger[0].generated == 0


def test_the_description_names_the_model_and_device():
    runner = make_runner()
    assert "TransformersStepwiseRunner" in runner.description
    assert "cpu" in runner.description


# --- sampling ---------------------------------------------------------------


def test_temperature_zero_is_argmax():
    from cudaforge.stepwise_transformers import _sample

    logits = torch.tensor([[0.1, 5.0, 0.2], [9.0, 0.0, 0.0]])
    chosen = _sample(logits, [GenerationConfig(temperature=0.0)] * 2)
    assert chosen.tolist() == [1, 0]


def test_rows_sample_under_their_own_settings():
    """Continuous batching mixes requests, so one row being greedy must not
    force the others to be."""
    from cudaforge.stepwise_transformers import _sample

    torch.manual_seed(0)
    logits = torch.tensor([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    settings = [
        GenerationConfig(temperature=0.0),
        GenerationConfig(temperature=2.0, top_p=1.0, top_k=0),
    ]
    draws = {tuple(_sample(logits, settings).tolist()) for _ in range(60)}

    assert {row[0] for row in draws} == {2}, "the greedy row must never wander"
    assert len({row[1] for row in draws}) > 1, "the sampled row must not be pinned"


def test_top_k_confines_the_choice():
    from cudaforge.stepwise_transformers import _sample

    torch.manual_seed(0)
    logits = torch.tensor([[3.0, 2.9, 2.8, 2.7]])
    settings = [GenerationConfig(temperature=1.0, top_k=2, top_p=1.0)]
    assert {int(_sample(logits, settings)) for _ in range(80)} <= {0, 1}


def test_top_p_confines_the_choice():
    from cudaforge.stepwise_transformers import _sample

    torch.manual_seed(0)
    # Softmax over these puts almost all mass on the first entry.
    logits = torch.tensor([[9.0, 0.0, 0.0, 0.0]])
    settings = [GenerationConfig(temperature=1.0, top_p=0.5, top_k=0)]
    assert {int(_sample(logits, settings)) for _ in range(80)} == {0}


def test_top_p_never_removes_the_leading_token():
    """A top_p below the leading token's own mass must still leave it selectable,
    or the row would have nothing to sample from."""
    from cudaforge.stepwise_transformers import _sample

    torch.manual_seed(0)
    logits = torch.tensor([[9.0, 0.0, 0.0]])
    settings = [GenerationConfig(temperature=1.0, top_p=0.01, top_k=0)]
    assert int(_sample(logits, settings)) == 0

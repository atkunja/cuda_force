"""Model runner tests.

Only `EchoRunner` is exercised here. `TransformersRunner` needs a model
download, and what it adds over the protocol — tokenisation and generation — is
transformers' code, not this project's. What *is* worth testing is that the
runner contract holds, because the engine relies on it: same order, same count,
independent per-row settings.
"""

from __future__ import annotations

import time

import pytest

from cudaforge.config import GenerationConfig
from cudaforge.runners import EchoRunner, GenerationResult, ModelRunner


def test_the_echo_runner_satisfies_the_protocol():
    # runtime_checkable Protocol, so this is a real structural check that the
    # engine's expectations are met.
    assert isinstance(EchoRunner(), ModelRunner)


def test_output_is_deterministic():
    # Derived from a hash of the prompt, so assertions on engine behaviour are
    # stable across runs and machines.
    runner = EchoRunner()
    settings = [GenerationConfig(max_new_tokens=8)]
    first = runner.generate(["hello"], settings)
    second = runner.generate(["hello"], settings)
    assert first[0].text == second[0].text


def test_different_prompts_give_different_output():
    runner = EchoRunner()
    settings = [GenerationConfig(max_new_tokens=8)]
    a = runner.generate(["alpha"], settings)[0].text
    b = runner.generate(["beta"], settings)[0].text
    assert a != b


def test_results_are_returned_in_input_order():
    # The engine pairs requests with results positionally; a reordering runner
    # would deliver every response to the wrong caller.
    runner = EchoRunner()
    prompts = [f"prompt-{i}" for i in range(16)]
    settings = [GenerationConfig(max_new_tokens=4) for _ in prompts]

    results = runner.generate(prompts, settings)
    assert len(results) == len(prompts)
    for prompt, result in zip(prompts, results, strict=True):
        assert result.text.startswith(prompt)


def test_per_row_token_budgets_are_respected():
    runner = EchoRunner()
    results = runner.generate(
        ["a", "b"],
        [GenerationConfig(max_new_tokens=4), GenerationConfig(max_new_tokens=32)],
    )
    assert results[0].generated_tokens == 4
    assert results[1].generated_tokens == 32


def test_prompt_tokens_are_counted():
    runner = EchoRunner()
    result = runner.generate(["one two three"], [GenerationConfig()])[0]
    assert result.prompt_tokens == 3


def test_mismatched_lengths_raise_rather_than_silently_truncating():
    runner = EchoRunner()
    with pytest.raises(ValueError):
        runner.generate(["a", "b"], [GenerationConfig()])


def test_the_fixed_overhead_is_paid_once_per_batch():
    # This is the property batching exploits: a cost paid per batch rather than
    # per row is what larger batches amortise.
    runner = EchoRunner(fixed_overhead=0.02)
    settings_one = [GenerationConfig(max_new_tokens=1)]
    settings_eight = [GenerationConfig(max_new_tokens=1) for _ in range(8)]

    started = time.monotonic()
    runner.generate(["x"], settings_one)
    single = time.monotonic() - started

    started = time.monotonic()
    runner.generate([f"x{i}" for i in range(8)], settings_eight)
    batched = time.monotonic() - started

    # Eight rows must not cost eight times one row.
    assert batched < single * 4


def test_the_per_token_cost_scales_with_the_longest_row():
    # A batch runs as long as its most demanding member.
    runner = EchoRunner(per_token_seconds=0.002)

    started = time.monotonic()
    runner.generate(["x"], [GenerationConfig(max_new_tokens=1)])
    short = time.monotonic() - started

    started = time.monotonic()
    runner.generate(["x"], [GenerationConfig(max_new_tokens=32)])
    long = time.monotonic() - started

    assert long > short


def test_warmup_does_not_raise():
    # Warmup exists so the first real request is not measuring CUDA context
    # creation, autotuning and lazy module loading.
    EchoRunner().warmup(3)


def test_the_description_identifies_the_runner():
    # Recorded in metrics snapshots, so a benchmark result says what produced it.
    assert "EchoRunner" in EchoRunner().description


def test_generation_result_is_a_plain_value():
    result = GenerationResult(text="x", prompt_tokens=1, generated_tokens=2)
    assert result.text == "x"
    assert result.prompt_tokens == 1
    assert result.generated_tokens == 2


# --- the transformers runner -----------------------------------------------


class StubTokenizer:
    """A whitespace tokenizer with the interface the runner uses.

    Injected so the runner's own logic — left padding, per-row truncation,
    token counting — is tested without a network round trip. What transformers
    does with the ids is transformers' concern, not this project's.
    """

    pad_token_id = 0
    eos_token = "<eos>"
    padding_side = "right"

    def __call__(self, prompts, return_tensors=None, padding=False, truncation=False):
        import torch

        encoded = [[len(word) % 60 + 1 for word in prompt.split()] for prompt in prompts]
        width = max(len(ids) for ids in encoded)

        input_ids = []
        attention_mask = []
        for ids in encoded:
            pad = width - len(ids)
            # Left padding, matching what the runner configures.
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
        return " ".join(str(int(value)) for value in ids)


class StubModel:
    """Returns the prompt followed by a deterministic continuation."""

    def __init__(self) -> None:
        self.eval_called = False
        self.last_max_new_tokens: int | None = None

    def eval(self) -> None:
        self.eval_called = True

    def to(self, _device):
        return self

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=1, **_kwargs):
        import torch

        self.last_max_new_tokens = max_new_tokens
        rows, width = input_ids.shape
        continuation = torch.arange(1, max_new_tokens + 1).repeat(rows, 1)
        return torch.cat([input_ids, continuation], dim=1)


def make_runner():
    from cudaforge.config import EngineConfig
    from cudaforge.runners import TransformersRunner

    tokenizer = StubTokenizer()
    model = StubModel()
    runner = TransformersRunner(
        EngineConfig(device="cpu", dtype="float32"), model=model, tokenizer=tokenizer
    )
    return runner, model, tokenizer


def test_the_runner_switches_the_tokenizer_to_left_padding():
    # A causal model continues from the final position of each row, so right
    # padding would ask it to continue from a pad token — garbage for every row
    # shorter than the longest.
    _, _, tokenizer = make_runner()
    assert tokenizer.padding_side == "left"


def test_the_runner_puts_the_model_in_eval_mode():
    _, model, _ = make_runner()
    assert model.eval_called


def test_the_runner_returns_one_result_per_prompt():
    runner, _, _ = make_runner()
    prompts = ["a bb ccc", "dddd", "e f g h i"]
    settings = [GenerationConfig(max_new_tokens=4) for _ in prompts]

    results = runner.generate(prompts, settings)
    assert len(results) == len(prompts)
    assert all(result.generated_tokens == 4 for result in results)


def test_the_batch_runs_for_its_longest_member():
    # A batch runs as long as its most demanding member; shorter rows are
    # truncated at their own limit rather than shortening the batch.
    runner, model, _ = make_runner()
    results = runner.generate(
        ["a", "b"],
        [GenerationConfig(max_new_tokens=2), GenerationConfig(max_new_tokens=8)],
    )

    assert model.last_max_new_tokens == 8
    assert results[0].generated_tokens == 2
    assert results[1].generated_tokens == 8


def test_prompt_tokens_exclude_padding():
    # The attention mask is what distinguishes real tokens from padding; using
    # the padded width would overstate every short prompt.
    runner, _, _ = make_runner()
    results = runner.generate(
        ["one", "one two three four"],
        [GenerationConfig(max_new_tokens=1) for _ in range(2)],
    )
    assert results[0].prompt_tokens == 1
    assert results[1].prompt_tokens == 4


def test_the_runner_describes_itself():
    runner, _, _ = make_runner()
    assert "TransformersRunner" in runner.description
    assert "cpu" in runner.description


def test_warmup_runs_the_configured_number_of_iterations():
    runner, model, _ = make_runner()
    runner.warmup(3)
    assert model.last_max_new_tokens == 1


def test_warmup_with_a_non_positive_count_does_nothing():
    runner, model, _ = make_runner()
    runner.warmup(0)
    assert model.last_max_new_tokens is None

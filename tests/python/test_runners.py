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

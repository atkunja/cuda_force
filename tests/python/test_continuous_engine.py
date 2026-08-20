"""Tests for the iteration-level serving engine.

`test_continuous.py` covers the scheduler. These cover the engine wrapped around
it: that futures are settled exactly once on every path, and that it is
genuinely substitutable for `InferenceEngine`.
"""

from __future__ import annotations

import threading
import time

import pytest

from cudaforge.config import EngineConfig, GenerationConfig
from cudaforge.continuous_engine import ContinuousEngine
from cudaforge.engine import EngineClosedError, InferenceEngine, ServingEngine
from cudaforge.stepwise import EchoStepwiseRunner


def engine(runner: EchoStepwiseRunner | None = None, **overrides) -> ContinuousEngine:
    """Build an engine.

    The runner goes through the constructor and is never assigned afterwards:
    the batcher captures its own reference at construction, so a later
    assignment to `engine._runner` changes nothing the scheduler can see. A test
    that does that silently exercises the default runner instead of the one it
    believes it installed.
    """
    settings = {
        "max_batch_size": 4,
        "queue_capacity": 64,
        "generation": GenerationConfig(max_new_tokens=3),
    }
    settings.update(overrides)
    return ContinuousEngine(config=EngineConfig(**settings), runner=runner)


def test_a_single_request_comes_back():
    with engine() as running:
        response = running.generate("hello")
    assert response.ok
    assert response.generated_tokens == 3
    assert response.text


def test_every_request_is_answered_exactly_once():
    with engine() as running:
        responses = running.generate_many([f"prompt {index}" for index in range(20)])

    assert len(responses) == 20
    assert all(response.ok for response in responses)
    assert len({response.request_id for response in responses}) == 20


def test_requests_of_different_lengths_all_finish():
    """The case continuous batching exists for: rows freed early get refilled."""
    lengths = [1, 12, 2, 2, 3, 1, 9, 2]
    with engine() as running:
        futures = [
            running.submit(f"p{index}", GenerationConfig(max_new_tokens=length))
            for index, length in enumerate(lengths)
        ]
        produced = [future.result(timeout=30).generated_tokens for future in futures]

    assert produced == lengths


def test_submitting_from_many_threads_is_safe():
    collected: list[object] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        response = engine_under_test.generate(f"thread {index}")
        with lock:
            collected.append(response)

    engine_under_test = engine(max_batch_size=8)
    with engine_under_test:
        threads = [threading.Thread(target=worker, args=(index,)) for index in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

    assert len(collected) == 12


def test_the_timing_split_is_reported():
    # A per-step cost makes inference time measurable rather than noise.
    with engine(
        runner=EchoStepwiseRunner(per_step_seconds=0.004),
        max_batch_size=1,
        generation=GenerationConfig(max_new_tokens=4),
    ) as running:
        response = running.generate("hello")

    # Four steps at 4 ms. Loose enough for a slow machine, tight enough to fail
    # if the configured runner were quietly ignored.
    assert response.inference_time > 0.008
    assert response.queue_time >= 0.0
    assert response.total_latency == response.queue_time + response.inference_time


def test_batch_size_is_reported_as_not_applicable():
    """A request under continuous batching has no single batch size.

    Reporting one would mean picking a moment arbitrarily out of a membership
    that changes for the sequence's whole life. Occupancy belongs to the
    schedule, and `stats()` is where it lives.
    """
    with engine() as running:
        response = running.generate("hello")
        occupancy = running.stats().utilisation

    assert response.batch_size == 0
    assert response.metadata["scheduler"] == "continuous"
    assert 0.0 < occupancy <= 1.0


def test_prompt_tokens_are_zero_without_a_tokeniser():
    """Zero means "not reported", and the echo runner does not tokenise."""
    with engine() as running:
        assert running.generate("a b c").prompt_tokens == 0


# --- failure paths ----------------------------------------------------------


def test_submitting_after_shutdown_is_refused():
    running = engine()
    running.shutdown()
    with pytest.raises(EngineClosedError):
        running.submit("hello")


def test_shutdown_is_idempotent():
    running = engine()
    running.shutdown()
    running.shutdown()


def test_an_oversized_prompt_is_rejected():
    with engine(max_prompt_chars=8) as running, pytest.raises(ValueError, match="exceeds"):
        running.submit("x" * 9)


def test_an_expired_request_is_settled_with_an_error():
    """A dropped request must fail its future, not leave the caller waiting."""
    with engine(runner=EchoStepwiseRunner(per_step_seconds=0.02), max_batch_size=1) as running:
        blocker = running.submit("blocker", GenerationConfig(max_new_tokens=10))
        # Let the blocker be admitted, so it holds the only row.
        time.sleep(0.05)
        # Already past its deadline before the row can free.
        victim = running.submit("victim", deadline_seconds=-1.0)

        response = victim.result(timeout=30)
        assert not response.ok
        assert "RequestExpired" in (response.error or "")
        blocker.result(timeout=30)


def test_outstanding_futures_fail_at_shutdown_rather_than_hang():
    running = engine(
        runner=EchoStepwiseRunner(per_step_seconds=0.5), max_batch_size=1, queue_capacity=64
    )
    futures = [
        running.submit(f"p{index}", GenerationConfig(max_new_tokens=20)) for index in range(6)
    ]

    # A short drain leaves work unfinished, which is the case under test.
    running.shutdown(timeout=0.05)

    outcomes = []
    for future in futures:
        try:
            outcomes.append(future.result(timeout=5).ok)
        except EngineClosedError:
            outcomes.append("abandoned")
    assert len(outcomes) == 6, "every future must be settled, one way or the other"


def test_a_full_queue_sheds_load_when_asked():
    running = engine(
        runner=EchoStepwiseRunner(per_step_seconds=0.5), max_batch_size=1, queue_capacity=1
    )
    try:
        rejected = 0
        for index in range(12):
            try:
                running.submit(f"p{index}", block_when_full=False)
            except EngineClosedError:
                rejected += 1
        assert rejected > 0, "a queue of one must refuse something under this load"
    finally:
        running.shutdown(timeout=0.05)


# --- substitutability -------------------------------------------------------


def test_both_engines_satisfy_the_serving_protocol():
    """The reason this is a sibling class rather than a mode flag.

    A caller written against `ServingEngine` must accept either. Checked against
    the protocol rather than by comparing attribute names, so a method that
    exists but has drifted in signature is still caught by mypy, and a missing
    one is caught here.
    """
    config = EngineConfig(max_batch_size=2, generation=GenerationConfig(max_new_tokens=2))
    with ContinuousEngine(config=config) as continuous, InferenceEngine(config=config) as static:
        assert isinstance(continuous, ServingEngine)
        assert isinstance(static, ServingEngine)


def test_both_engines_answer_the_same_prompts():
    prompts = [f"prompt {index}" for index in range(6)]
    generation = GenerationConfig(max_new_tokens=3)
    config = EngineConfig(max_batch_size=4, generation=generation)

    with ContinuousEngine(config=config) as continuous:
        from_continuous = continuous.generate_many(prompts)
    with InferenceEngine(config=config) as static:
        from_static = static.generate_many(prompts)

    assert [r.generated_tokens for r in from_continuous] == [3] * 6
    assert all(r.ok for r in from_continuous)
    assert len(from_continuous) == len(from_static)


def test_the_snapshot_identifies_the_scheduler():
    with engine() as running:
        running.generate("hello")
        snapshot = running.snapshot()

    assert snapshot.extra["scheduler"] == "continuous"
    assert snapshot.extra["max_batch_size"] == 4
    assert "runner" in snapshot.extra

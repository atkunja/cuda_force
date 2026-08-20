"""Continuous batching tests.

The property under test is scheduling, not generation: that rows freed by a
finished sequence are refilled at the next step rather than at the end of the
batch. Everything runs against the deterministic step-wise runner, so the
assertions are stable and no model is involved.
"""

from __future__ import annotations

import threading
import time

import pytest

from cudaforge.config import GenerationConfig
from cudaforge.continuous import ContinuousBatcher, ContinuousStats, run_static
from cudaforge.metrics import MetricsRegistry
from cudaforge.scheduler import Request
from cudaforge.stepwise import EchoStepwiseRunner, SequenceState


class Collector:
    """Records completions so a test can assert after shutdown."""

    def __init__(self) -> None:
        self.done: list[tuple[Request, SequenceState]] = []
        self._lock = threading.Lock()
        self._target = 0
        self._reached = threading.Event()

    def expect(self, count: int) -> None:
        self._target = count

    def __call__(self, request: Request, state: SequenceState) -> None:
        with self._lock:
            self.done.append((request, state))
            if self._target and len(self.done) >= self._target:
                self._reached.set()

    def wait(self, timeout: float = 30.0) -> bool:
        return self._reached.wait(timeout)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self.done)


def submit_all(batcher: ContinuousBatcher, lengths: list[int]) -> None:
    for index, length in enumerate(lengths):
        batcher.submit(
            Request(prompt=f"p{index}", generation=GenerationConfig(max_new_tokens=length))
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_batch_size": 0}, "max_batch_size"),
        ({"max_batch_size": 32, "queue_capacity": 8}, "queue_capacity"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ContinuousBatcher(EchoStepwiseRunner(), lambda _r, _s: None, **kwargs)


def test_every_submitted_sequence_completes():
    collector = Collector()
    lengths = [3, 7, 2, 11, 5]
    collector.expect(len(lengths))

    with ContinuousBatcher(EchoStepwiseRunner(), collector, max_batch_size=4) as batcher:
        submit_all(batcher, lengths)
        assert collector.wait()

    assert collector.count == len(lengths)


def test_each_sequence_gets_the_tokens_it_asked_for():
    collector = Collector()
    lengths = [1, 4, 9, 16]
    collector.expect(len(lengths))

    with ContinuousBatcher(EchoStepwiseRunner(), collector, max_batch_size=4) as batcher:
        submit_all(batcher, lengths)
        assert collector.wait()

    by_prompt = {request.prompt: state for request, state in collector.done}
    for index, length in enumerate(lengths):
        assert by_prompt[f"p{index}"].generated == length


def test_results_are_paired_with_their_own_request():
    # A scheduling bug that swaps rows would deliver the wrong text to a caller,
    # and nothing about the output would look wrong.
    collector = Collector()
    collector.expect(12)

    with ContinuousBatcher(EchoStepwiseRunner(), collector, max_batch_size=4) as batcher:
        submit_all(batcher, [3] * 12)
        assert collector.wait()

    for request, state in collector.done:
        assert state.prompt == request.prompt
        assert state.text.startswith(request.prompt[:2])


def test_batch_width_never_exceeds_capacity():
    collector = Collector()
    collector.expect(40)
    runner = EchoStepwiseRunner()

    with ContinuousBatcher(runner, collector, max_batch_size=6, queue_capacity=128) as batcher:
        submit_all(batcher, [4] * 40)
        assert collector.wait()

    assert runner.step_widths
    assert max(runner.step_widths) <= 6


def test_a_freed_row_is_refilled_rather_than_left_idle():
    # The defining behaviour. One long sequence alongside many short ones: with
    # static batching the short rows idle until the long one finishes.
    collector = Collector()
    lengths = [40] + [2] * 20
    collector.expect(len(lengths))
    runner = EchoStepwiseRunner(per_step_seconds=0.0005)

    with ContinuousBatcher(runner, collector, max_batch_size=4, queue_capacity=64) as batcher:
        submit_all(batcher, lengths)
        assert collector.wait()

    # 40 + 20*2 = 80 tokens at width 4 is 20 steps ideally. Static batching
    # would take 40 steps for the first batch alone, since every row waits for
    # the 40-token member.
    assert runner.steps < 40


def test_continuous_beats_static_on_a_long_tailed_workload():
    # The comparison the scheduler exists to win. Both sides use the identical
    # runner, so the difference is attributable to scheduling alone.
    lengths = [2, 2, 2, 2, 2, 2, 2, 2, 60, 2, 2, 2, 2, 2, 2, 2]
    work = [
        (f"p{index}", GenerationConfig(max_new_tokens=length))
        for index, length in enumerate(lengths)
    ]

    static = run_static(EchoStepwiseRunner(), work, max_batch_size=8)

    collector = Collector()
    collector.expect(len(lengths))
    runner = EchoStepwiseRunner(per_step_seconds=0.0005)
    with ContinuousBatcher(runner, collector, max_batch_size=8, queue_capacity=64) as batcher:
        submit_all(batcher, lengths)
        assert collector.wait()
    continuous = batcher.stats()

    assert continuous.completions == static.completions == len(lengths)
    assert continuous.decode_steps < static.decode_steps
    assert continuous.utilisation > static.utilisation


def test_static_batching_holds_rows_for_the_longest_member():
    # Establishes the baseline the comparison above relies on, so a regression
    # in `run_static` cannot make continuous batching look good by default.
    work = [
        ("short", GenerationConfig(max_new_tokens=2)),
        ("long", GenerationConfig(max_new_tokens=30)),
    ]
    stats = run_static(EchoStepwiseRunner(), work, max_batch_size=2)

    # Every step is charged against both rows even after the short one is done.
    assert stats.decode_steps == 30
    assert stats.available_rows == 60
    assert stats.occupied_rows < 60


def test_utilisation_is_one_when_every_row_is_busy():
    stats = ContinuousStats(decode_steps=10, occupied_rows=40, available_rows=40)
    assert stats.utilisation == 1.0


def test_utilisation_of_an_idle_scheduler_is_zero():
    assert ContinuousStats().utilisation == 0.0


def test_a_sequence_that_stops_early_retires_before_its_budget():
    # An end-of-sequence marker finishes a sequence even though tokens remain in
    # its budget — which is exactly the row continuous batching reclaims.
    collector = Collector()
    collector.expect(2)
    runner = EchoStepwiseRunner(stop_after={"p0": 3})

    with ContinuousBatcher(runner, collector, max_batch_size=2) as batcher:
        submit_all(batcher, [50, 4])
        assert collector.wait()

    by_prompt = {request.prompt: state for request, state in collector.done}
    assert by_prompt["p0"].stopped_early
    assert by_prompt["p0"].generated == 3
    assert not by_prompt["p1"].stopped_early


def test_expired_requests_are_dropped_at_admission():
    collector = Collector()
    dropped: list[Request] = []
    metrics = MetricsRegistry()

    batcher = ContinuousBatcher(
        EchoStepwiseRunner(),
        collector,
        max_batch_size=4,
        metrics=metrics,
        on_expired=dropped.append,
    )
    past = time.monotonic() - 1
    for index in range(5):
        batcher.submit(
            Request(
                prompt=f"stale{index}", generation=GenerationConfig(max_new_tokens=2), deadline=past
            )
        )
    collector.expect(2)
    submit_all(batcher, [2, 2])
    assert collector.wait()
    batcher.shutdown()

    assert len(dropped) == 5
    assert metrics.snapshot().requests_expired == 5
    assert batcher.stats().expired == 5
    assert collector.count == 2


def test_shutdown_finishes_what_is_running():
    collector = Collector()
    batcher = ContinuousBatcher(EchoStepwiseRunner(), collector, max_batch_size=4)
    submit_all(batcher, [5] * 4)
    batcher.shutdown()

    # Sequences already admitted run to completion rather than being abandoned.
    assert collector.count == 4


def test_shutdown_is_idempotent():
    batcher = ContinuousBatcher(EchoStepwiseRunner(), Collector(), max_batch_size=2)
    batcher.shutdown()
    batcher.shutdown()
    assert not batcher.submit(Request(prompt="x"))


def test_a_failing_completion_callback_does_not_stall_the_scheduler():
    calls = []

    def explode(request: Request, state: SequenceState) -> None:
        calls.append(request.prompt)
        raise RuntimeError("callback failure")

    with ContinuousBatcher(EchoStepwiseRunner(), explode, max_batch_size=2) as batcher:
        submit_all(batcher, [2] * 6)
        deadline = time.monotonic() + 10
        while len(calls) < 6 and time.monotonic() < deadline:
            time.sleep(0.005)

    assert len(calls) == 6


def test_load_shedding_rejects_when_the_queue_is_full():
    collector = Collector()
    runner = EchoStepwiseRunner(per_step_seconds=0.01)
    rejected = 0

    batcher = ContinuousBatcher(runner, collector, max_batch_size=2, queue_capacity=2)
    for index in range(40):
        if not batcher.try_submit(
            Request(prompt=f"p{index}", generation=GenerationConfig(max_new_tokens=5))
        ):
            rejected += 1
    batcher.shutdown()

    assert rejected > 0


def test_metrics_record_queue_delay_for_admitted_sequences():
    metrics = MetricsRegistry()
    collector = Collector()
    collector.expect(6)

    with ContinuousBatcher(
        EchoStepwiseRunner(), collector, max_batch_size=3, metrics=metrics
    ) as batcher:
        submit_all(batcher, [3] * 6)
        assert collector.wait()

    assert metrics.queue_delay.count == 6


def test_stats_are_a_snapshot_not_a_live_view():
    collector = Collector()
    collector.expect(4)

    with ContinuousBatcher(EchoStepwiseRunner(), collector, max_batch_size=2) as batcher:
        submit_all(batcher, [3] * 4)
        assert collector.wait()
        first = batcher.stats()
        first.decode_steps = -1

    assert batcher.stats().decode_steps >= 0

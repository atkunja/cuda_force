"""Continuous batching: scheduling between decode steps rather than between batches.

## What static batching wastes

`DynamicBatcher` forms a batch and hands it to the runner, which returns when
every member is done. A batch asking for 8, 8, 8 and 256 tokens therefore
occupies four rows for 256 steps, three of them idle after step 8.

The waste is not a rounding error. Real traffic has a long tail of generation
lengths, so a batch's longest member is routinely many times its median, and
utilisation falls in proportion.

## What this does instead

The scheduler regains control after every decode step:

    while running or queued:
        admit whatever fits into free rows     ← the whole point
        decode_step(running)                   ← one token, whole batch
        retire finished sequences

A row freed at step 8 is filled at step 9, not at step 256. The batch is never
drained and refilled; it is continuously topped up.

## Why one token for the whole batch

`decode_step` advances *every* active sequence by one token, because a decode
step is a single forward pass and its cost is dominated by reading the weights
once. Advancing sequences individually would forfeit exactly the amortisation
batching exists for.

## What this does not do

No KV cache is attached here. `KVCacheManager` in the C++ runtime implements the
admission and preemption this scheduler would drive on real hardware, but the
Python path has no cache to page, so admission is bounded by `max_batch_size`
alone. Pretending otherwise would be the interesting half of the problem left
implicit.
"""

from __future__ import annotations

import contextlib
import itertools
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from cudaforge.config import GenerationConfig
from cudaforge.metrics import MetricsRegistry
from cudaforge.scheduler import Request
from cudaforge.stepwise import SequenceState, StepwiseRunner


@dataclass
class ContinuousStats:
    """What the scheduler did, in terms that make utilisation checkable."""

    decode_steps: int = 0
    #: Sum of batch widths across steps — the work actually done.
    occupied_rows: int = 0
    #: Steps times capacity — the work a full batch would have done.
    available_rows: int = 0
    admissions: int = 0
    completions: int = 0
    expired: int = 0
    max_observed_batch: int = 0

    @property
    def utilisation(self) -> float:
        """Fraction of available rows that held a live sequence.

        The number static batching loses and this recovers. 1.0 means every row
        was busy at every step.
        """
        if self.available_rows == 0:
            return 0.0
        return self.occupied_rows / self.available_rows


class ContinuousBatcher:
    """Iteration-level scheduler over a step-wise runner.

    One scheduler thread owns the running set, for the same reason the static
    batcher has one formation thread: admission is a decision about the batch as
    a whole, and two threads racing to fill the same rows would need a lock that
    serialises them anyway.
    """

    def __init__(
        self,
        runner: StepwiseRunner,
        on_complete: Callable[[Request, SequenceState], None],
        max_batch_size: int = 16,
        queue_capacity: int = 1024,
        idle_poll_seconds: float = 0.001,
        metrics: MetricsRegistry | None = None,
        on_expired: Callable[[Request], None] | None = None,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")
        if queue_capacity < max_batch_size:
            raise ValueError(
                f"queue_capacity ({queue_capacity}) must be at least "
                f"max_batch_size ({max_batch_size})"
            )

        self._runner = runner
        self._on_complete = on_complete
        self._on_expired = on_expired
        self._max_batch = max_batch_size
        self._idle_poll = idle_poll_seconds
        self._metrics = metrics or MetricsRegistry()

        self._queue: queue.Queue[Request | None] = queue.Queue(maxsize=queue_capacity)
        self._ids = itertools.count(1)
        self._running: list[tuple[Request, SequenceState]] = []
        self._stats = ContinuousStats()
        self._stats_lock = threading.Lock()
        self._stopping = threading.Event()

        self._thread = threading.Thread(target=self._run, name="cudaforge-continuous", daemon=True)
        self._thread.start()

    @property
    def metrics(self) -> MetricsRegistry:
        return self._metrics

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    def stats(self) -> ContinuousStats:
        with self._stats_lock:
            return ContinuousStats(**vars(self._stats))

    def submit(self, request: Request, timeout: float | None = None) -> bool:
        """Enqueue a request, blocking while the queue is full."""
        if self._stopping.is_set():
            self._metrics.record_rejected()
            return False

        self._metrics.record_received()
        try:
            self._queue.put(request, block=True, timeout=timeout)
        except queue.Full:
            self._metrics.record_rejected()
            return False
        self._metrics.set_queue_depth(self._queue.qsize())
        return True

    def try_submit(self, request: Request) -> bool:
        """Reject rather than block when the queue is full."""
        if self._stopping.is_set():
            self._metrics.record_rejected()
            return False

        self._metrics.record_received()
        try:
            self._queue.put_nowait(request)
        except queue.Full:
            self._metrics.record_rejected()
            return False
        self._metrics.set_queue_depth(self._queue.qsize())
        return True

    def shutdown(self, timeout: float = 30.0) -> None:
        """Stop admitting, finish what is running, and join. Idempotent."""
        if self._stopping.is_set():
            return
        self._stopping.set()
        with contextlib.suppress(queue.Full):
            self._queue.put(None, timeout=timeout)
        self._thread.join(timeout=timeout)

    def __enter__(self) -> ContinuousBatcher:
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()

    # -- scheduler thread ---------------------------------------------------

    def _admit(self) -> None:
        """Fill free rows from the queue. Never blocks when work is running."""
        admitted: list[SequenceState] = []

        while len(self._running) + len(admitted) < self._max_batch:
            try:
                # Blocking only when nothing is running: with a live batch the
                # scheduler must return to stepping it rather than waiting.
                if self._running or admitted:
                    request = self._queue.get_nowait()
                else:
                    request = self._queue.get(timeout=self._idle_poll)
            except queue.Empty:
                break

            if request is None:
                self._stopping.set()
                with contextlib.suppress(queue.Full):
                    self._queue.put_nowait(None)
                break

            request.dequeued_at = time.monotonic()
            if request.expired():
                # Same rule as the static batcher: work nobody is waiting for
                # displaces work someone is.
                self._metrics.record_expired()
                with self._stats_lock:
                    self._stats.expired += 1
                if self._on_expired is not None:
                    with contextlib.suppress(Exception):
                        self._on_expired(request)
                continue

            self._metrics.record_queue_delay(request.queue_delay)
            state = SequenceState(
                sequence_id=next(self._ids),
                prompt=request.prompt,
                generation=request.generation,
            )
            self._running.append((request, state))
            admitted.append(state)

        if admitted:
            self._runner.prefill(admitted)
            with self._stats_lock:
                self._stats.admissions += len(admitted)

    def _retire(self) -> None:
        """Hand back finished sequences and free their rows."""
        still_running: list[tuple[Request, SequenceState]] = []

        for request, state in self._running:
            if not state.finished:
                still_running.append((request, state))
                continue

            self._runner.evict(state.sequence_id)
            with self._stats_lock:
                self._stats.completions += 1
            try:
                self._on_complete(request, state)
            except Exception:  # noqa: BLE001
                # A failing completion callback must not stop the scheduler, or
                # every sequence behind it stalls until shutdown.
                self._metrics.record_failed()

        self._running = still_running

    def _run(self) -> None:
        while True:
            if not self._stopping.is_set():
                self._admit()

            if not self._running:
                if self._stopping.is_set():
                    return
                continue

            # One token for every live sequence, then back to the scheduler.
            # This is the whole difference from static batching.
            width = len(self._running)
            self._runner.decode_step([state for _, state in self._running])

            with self._stats_lock:
                self._stats.decode_steps += 1
                self._stats.occupied_rows += width
                self._stats.available_rows += self._max_batch
                self._stats.max_observed_batch = max(self._stats.max_observed_batch, width)

            self._retire()
            self._metrics.set_queue_depth(self._queue.qsize())


def run_static(
    runner: StepwiseRunner,
    prompts: list[tuple[str, GenerationConfig]],
    max_batch_size: int,
) -> ContinuousStats:
    """Run the same workload with static batching, for comparison.

    Forms full batches and runs each until *every* member finishes, which is
    what `DynamicBatcher` plus a one-shot runner does. Written here rather than
    measured through the real static path so both sides use the identical
    runner and the difference is attributable to scheduling alone.
    """
    stats = ContinuousStats()
    pending = list(prompts)

    while pending:
        batch = pending[:max_batch_size]
        pending = pending[max_batch_size:]

        states = [
            SequenceState(sequence_id=index, prompt=prompt, generation=generation)
            for index, (prompt, generation) in enumerate(batch)
        ]
        runner.prefill(states)
        stats.admissions += len(states)

        # The defining behaviour: the batch is held until its longest member is
        # done, so finished rows sit idle rather than being refilled.
        while any(not state.finished for state in states):
            width = sum(1 for state in states if not state.finished)
            runner.decode_step(states)
            stats.decode_steps += 1
            stats.occupied_rows += width
            stats.available_rows += max_batch_size
            stats.max_observed_batch = max(stats.max_observed_batch, width)

        for state in states:
            runner.evict(state.sequence_id)
        stats.completions += len(states)

    return stats

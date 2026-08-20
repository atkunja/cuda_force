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
from cudaforge.kv_cache import KVCacheManager
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
    #: Admissions the cache refused for want of blocks. Distinct from `expired`:
    #: the request is still wanted, it just cannot be started yet.
    cache_refusals: int = 0
    #: Requests that could never be served — larger than the whole cache.
    #: Distinct from `cache_refusals`, which counts transient pressure.
    rejected: int = 0
    #: Running sequences dropped to make room for another, or because they could
    #: not grow. Non-zero means the cache is the binding constraint, not the row
    #: count — which is the whole reason for wiring one in.
    preempted: int = 0

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
        on_rejected: Callable[[Request, str], None] | None = None,
        cache: KVCacheManager | None = None,
        estimate_tokens: Callable[[str], int] | None = None,
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
        #: Called when a request can never be served — distinct from expiry,
        #: which is about deadlines, and from completion, which produced output.
        self._on_rejected = on_rejected
        self._max_batch = max_batch_size
        self._idle_poll = idle_poll_seconds
        self._metrics = metrics or MetricsRegistry()

        #: When present, admission is bounded by cache capacity rather than by
        #: `max_batch_size` alone. Rows are the wrong unit: four sequences of
        #: sixteen tokens and four of four thousand occupy the same number of
        #: rows and wildly different amounts of memory, and only the second set
        #: can exhaust the device.
        self._cache = cache
        #: How many tokens a prompt will occupy. Admission has to decide before
        #: the runner has tokenised anything, so this is necessarily an estimate;
        #: the default counts whitespace-separated words, which is close enough
        #: for a capacity decision and wrong enough to be worth naming.
        self._estimate_tokens = estimate_tokens or (lambda prompt: max(1, len(prompt.split())))
        #: A request dequeued but refused by the cache. Held rather than
        #: rejected: the blocks it needs are about to be freed by whatever
        #: finishes next, and dropping it would turn transient pressure into a
        #: failed request.
        self._deferred: Request | None = None

        self._queue: queue.Queue[Request | None] = queue.Queue(maxsize=queue_capacity)
        self._ids = itertools.count(1)
        self._running: list[tuple[Request, SequenceState]] = []
        self._stats = ContinuousStats()
        self._stats_lock = threading.Lock()
        self._stopping = threading.Event()
        #: Set once the shutdown sentinel has been seen, meaning no further
        #: request can arrive. Distinct from `_stopping`, which only stops new
        #: submissions being accepted.
        self._closed = False

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
        # Annotated because the queue yields the shutdown sentinel as None while
        # the deferred slot never holds one; without this the first assignment
        # would fix the narrower type.
        request: Request | None

        while len(self._running) + len(admitted) < self._max_batch:
            # A request the cache refused last pass gets first refusal now,
            # ahead of the queue, so pressure does not reorder the stream.
            if self._deferred is not None:
                deferred = self._deferred
                self._deferred = None
                if not self._offer_to_cache(deferred, admitted):
                    break
                continue

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
                # The sentinel is the last thing shutdown() enqueues, and
                # submit() refuses everything afterwards, so nothing can arrive
                # behind it. It is consumed rather than re-posted: leaving it in
                # the queue would make "drained" untestable, and re-posting it
                # each pass spins.
                self._stopping.set()
                self._closed = True
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
            if not self._offer_to_cache(request, admitted):
                break

        if admitted:
            self._runner.prefill(admitted)
            with self._stats_lock:
                self._stats.admissions += len(admitted)

    def _reserve_step(self) -> None:
        """One more token of cache for every running sequence.

        Decoding appends a token per sequence per step, so the cache grows by
        the batch width every step. Most of those extensions cost nothing — they
        land in the slack of a block already held — which is exactly why a paged
        cache can run at high occupancy.
        """
        if self._cache is None:
            return

        casualties: list[int] = []
        for _, state in list(self._running):
            outcome = self._cache.extend(state.sequence_id, 1)
            if not outcome.ok:
                # Cannot grow even after eviction. The sequence is dropped with
                # whatever it has produced rather than left to spin.
                casualties.append(state.sequence_id)
            else:
                casualties.extend(outcome.preempted)

        if casualties:
            # Deduplicated: a sequence can be both evicted for someone else and
            # unable to grow itself within one pass.
            self._evict_preempted(list(dict.fromkeys(casualties)))

    def _offer_to_cache(self, request: Request, admitted: list[SequenceState]) -> bool:
        """Reserve cache for a request and start it, or defer it.

        Returns False when the cache refused, in which case the request is held
        in `_deferred` and admission stops for this pass — trying the next
        request instead would let short prompts starve a long one indefinitely.
        """
        state = SequenceState(
            sequence_id=next(self._ids),
            prompt=request.prompt,
            generation=request.generation,
        )

        if self._cache is not None:
            outcome = self._cache.admit(state.sequence_id, self._estimate_tokens(request.prompt))
            if not outcome.ok:
                with self._stats_lock:
                    self._stats.cache_refusals += 1

                if not self._running and not admitted:
                    # Nothing is running, so nothing will free blocks: waiting
                    # cannot help and deferring would spin the scheduler until
                    # shutdown. Rejecting is the honest answer — the request is
                    # bigger than the cache, not merely unlucky in its timing.
                    self._reject(
                        request,
                        f"prompt needs more KV cache than the "
                        f"{self._cache.total_blocks}-block pool holds",
                    )
                    return True

                self._deferred = request
                return False
            if outcome.preempted:
                # The cache evicted running sequences to fit this one. They are
                # retired rather than resumed: resuming means recomputing their
                # prompts, which needs a runner that can be handed a partial
                # sequence — see the gap noted in continuous-batching.md.
                self._evict_preempted(outcome.preempted)

        self._running.append((request, state))
        admitted.append(state)
        return True

    def _reject(self, request: Request, reason: str) -> None:
        """Settle a request that can never be served."""
        with self._stats_lock:
            self._stats.rejected += 1
        if self._on_rejected is not None:
            with contextlib.suppress(Exception):
                self._on_rejected(request, reason)
            return
        # Without a rejection callback the request would otherwise vanish and
        # its caller wait forever, so it is completed with nothing.
        with contextlib.suppress(Exception):
            self._on_complete(
                request,
                SequenceState(
                    sequence_id=next(self._ids),
                    prompt=request.prompt,
                    generation=request.generation,
                ),
            )

    def _evict_preempted(self, preempted: list[int]) -> None:
        """Drop sequences the cache evicted to make room for another."""
        survivors: list[tuple[Request, SequenceState]] = []
        for request, state in self._running:
            if state.sequence_id not in preempted:
                survivors.append((request, state))
                continue
            self._runner.evict(state.sequence_id)
            if self._cache is not None:
                self._cache.release(state.sequence_id)
            with self._stats_lock:
                self._stats.preempted += 1
            try:
                self._on_complete(request, state)
            except Exception:  # noqa: BLE001
                self._metrics.record_failed()
        self._running = survivors

    def _retire(self) -> None:
        """Hand back finished sequences and free their rows."""
        still_running: list[tuple[Request, SequenceState]] = []

        for request, state in self._running:
            if not state.finished:
                still_running.append((request, state))
                continue

            self._runner.evict(state.sequence_id)
            if self._cache is not None:
                self._cache.release(state.sequence_id)
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
            # Admission continues after shutdown so that work already accepted
            # is not abandoned. `submit` stops accepting new work at that point,
            # so this drains a bounded backlog rather than running forever.
            self._admit()

            if not self._running:
                if self._closed and self._queue.empty():
                    return
                continue

            # One token for every live sequence, then back to the scheduler.
            # This is the whole difference from static batching.
            # Cache first: a step that cannot be paid for must not be taken.
            self._reserve_step()
            if not self._running:
                continue

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

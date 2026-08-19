"""Dynamic batching for concurrently submitted requests.

Mirrors the C++ ``DynamicBatcher``. Both exist because they serve different
callers: the C++ one is what a native serving binary uses, this one is what the
Python inference engine uses, and keeping the semantics identical means the
behaviour documented in ``docs/concurrency.md`` describes both.
"""

from __future__ import annotations

import itertools
import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from cudaforge.config import GenerationConfig
from cudaforge.metrics import MetricsRegistry


class BatchTrigger(Enum):
    """Why the batcher stopped accumulating.

    Worth recording: a batcher that only ever closes on ``TIMEOUT`` is starved
    and its ``max_wait`` is pure added latency, while one that only ever closes
    on ``MAX_SIZE`` is saturated and a larger batch would help. Batch size alone
    does not distinguish the two.
    """

    MAX_SIZE = "max_size"
    TIMEOUT = "timeout"
    SHUTDOWN = "shutdown"


@dataclass
class Request:
    """A unit of work moving through queue to batcher to executor."""

    prompt: str
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    enqueued_at: float = field(default_factory=time.monotonic)
    dequeued_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def queue_delay(self) -> float:
        """Seconds spent waiting to be batched. 0.0 until dequeued."""
        if self.dequeued_at is None:
            return 0.0
        return self.dequeued_at - self.enqueued_at


@dataclass
class Batch:
    """Requests the batcher decided to execute together."""

    requests: list[Request]
    trigger: BatchTrigger
    formed_at: float = field(default_factory=time.monotonic)

    def __len__(self) -> int:
        return len(self.requests)

    @property
    def prompts(self) -> list[str]:
        return [request.prompt for request in self.requests]

    @property
    def max_new_tokens(self) -> int:
        """Longest generation any member asked for.

        A batch runs for as long as its most demanding member, so this is what
        the executor must budget. Members that finish earlier are masked out
        rather than shortening the batch.
        """
        return max(request.generation.max_new_tokens for request in self.requests)


class DynamicBatcher:
    """Aggregates concurrent requests into batches on a background thread.

    A batch closes on whichever comes first:

    1. it holds ``max_batch_size`` requests, or
    2. the **oldest** request in it has waited ``max_wait_seconds``.

    The deadline is anchored to the oldest request and never extended. If it
    were reset on each arrival, a steady stream of requests would postpone
    execution indefinitely, and no request would have a bounded wait. As written,
    nothing waits longer than ``max_wait_seconds`` plus the batch's own service
    time.

    Batch formation runs on a single thread because it is an inherently serial
    decision: two threads competing to claim requests from one queue would need
    a lock that serialises them anyway, and would make the deadline
    non-deterministic. Parallelism belongs downstream, in the handler.
    """

    def __init__(
        self,
        handler: Callable[[Batch], None],
        max_batch_size: int = 16,
        max_wait_seconds: float = 0.005,
        queue_capacity: int = 1024,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")
        if max_wait_seconds < 0:
            raise ValueError(f"max_wait_seconds must not be negative, got {max_wait_seconds}")
        if queue_capacity < max_batch_size:
            raise ValueError(
                f"queue_capacity ({queue_capacity}) must be at least "
                f"max_batch_size ({max_batch_size})"
            )

        self._handler = handler
        self._max_batch_size = max_batch_size
        self._max_wait = max_wait_seconds
        self._metrics = metrics or MetricsRegistry()

        # Bounded on purpose. An unbounded queue converts overload into
        # unbounded latency and eventual memory exhaustion; a bounded one
        # applies backpressure to the submitter, where it can be observed.
        self._queue: queue.Queue[Request | None] = queue.Queue(maxsize=queue_capacity)
        self._stopping = threading.Event()
        self._counter = itertools.count()
        self._thread = threading.Thread(
            target=self._run, name="cudaforge-batcher", daemon=True
        )
        self._thread.start()

    @property
    def metrics(self) -> MetricsRegistry:
        return self._metrics

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    def submit(self, request: Request, timeout: float | None = None) -> bool:
        """Enqueue a request, blocking while the queue is full.

        Returns False if the batcher is shutting down or the wait expired.
        """
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
        """Reject rather than block when the queue is full.

        This is the load-shedding path: an ingress that would rather return a
        fast 503 than let the caller wait behind a full queue.
        """
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
        """Stop accepting work, drain what is queued, and join. Idempotent."""
        if self._stopping.is_set():
            return
        self._stopping.set()
        # The sentinel unblocks the collector if it is parked on an empty queue.
        self._queue.put(None)
        self._thread.join(timeout=timeout)

    def __enter__(self) -> DynamicBatcher:
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()

    def _run(self) -> None:
        while True:
            batch = self._collect()
            if batch is None:
                return

            self._metrics.record_batch(
                len(batch), closed_by_timeout=batch.trigger is BatchTrigger.TIMEOUT
            )
            self._metrics.set_queue_depth(self._queue.qsize())
            for request in batch.requests:
                self._metrics.record_queue_delay(request.queue_delay)

            try:
                self._handler(batch)
            except Exception:  # noqa: BLE001 - a failing handler must not stop the batcher
                # Otherwise one bad batch stalls every subsequent request until
                # shutdown. The failure is counted and the loop continues.
                self._metrics.record_failed()

    def _collect(self) -> Batch | None:
        """Block for the first request, then fill until size or deadline.

        Returns None once the queue is closed and drained.
        """
        first = self._queue.get()
        if first is None:
            return None
        first.dequeued_at = time.monotonic()

        requests = [first]
        # Anchored to the first request. This is what bounds queue delay at
        # max_wait regardless of the arrival rate.
        deadline = first.dequeued_at + self._max_wait
        trigger = BatchTrigger.MAX_SIZE

        while len(requests) < self._max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                trigger = BatchTrigger.TIMEOUT
                break
            try:
                nxt = self._queue.get(timeout=remaining)
            except queue.Empty:
                trigger = BatchTrigger.TIMEOUT
                break

            if nxt is None:
                # Shutdown sentinel: drain whatever is still buffered rather
                # than dropping it, then let the next _collect see the sentinel.
                trigger = BatchTrigger.SHUTDOWN
                while len(requests) < self._max_batch_size:
                    try:
                        pending = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if pending is None:
                        break
                    pending.dequeued_at = time.monotonic()
                    requests.append(pending)
                self._queue.put(None)  # keep the sentinel for the next iteration
                break

            nxt.dequeued_at = time.monotonic()
            requests.append(nxt)

        return Batch(requests=requests, trigger=trigger)

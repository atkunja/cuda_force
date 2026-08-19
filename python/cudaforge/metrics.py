"""Counters and latency percentiles for the Python runtime.

Mirrors ``cpp/include/cudaforge/metrics.hpp`` closely enough that the two
report the same field names, so a dashboard does not need to know which layer
produced a snapshot.
"""

from __future__ import annotations

import bisect
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any


class LatencyHistogram:
    """Latency samples with exact percentiles over a bounded window.

    The C++ side uses log-linear buckets because it must be O(1) in memory
    under sustained load. Python's per-request overhead is already far above
    the cost of an insert, so this keeps a sorted reservoir of the most recent
    ``capacity`` samples and reports exact percentiles over that window.

    Keeping the *most recent* samples rather than a uniform reservoir is
    deliberate: for a serving system, the question is almost always "what is
    p99 right now", and old samples from a different load regime actively
    mislead.
    """

    def __init__(self, capacity: int = 100_000) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._capacity = capacity
        self._samples: list[float] = []
        # Eviction order. A deque rather than a list because popping the front
        # of a list is O(n), which would make recording O(n) once the window is
        # full — on the per-request path.
        self._order: deque[float] = deque()
        self._lock = threading.Lock()
        self._count = 0
        self._total = 0.0

    def record(self, value_seconds: float) -> None:
        with self._lock:
            self._count += 1
            self._total += value_seconds
            if len(self._samples) >= self._capacity:
                oldest = self._order.popleft()
                index = bisect.bisect_left(self._samples, oldest)
                if index < len(self._samples) and self._samples[index] == oldest:
                    self._samples.pop(index)
            bisect.insort(self._samples, value_seconds)
            self._order.append(value_seconds)

    def percentile(self, quantile: float) -> float:
        """``quantile`` in [0, 1]. Returns 0.0 when nothing has been recorded."""
        with self._lock:
            if not self._samples:
                return 0.0
            clamped = min(max(quantile, 0.0), 1.0)
            index = int(clamped * (len(self._samples) - 1))
            return self._samples[index]

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    @property
    def mean(self) -> float:
        with self._lock:
            return self._total / self._count if self._count else 0.0

    @property
    def maximum(self) -> float:
        with self._lock:
            return self._samples[-1] if self._samples else 0.0

    @property
    def minimum(self) -> float:
        with self._lock:
            return self._samples[0] if self._samples else 0.0

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._order.clear()
            self._count = 0
            self._total = 0.0


@dataclass
class MetricsSnapshot:
    """Point-in-time view. Field names match the C++ snapshot."""

    requests_received: int = 0
    requests_completed: int = 0
    requests_failed: int = 0
    requests_rejected: int = 0
    requests_expired: int = 0

    batches_processed: int = 0
    batched_requests: int = 0
    batches_closed_by_size: int = 0
    batches_closed_by_timeout: int = 0

    tokens_generated: int = 0
    queue_depth: int = 0

    average_batch_size: float = 0.0
    uptime_seconds: float = 0.0
    requests_per_second: float = 0.0
    tokens_per_second: float = 0.0

    queue_delay_p50_ms: float = 0.0
    queue_delay_p95_ms: float = 0.0
    queue_delay_p99_ms: float = 0.0

    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_max_ms: float = 0.0
    latency_mean_ms: float = 0.0

    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MetricsRegistry:
    """Thread-safe counters plus latency and queue-delay histograms.

    One lock covers the counters. Unlike the C++ side — where relaxed atomics
    avoid contention on a genuinely hot path — Python's GIL means a lock here
    costs almost nothing relative to the interpreter overhead already present,
    and it buys a consistent snapshot across every counter.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start = time.monotonic()

        self._received = 0
        self._completed = 0
        self._failed = 0
        self._rejected = 0
        self._expired = 0
        self._batches = 0
        self._batched_requests = 0
        self._size_closures = 0
        self._timeout_closures = 0
        self._tokens = 0
        self._queue_depth = 0

        self.latency = LatencyHistogram()
        self.queue_delay = LatencyHistogram()

    def record_received(self) -> None:
        with self._lock:
            self._received += 1

    def record_rejected(self) -> None:
        with self._lock:
            self._rejected += 1

    def record_failed(self) -> None:
        with self._lock:
            self._failed += 1

    def record_expired(self) -> None:
        """A request dropped at dequeue because its deadline had passed.

        Counted separately from `rejected` (refused at admission) and `failed`
        (execution error): a rising expiry count means the queue is deeper than
        clients are willing to wait for, which calls for shedding earlier rather
        than for more capacity.
        """
        with self._lock:
            self._expired += 1

    def record_queue_delay(self, seconds: float) -> None:
        self.queue_delay.record(seconds)

    def record_completion(self, latency_seconds: float, tokens: int) -> None:
        self.latency.record(latency_seconds)
        with self._lock:
            self._completed += 1
            self._tokens += tokens

    def record_batch(self, size: int, closed_by_timeout: bool) -> None:
        """Record a batch at the moment it was *formed*.

        `batched_requests` and `average_batch_size` therefore describe batch
        formation, not execution. A request can be counted here and then dropped
        before the runner sees it, if its deadline passes while the batch waits
        behind the worker pool — so under heavy expiry, `average_batch_size`
        slightly overstates what actually ran.

        Correcting it would mean adjusting a counter backwards from a worker
        thread, which buys precision in a number that is already only a guide
        and costs a synchronisation point on the execution path.
        `requests_expired` is the figure to read alongside it.
        """
        with self._lock:
            self._batches += 1
            self._batched_requests += size
            if closed_by_timeout:
                self._timeout_closures += 1
            else:
                self._size_closures += 1

    def set_queue_depth(self, depth: int) -> None:
        with self._lock:
            self._queue_depth = depth

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            elapsed = max(time.monotonic() - self._start, 1e-9)
            batches = self._batches
            batched = self._batched_requests
            completed = self._completed
            tokens = self._tokens

            snapshot = MetricsSnapshot(
                requests_received=self._received,
                requests_completed=completed,
                requests_failed=self._failed,
                requests_rejected=self._rejected,
                requests_expired=self._expired,
                batches_processed=batches,
                batched_requests=batched,
                batches_closed_by_size=self._size_closures,
                batches_closed_by_timeout=self._timeout_closures,
                tokens_generated=tokens,
                queue_depth=self._queue_depth,
                average_batch_size=batched / batches if batches else 0.0,
                uptime_seconds=elapsed,
                requests_per_second=completed / elapsed,
                tokens_per_second=tokens / elapsed,
            )

        snapshot.queue_delay_p50_ms = self.queue_delay.percentile(0.50) * 1e3
        snapshot.queue_delay_p95_ms = self.queue_delay.percentile(0.95) * 1e3
        snapshot.queue_delay_p99_ms = self.queue_delay.percentile(0.99) * 1e3
        snapshot.latency_p50_ms = self.latency.percentile(0.50) * 1e3
        snapshot.latency_p95_ms = self.latency.percentile(0.95) * 1e3
        snapshot.latency_p99_ms = self.latency.percentile(0.99) * 1e3
        snapshot.latency_max_ms = self.latency.maximum * 1e3
        snapshot.latency_mean_ms = self.latency.mean * 1e3
        return snapshot

    def reset(self) -> None:
        with self._lock:
            self._start = time.monotonic()
            self._received = 0
            self._completed = 0
            self._failed = 0
            self._rejected = 0
            self._expired = 0
            self._batches = 0
            self._batched_requests = 0
            self._size_closures = 0
            self._timeout_closures = 0
            self._tokens = 0
            self._queue_depth = 0
        self.latency.reset()
        self.queue_delay.reset()

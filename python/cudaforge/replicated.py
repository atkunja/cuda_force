"""Serving across several GPUs, by replication.

Training already scales out through DDP. Serving did not scale at all: one
engine, one device, and a second GPU sitting idle.

## Replication, not tensor parallelism

This runs an independent engine per device and routes each request to one of
them. That is **data parallelism** — every replica holds the whole model — and
it is the right first step for exactly one reason: it multiplies throughput
without touching the model at all.

What it does not do is let you serve a model larger than one GPU. That needs the
weights split across devices, which means a collective on the critical path of
every layer and a rewrite of the operator layer to match. Naming the limit
matters more than working around it: a reader who needs 70B on 2x24GB is not
served by anything here.

## Routing

Requests go to the replica with the shallowest queue, ties broken by rotation.
Queue depth is the right signal because it measures what the caller waits for,
where round-robin alone sends work to a replica already backed up behind a long
generation. The measurement is a snapshot and the decision immediately stale,
which is fine: being approximately right continuously beats being exactly right
once.
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future

from cudaforge.config import EngineConfig, GenerationConfig
from cudaforge.engine import EngineClosedError, Response, ServingEngine
from cudaforge.metrics import MetricsRegistry, MetricsSnapshot


class ReplicatedEngine:
    """One engine per device, with load-aware routing.

    Satisfies `ServingEngine`, so the HTTP server and the load driver take it
    exactly as they take a single-device engine.
    """

    def __init__(
        self,
        replicas: Sequence[ServingEngine],
        config: EngineConfig | None = None,
    ) -> None:
        if not replicas:
            raise ValueError("ReplicatedEngine needs at least one replica")
        self._replicas = list(replicas)
        self._config = config or self._replicas[0].config
        self._rotation = itertools.cycle(range(len(self._replicas)))
        self._lock = threading.Lock()
        self._closed = threading.Event()
        #: Requests routed to each replica, for checking the routing is not
        #: quietly degenerate — every request landing on replica 0 would still
        #: pass every functional test here.
        self._routed = [0] * len(self._replicas)

    @classmethod
    def across_devices(
        cls,
        devices: Sequence[str | int],
        build: Callable[[str], ServingEngine],
        config: EngineConfig | None = None,
    ) -> ReplicatedEngine:
        """Build one replica per device.

        `build` is passed a device string and returns an engine. Injected rather
        than assumed so the caller chooses the scheduler, the runner and the
        model, and so this is testable without a GPU.
        """
        return cls([build(f"cuda:{d}" if isinstance(d, int) else d) for d in devices], config)

    # -- ServingEngine ------------------------------------------------------

    @property
    def config(self) -> EngineConfig:
        return self._config

    @property
    def metrics(self) -> MetricsRegistry:
        """The first replica's registry.

        Aggregating counters across replicas into a synthetic registry would
        misreport percentiles — merging two latency histograms is not the
        histogram of the merged samples. `snapshot()` sums what can be summed
        and says so.
        """
        return self._replicas[0].metrics

    @property
    def queue_depth(self) -> int:
        return sum(replica.queue_depth for replica in self._replicas)

    @property
    def replica_count(self) -> int:
        return len(self._replicas)

    def routed_counts(self) -> list[int]:
        """Requests sent to each replica so far."""
        with self._lock:
            return list(self._routed)

    def submit(
        self,
        prompt: str,
        generation: GenerationConfig | None = None,
        block_when_full: bool = True,
        deadline_seconds: float | None = None,
    ) -> Future[Response]:
        if self._closed.is_set():
            raise EngineClosedError("engine is shut down")

        index = self._choose()
        with self._lock:
            self._routed[index] += 1
        return self._replicas[index].submit(
            prompt,
            generation,
            block_when_full=block_when_full,
            deadline_seconds=deadline_seconds,
        )

    def generate(
        self, prompt: str, generation: GenerationConfig | None = None, timeout: float = 60.0
    ) -> Response:
        return self.submit(prompt, generation).result(timeout=timeout)

    def generate_many(
        self,
        prompts: list[str],
        generation: GenerationConfig | None = None,
        timeout: float = 120.0,
    ) -> list[Response]:
        futures = [self.submit(prompt, generation) for prompt in prompts]
        return [future.result(timeout=timeout) for future in futures]

    def snapshot(self) -> MetricsSnapshot:
        """Counters summed across replicas; percentiles from the first.

        Latency percentiles are *not* aggregated. Two histograms cannot be
        merged into the histogram of their combined samples, and averaging
        percentiles produces a number that describes no request that ever ran.
        The first replica's are reported, and `replicas` in `extra` says how many
        were left out.
        """
        snapshots = [replica.snapshot() for replica in self._replicas]
        combined = snapshots[0]

        for field in (
            "requests_completed",
            "requests_failed",
            "requests_expired",
            "batches_processed",
        ):
            if hasattr(combined, field):
                setattr(combined, field, sum(getattr(s, field, 0) for s in snapshots))

        combined.queue_depth = self.queue_depth
        combined.extra["replicas"] = len(self._replicas)
        combined.extra["routing"] = "least-queue-depth"
        combined.extra["parallelism"] = "data"
        combined.extra["percentiles_from"] = "replica 0 only; histograms do not merge"
        return combined

    def shutdown(self, timeout: float = 30.0) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        # Every replica is shut down even if one raises: leaving the others
        # running would strand their threads and their futures.
        errors: list[BaseException] = []
        for replica in self._replicas:
            try:
                replica.shutdown(timeout=timeout)
            except BaseException as error:  # noqa: BLE001
                errors.append(error)
        if errors:
            raise errors[0]

    def __enter__(self) -> ReplicatedEngine:
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()

    # -- routing ------------------------------------------------------------

    def _choose(self) -> int:
        """The replica with the shallowest queue, ties broken by rotation.

        The rotation matters: with every queue at zero — the common case at low
        load — a plain `min` would return index 0 every time and the other
        devices would never see a request.
        """
        depths = [replica.queue_depth for replica in self._replicas]
        shallowest = min(depths)
        candidates = [index for index, depth in enumerate(depths) if depth == shallowest]
        if len(candidates) == 1:
            return candidates[0]
        with self._lock:
            for _ in range(len(self._replicas)):
                index = next(self._rotation)
                if index in candidates:
                    return index
        return candidates[0]


__all__ = ["ReplicatedEngine"]

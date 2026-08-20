"""A serving engine scheduled at iteration level rather than by batch.

`InferenceEngine` forms a batch, runs it to completion, and forms the next one.
That is the right shape for a one-shot runner, and it is what `ModelRunner`
expresses: hand it prompts, get text back.

Continuous batching cannot be expressed that way. Refilling a row the moment a
sequence finishes means the engine has to see individual decode steps, which is
a different protocol — `StepwiseRunner` rather than `ModelRunner`. Rather than
give `InferenceEngine` a mode flag whose two halves accept incompatible runners,
this is a sibling that presents the same public surface over the other
scheduler. Callers that only use `submit`/`generate`/`shutdown` can swap one for
the other.

## What differs from `InferenceEngine`

`Response.batch_size` is always 0. Under continuous batching a request has no
single batch size — it shares a changing set of rows for its whole life, and
reporting any one number would be picking a moment arbitrarily. Occupancy is a
property of the schedule, not of a request, and `stats()` reports it.

`Response.prompt_tokens` is whatever the runner filled in, and is 0 for runners
without a tokeniser. `EchoStepwiseRunner` is one of those.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future

from cudaforge.config import EngineConfig, GenerationConfig
from cudaforge.continuous import ContinuousBatcher, ContinuousStats
from cudaforge.engine import EngineClosedError, Response
from cudaforge.metrics import MetricsRegistry, MetricsSnapshot
from cudaforge.scheduler import Request, RequestExpired
from cudaforge.stepwise import EchoStepwiseRunner, SequenceState, StepwiseRunner

_LOG = logging.getLogger(__name__)


class ContinuousEngine:
    """Concurrent inference with iteration-level scheduling.

    Thread-safe: `submit` may be called from any number of threads.
    """

    def __init__(
        self,
        config: EngineConfig | None = None,
        runner: StepwiseRunner | None = None,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self._config = config or EngineConfig()
        self._metrics = metrics or MetricsRegistry()
        self._runner = runner or EchoStepwiseRunner()

        self._futures: dict[str, Future[Response]] = {}
        # Guards _futures only, and is never held across a callback — doing so
        # would serialise every submission behind the scheduler thread.
        self._futures_lock = threading.Lock()
        self._closed = threading.Event()

        self._batcher = ContinuousBatcher(
            self._runner,
            on_complete=self._finish,
            max_batch_size=self._config.max_batch_size,
            queue_capacity=self._config.queue_capacity,
            metrics=self._metrics,
            on_expired=self._expire,
        )

        _LOG.info(
            "continuous engine ready: %s | %s",
            self._config.describe(),
            self._runner.description,
        )

    @property
    def config(self) -> EngineConfig:
        return self._config

    @property
    def metrics(self) -> MetricsRegistry:
        return self._metrics

    @property
    def queue_depth(self) -> int:
        return self._batcher.queue_depth

    def stats(self) -> ContinuousStats:
        """Scheduling counters: decode steps, admissions, and row occupancy."""
        return self._batcher.stats()

    def submit(
        self,
        prompt: str,
        generation: GenerationConfig | None = None,
        block_when_full: bool = True,
        deadline_seconds: float | None = None,
    ) -> Future[Response]:
        """Enqueue a prompt and return a future for its response.

        Same contract as `InferenceEngine.submit`, including the meaning of
        `block_when_full` and `deadline_seconds`.
        """
        if self._closed.is_set():
            raise EngineClosedError("engine is shut down")

        limit = self._config.max_prompt_chars
        if limit and len(prompt) > limit:
            raise ValueError(f"prompt of {len(prompt)} characters exceeds the limit of {limit}")

        request = Request(
            prompt=prompt,
            generation=generation or self._config.generation,
            deadline=(
                time.monotonic() + deadline_seconds if deadline_seconds is not None else None
            ),
        )
        future: Future[Response] = Future()

        # Registered before submission: the scheduler thread can admit the
        # request the instant it is enqueued, and would find no future to settle.
        with self._futures_lock:
            self._futures[request.request_id] = future

        accepted = (
            self._batcher.submit(request) if block_when_full else self._batcher.try_submit(request)
        )
        if not accepted:
            with self._futures_lock:
                self._futures.pop(request.request_id, None)
            raise EngineClosedError("request rejected: queue full or engine shutting down")

        return future

    def generate(
        self, prompt: str, generation: GenerationConfig | None = None, timeout: float = 60.0
    ) -> Response:
        """Blocking convenience wrapper around :meth:`submit`."""
        return self.submit(prompt, generation).result(timeout=timeout)

    def generate_many(
        self,
        prompts: list[str],
        generation: GenerationConfig | None = None,
        timeout: float = 120.0,
    ) -> list[Response]:
        """Submit every prompt, then collect.

        Submitting all of them first is what gives the scheduler a pool to draw
        from. Under continuous batching it matters more than it does under
        static batching: rows freed by short sequences are refilled from exactly
        this backlog, and an empty queue means they sit idle instead.
        """
        futures = [self.submit(prompt, generation) for prompt in prompts]
        return [future.result(timeout=timeout) for future in futures]

    def snapshot(self) -> MetricsSnapshot:
        snapshot = self._metrics.snapshot()
        snapshot.queue_depth = self.queue_depth
        snapshot.extra["model"] = self._config.model_name
        snapshot.extra["device"] = str(self._config.resolve_device())
        snapshot.extra["runner"] = self._runner.description
        snapshot.extra["max_batch_size"] = self._config.max_batch_size
        # Named so a dashboard can tell the two schedulers apart, which matters
        # when comparing them on the same traffic.
        snapshot.extra["scheduler"] = "continuous"
        return snapshot

    def shutdown(self, timeout: float = 30.0) -> None:
        """Drain in-flight work and release resources. Idempotent."""
        if self._closed.is_set():
            return
        self._closed.set()

        # The batcher keeps admitting from its backlog after this, so accepted
        # work still runs; `submit` is what stops taking new work.
        self._batcher.shutdown(timeout=timeout)

        # Anything still outstanding never reached the scheduler. Failing the
        # futures is essential: a caller blocked on result() would otherwise
        # wait out its own timeout with nothing explaining why.
        with self._futures_lock:
            outstanding = list(self._futures.items())
            self._futures.clear()
        for request_id, future in outstanding:
            if not future.done():
                future.set_exception(
                    EngineClosedError(f"request {request_id} abandoned at shutdown")
                )

    def __enter__(self) -> ContinuousEngine:
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()

    def _finish(self, request: Request, state: SequenceState) -> None:
        """Settle one sequence's future. Runs on the scheduler thread."""
        # Admission stamps `dequeued_at`, so this is time spent generating rather
        # than time spent queued — the same split `InferenceEngine` reports.
        started = request.dequeued_at or request.enqueued_at
        response = Response(
            request_id=request.request_id,
            text=state.text,
            prompt_tokens=state.prompt_tokens,
            generated_tokens=state.generated,
            queue_time=request.queue_delay,
            inference_time=max(0.0, time.monotonic() - started),
            # Deliberately 0: see the module docstring. A request under
            # continuous batching does not have one batch size.
            batch_size=0,
            metadata={
                "scheduler": "continuous",
                "stopped_early": state.stopped_early,
            },
        )
        self._metrics.record_completion(response.total_latency, state.generated)
        self._settle(request.request_id, response)

    def _expire(self, request: Request) -> None:
        """Settle the future of a request dropped for missing its deadline."""
        self._settle(
            request.request_id,
            Response(
                request_id=request.request_id,
                text="",
                queue_time=request.queue_delay,
                batch_size=0,
                error=(
                    f"{RequestExpired.__name__}: dropped after "
                    f"{request.queue_delay * 1e3:.1f} ms in the queue, past its deadline"
                ),
            ),
        )

    def _settle(self, request_id: str, response: Response) -> None:
        with self._futures_lock:
            future = self._futures.pop(request_id, None)
        # A missing future means the caller abandoned the request, which is not
        # an error — the result is simply discarded.
        if future is not None and not future.done():
            future.set_result(response)


__all__ = ["ContinuousEngine"]

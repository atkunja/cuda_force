"""Concurrent inference engine.

Ties together the pieces: requests arrive from many threads, the batcher
aggregates them, a worker pool executes batches, and results are handed back to
the waiting caller through per-request futures.

    submit() ──► DynamicBatcher ──► ThreadPoolExecutor ──► ModelRunner
        │              (one thread)      (N workers)             │
        └──────────────── Future.result() ◄─────────────────────┘

The batcher thread must not execute batches itself. If it did, no batch could
be formed while one was running, and arrival-time batching would collapse into
strict serialisation — the queue would fill during execution and the next batch
would always be a full one formed from a backlog, regardless of the configured
wait. Handing execution to a pool keeps formation and execution concurrent.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from cudaforge.config import EngineConfig, GenerationConfig
from cudaforge.metrics import MetricsRegistry, MetricsSnapshot
from cudaforge.runners import EchoRunner, ModelRunner
from cudaforge.scheduler import Batch, BatchTrigger, DynamicBatcher, Request

_LOG = logging.getLogger(__name__)


@dataclass
class Response:
    """Result of one request, with the timing breakdown that explains it."""

    request_id: str
    text: str
    prompt_tokens: int = 0
    generated_tokens: int = 0
    queue_time: float = 0.0
    inference_time: float = 0.0
    batch_size: int = 1
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_latency(self) -> float:
        return self.queue_time + self.inference_time

    @property
    def ok(self) -> bool:
        return self.error is None


class EngineClosedError(RuntimeError):
    """Raised when work is submitted to an engine that is shutting down."""


class InferenceEngine:
    """Concurrent, dynamically batched inference.

    Thread-safe: `submit` may be called from any number of threads.
    """

    def __init__(
        self,
        config: EngineConfig | None = None,
        runner: ModelRunner | None = None,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self._config = config or EngineConfig()
        self._metrics = metrics or MetricsRegistry()
        self._runner = runner or EchoRunner()

        self._futures: dict[str, Future[Response]] = {}
        # Guards _futures only. Held for the dictionary operation and released
        # immediately — never across a generate() call, which would serialise
        # every submission behind inference.
        self._futures_lock = threading.Lock()

        self._executor = ThreadPoolExecutor(
            max_workers=self._config.worker_threads, thread_name_prefix="cudaforge-worker"
        )
        self._batcher = DynamicBatcher(
            handler=self._dispatch,
            max_batch_size=self._config.max_batch_size,
            max_wait_seconds=self._config.max_wait_seconds,
            queue_capacity=self._config.queue_capacity,
            metrics=self._metrics,
        )
        self._closed = threading.Event()

        if self._config.warmup_iterations > 0:
            self._runner.warmup(self._config.warmup_iterations)

        _LOG.info("engine ready: %s | %s", self._config.describe(), self._runner.description)

    @property
    def config(self) -> EngineConfig:
        return self._config

    @property
    def metrics(self) -> MetricsRegistry:
        return self._metrics

    @property
    def queue_depth(self) -> int:
        return self._batcher.queue_depth

    def submit(
        self,
        prompt: str,
        generation: GenerationConfig | None = None,
        block_when_full: bool = True,
    ) -> Future[Response]:
        """Enqueue a prompt and return a future for its response.

        Args:
            block_when_full: apply backpressure when True; shed load with a
                rejection when False. Which is correct depends on the caller —
                a batch client wants backpressure, an HTTP frontend generally
                wants to return 503 rather than hold a connection open.

        Raises:
            EngineClosedError: if the engine is shutting down or the request was
                rejected.
            ValueError: if the prompt exceeds ``max_prompt_chars``.
        """
        if self._closed.is_set():
            raise EngineClosedError("engine is shut down")

        limit = self._config.max_prompt_chars
        if limit and len(prompt) > limit:
            raise ValueError(f"prompt of {len(prompt)} characters exceeds the limit of {limit}")

        request = Request(prompt=prompt, generation=generation or self._config.generation)
        future: Future[Response] = Future()

        # Registered before submission: the batcher thread can pick the request
        # up the instant it is enqueued, and would find no future to complete.
        with self._futures_lock:
            self._futures[request.request_id] = future

        accepted = (
            self._batcher.submit(request)
            if block_when_full
            else self._batcher.try_submit(request)
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

        Submitting all of them before waiting on any is the point: it is what
        gives the batcher several requests to aggregate. Submitting and waiting
        one at a time would produce batches of one and measure nothing.
        """
        futures = [self.submit(prompt, generation) for prompt in prompts]
        return [future.result(timeout=timeout) for future in futures]

    def snapshot(self) -> MetricsSnapshot:
        snapshot = self._metrics.snapshot()
        snapshot.queue_depth = self.queue_depth
        snapshot.extra["runner"] = self._runner.description
        snapshot.extra["max_batch_size"] = self._config.max_batch_size
        snapshot.extra["max_wait_us"] = self._config.max_wait_us
        return snapshot

    def shutdown(self, timeout: float = 30.0) -> None:
        """Drain in-flight work and release resources. Idempotent."""
        if self._closed.is_set():
            return
        self._closed.set()

        # Order matters: stop accepting and drain the batcher first, so every
        # queued request is dispatched, then let the executor finish those
        # batches. Shutting the executor first would leave drained batches with
        # nowhere to run.
        self._batcher.shutdown(timeout=timeout)
        self._executor.shutdown(wait=True)

        # Anything still outstanding never reached a worker. Failing the futures
        # is essential: a caller blocked on result() would otherwise hang until
        # its timeout with no explanation.
        with self._futures_lock:
            outstanding = list(self._futures.items())
            self._futures.clear()
        for request_id, future in outstanding:
            if not future.done():
                future.set_exception(
                    EngineClosedError(f"request {request_id} abandoned at shutdown")
                )

    def __enter__(self) -> InferenceEngine:
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()

    def _dispatch(self, batch: Batch) -> None:
        """Called on the batcher thread; hands the batch to the pool and returns."""
        self._executor.submit(self._execute, batch)

    def _execute(self, batch: Batch) -> None:
        started = time.monotonic()
        try:
            results = self._runner.generate(
                batch.prompts, [request.generation for request in batch.requests]
            )
        except Exception as error:  # noqa: BLE001 - the failure belongs to the requests
            # One bad batch must not take the engine down. Every request in it
            # is failed individually so each caller learns what happened.
            _LOG.exception("batch execution failed")
            elapsed = time.monotonic() - started
            for request in batch.requests:
                self._metrics.record_failed()
                self._complete(
                    request.request_id,
                    Response(
                        request_id=request.request_id,
                        text="",
                        queue_time=request.queue_delay,
                        inference_time=elapsed,
                        batch_size=len(batch),
                        error=f"{type(error).__name__}: {error}",
                    ),
                )
            return

        elapsed = time.monotonic() - started
        for request, result in zip(batch.requests, results, strict=True):
            response = Response(
                request_id=request.request_id,
                text=result.text,
                prompt_tokens=result.prompt_tokens,
                generated_tokens=result.generated_tokens,
                queue_time=request.queue_delay,
                inference_time=elapsed,
                batch_size=len(batch),
                metadata={"batch_trigger": batch.trigger.value},
            )
            self._metrics.record_completion(response.total_latency, result.generated_tokens)
            self._complete(request.request_id, response)

    def _complete(self, request_id: str, response: Response) -> None:
        with self._futures_lock:
            future = self._futures.pop(request_id, None)
        # A missing future means the caller abandoned the request, which is not
        # an error — the work is simply discarded.
        if future is not None and not future.done():
            future.set_result(response)


__all__ = ["BatchTrigger", "EngineClosedError", "InferenceEngine", "Response"]

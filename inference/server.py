"""FastAPI front end for the inference engine.

Endpoints:

    GET  /health   liveness and configuration
    GET  /metrics  counters and latency percentiles
    POST /generate single generation request

The server is a thin adapter. Batching, queueing and metrics all live in
``cudaforge.engine`` so they are testable without an HTTP client, and so a
non-HTTP caller gets the same behaviour.

Requests are shed rather than queued when the runtime is saturated: an HTTP
client holding a connection open behind a full queue is worse than a fast 503,
because it converts a throughput problem into a connection-exhaustion problem.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from cudaforge.config import EngineConfig, GenerationConfig
from cudaforge.engine import EngineClosedError, InferenceEngine
from cudaforge.ops import backend_report
from cudaforge.runners import EchoRunner, TransformersRunner
from inference.schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    MetricsResponse,
)

_LOG = logging.getLogger(__name__)

# Populated by the lifespan handler. A module-level slot rather than app.state
# so the type is visible to mypy.
_engine: InferenceEngine | None = None


def build_engine(config: EngineConfig | None = None) -> InferenceEngine:
    """Construct the engine, falling back to the deterministic runner.

    A missing `transformers` install or an unavailable model should not prevent
    the server from starting: the concurrency machinery is worth exercising on
    its own, and a startup crash gives no signal about which part failed.
    """
    config = config or EngineConfig(
        model_name=os.environ.get("CUDAFORGE_MODEL", "sshleifer/tiny-gpt2"),
        max_batch_size=int(os.environ.get("CUDAFORGE_MAX_BATCH", "16")),
        max_wait_us=int(os.environ.get("CUDAFORGE_MAX_WAIT_US", "5000")),
    )

    if os.environ.get("CUDAFORGE_ECHO_RUNNER"):
        _LOG.info("CUDAFORGE_ECHO_RUNNER set; using the deterministic runner")
        return InferenceEngine(config=config, runner=EchoRunner())

    try:
        return InferenceEngine(config=config, runner=TransformersRunner(config))
    except Exception as error:
        _LOG.warning(
            "falling back to the deterministic runner; could not load %s: %s",
            config.model_name,
            error,
        )
        return InferenceEngine(config=config, runner=EchoRunner())


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _engine
    _engine = build_engine()
    _LOG.info("serving: %s", backend_report())
    try:
        yield
    finally:
        if _engine is not None:
            _engine.shutdown()
            _engine = None


app = FastAPI(
    title="CudaForge",
    version="0.1.0",
    description="Concurrent, dynamically batched LLM inference",
    lifespan=lifespan,
)


def _require_engine() -> InferenceEngine:
    if _engine is None:
        raise HTTPException(status_code=503, detail="engine is not ready")
    return _engine


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    engine = _require_engine()
    snapshot = engine.snapshot()
    return HealthResponse(
        status="ok",
        model=engine.config.model_name,
        device=str(engine.config.resolve_device()),
        dtype=str(engine.config.resolve_dtype()),
        custom_cuda_kernels=backend_report().using_custom_kernels,
        queue_depth=engine.queue_depth,
        uptime_seconds=snapshot.uptime_seconds,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    engine = _require_engine()
    return MetricsResponse(**engine.snapshot().to_dict())


@app.post(
    "/generate",
    response_model=GenerateResponse,
    responses={503: {"description": "queue full or shutting down"}},
)
async def generate(request: GenerateRequest) -> GenerateResponse:
    engine = _require_engine()

    generation = GenerationConfig(
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        seed=request.seed,
    )

    try:
        # block_when_full=False: shed load rather than hold the connection.
        future = engine.submit(request.prompt, generation, block_when_full=False)
    except EngineClosedError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    # The engine's future is a concurrent.futures.Future backed by a worker
    # thread. Awaiting it through the event loop's executor keeps the loop free
    # to accept further connections while this request is being generated —
    # calling .result() directly would block the entire server.
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, future.result, 300.0)

    if not response.ok:
        return JSONResponse(
            status_code=500,
            content={"detail": response.error, "request_id": response.request_id},
        )

    return GenerateResponse(
        request_id=response.request_id,
        text=response.text,
        prompt_tokens=response.prompt_tokens,
        generated_tokens=response.generated_tokens,
        queue_time_ms=response.queue_time * 1e3,
        inference_time_ms=response.inference_time * 1e3,
        total_latency_ms=response.total_latency * 1e3,
        batch_size=response.batch_size,
    )

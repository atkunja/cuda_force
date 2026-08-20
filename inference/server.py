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
from fastapi import Response as FastAPIResponse
from fastapi.responses import JSONResponse, PlainTextResponse

from cudaforge.config import EngineConfig, GenerationConfig
from cudaforge.continuous_engine import ContinuousEngine
from cudaforge.engine import EngineClosedError, InferenceEngine, ServingEngine
from cudaforge.exposition import PROMETHEUS_CONTENT_TYPE, render_prometheus
from cudaforge.ops import backend_report
from cudaforge.runners import EchoRunner, TransformersRunner
from cudaforge.stepwise import EchoStepwiseRunner
from inference.schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    MetricsResponse,
    ReadinessResponse,
)

_LOG = logging.getLogger(__name__)

# Populated by the lifespan handler. A module-level slot rather than app.state
# so the type is visible to mypy.
_engine: ServingEngine | None = None


def config_from_environment() -> EngineConfig:
    """Build the engine config from the environment.

    `CUDAFORGE_CONFIG` names a YAML file; the individual variables override it,
    so a container can ship a config and still be adjusted per deployment
    without rebuilding the image.
    """
    path = os.environ.get("CUDAFORGE_CONFIG")
    config = EngineConfig.from_yaml(path) if path else EngineConfig()

    overrides = {
        "model_name": os.environ.get("CUDAFORGE_MODEL"),
        "device": os.environ.get("CUDAFORGE_DEVICE"),
        "max_batch_size": os.environ.get("CUDAFORGE_MAX_BATCH"),
        "max_wait_us": os.environ.get("CUDAFORGE_MAX_WAIT_US"),
        "queue_capacity": os.environ.get("CUDAFORGE_QUEUE_CAPACITY"),
        "worker_threads": os.environ.get("CUDAFORGE_WORKER_THREADS"),
    }
    integer_fields = {
        "max_batch_size",
        "max_wait_us",
        "queue_capacity",
        "worker_threads",
    }
    for field, value in overrides.items():
        if value is None:
            continue
        setattr(config, field, int(value) if field in integer_fields else value)

    # A queue smaller than a batch would make the configured batch size
    # unreachable, so an override that widens the batch widens the queue too.
    config.queue_capacity = max(config.queue_capacity, config.max_batch_size)
    config.__post_init__()
    return config


def build_engine(config: EngineConfig | None = None) -> ServingEngine:
    """Construct the engine, falling back to the deterministic runner.

    A missing `transformers` install or an unavailable model should not prevent
    the server from starting: the concurrency machinery is worth exercising on
    its own, and a startup crash gives no signal about which part failed.

    `CUDAFORGE_CONTINUOUS` selects iteration-level scheduling. It is a different
    engine class rather than a flag on one, because the two need incompatible
    runners — see `ServingEngine`.
    """
    config = config or config_from_environment()
    continuous = bool(os.environ.get("CUDAFORGE_CONTINUOUS"))

    if os.environ.get("CUDAFORGE_ECHO_RUNNER"):
        _LOG.info("CUDAFORGE_ECHO_RUNNER set; using the deterministic runner")
        if continuous:
            return ContinuousEngine(config=config, runner=EchoStepwiseRunner())
        return InferenceEngine(config=config, runner=EchoRunner())

    try:
        if continuous:
            from cudaforge.stepwise_transformers import (  # imported lazily
                TransformersStepwiseRunner,
            )

            return ContinuousEngine(config=config, runner=TransformersStepwiseRunner(config))
        return InferenceEngine(config=config, runner=TransformersRunner(config))
    except Exception as error:  # noqa: BLE001
        # Any model-loading failure should degrade to the deterministic runner
        # rather than prevent the server starting; a crash here gives no signal
        # about which part failed.
        _LOG.warning(
            "falling back to the deterministic runner; could not load %s: %s",
            config.model_name,
            error,
        )
        if continuous:
            return ContinuousEngine(config=config, runner=EchoStepwiseRunner())
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


def _require_engine() -> ServingEngine:
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


@app.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={503: {"description": "not accepting traffic"}},
)
async def ready() -> JSONResponse:
    """Readiness, as distinct from liveness.

    `/health` answers "is this process alive" — a failure there should restart
    the container. This answers "should this instance receive a request right
    now", and a failure should only remove it from rotation. An instance whose
    queue is already full is healthy but not ready; restarting it would drop the
    work it is currently draining.
    """
    if _engine is None:
        payload = ReadinessResponse(
            ready=False,
            reason="engine is still starting",
            queue_depth=0,
            queue_capacity=0,
            saturation=0.0,
        )
        return JSONResponse(status_code=503, content=payload.model_dump())

    capacity = _engine.config.queue_capacity
    depth = _engine.queue_depth
    saturation = depth / capacity if capacity else 0.0

    # The threshold is below 1.0 on purpose. Reporting unready only once the
    # queue is completely full leaves no headroom: by the time the orchestrator
    # reacts, requests are already being rejected.
    ready_now = saturation < 0.9
    payload = ReadinessResponse(
        ready=ready_now,
        reason="accepting requests" if ready_now else "queue is near capacity",
        queue_depth=depth,
        queue_capacity=capacity,
        saturation=round(saturation, 4),
    )
    return JSONResponse(status_code=200 if ready_now else 503, content=payload.model_dump())


@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    engine = _require_engine()
    return MetricsResponse(**engine.snapshot().to_dict())


@app.get(
    "/metrics/prometheus",
    response_class=PlainTextResponse,
    responses={200: {"content": {"text/plain": {}}}},
)
async def metrics_prometheus() -> PlainTextResponse:
    """The same snapshot in the Prometheus text exposition format.

    Kept on its own path rather than negotiated on `/metrics`: the JSON body is
    a documented response model that clients already parse, and switching its
    shape on an `Accept` header would break them silently.
    """
    engine = _require_engine()
    return PlainTextResponse(
        render_prometheus(engine.snapshot()), media_type=PROMETHEUS_CONTENT_TYPE
    )


@app.post(
    "/generate",
    response_model=GenerateResponse,
    responses={503: {"description": "queue full, shutting down, or past its deadline"}},
)
async def generate(
    request: GenerateRequest, response: FastAPIResponse
) -> GenerateResponse | JSONResponse:
    engine = _require_engine()

    # Omitted fields fall back to the engine's configured defaults, so a
    # `generation:` block in a serving config actually takes effect.
    defaults = engine.config.generation
    generation = GenerationConfig(
        max_new_tokens=request.max_new_tokens
        if request.max_new_tokens is not None
        else defaults.max_new_tokens,
        temperature=request.temperature
        if request.temperature is not None
        else defaults.temperature,
        top_p=request.top_p if request.top_p is not None else defaults.top_p,
        top_k=request.top_k if request.top_k is not None else defaults.top_k,
        seed=request.seed if request.seed is not None else defaults.seed,
    )

    try:
        # block_when_full=False: shed load rather than hold the connection.
        future = engine.submit(
            request.prompt,
            generation,
            block_when_full=False,
            deadline_seconds=request.deadline_seconds,
        )
    except EngineClosedError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    # The engine's future is a concurrent.futures.Future backed by a worker
    # thread. Awaiting it through the event loop's executor keeps the loop free
    # to accept further connections while this request is being generated —
    # calling .result() directly would block the entire server.
    loop = asyncio.get_running_loop()
    engine_response = await loop.run_in_executor(None, future.result, 300.0)

    if not engine_response.ok:
        # A missed deadline is the runtime shedding load deliberately, not a
        # server fault. 503 tells the client to retry or back off; 500 would
        # suggest something is broken.
        expired = "RequestExpired" in (engine_response.error or "")
        return JSONResponse(
            status_code=503 if expired else 500,
            content={
                "detail": engine_response.error,
                "request_id": engine_response.request_id,
            },
            headers={"X-Request-ID": engine_response.request_id},
        )

    # Echoed as a header as well as in the body, so a client can correlate a
    # request with a server log line without parsing the payload — which it
    # cannot do at all for the error responses above.
    response.headers["X-Request-ID"] = engine_response.request_id

    return GenerateResponse(
        request_id=engine_response.request_id,
        text=engine_response.text,
        prompt_tokens=engine_response.prompt_tokens,
        generated_tokens=engine_response.generated_tokens,
        queue_time_ms=engine_response.queue_time * 1e3,
        inference_time_ms=engine_response.inference_time * 1e3,
        total_latency_ms=engine_response.total_latency * 1e3,
        batch_size=engine_response.batch_size,
    )

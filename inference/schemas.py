"""Request and response models for the HTTP API.

Validation lives in the schema rather than in the handler so that a malformed
request is rejected at the boundary with a structured 422, before it reaches
the engine and occupies a queue slot.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class GenerateRequest(BaseModel):
    """A single generation request."""

    prompt: str = Field(..., min_length=1, max_length=8192)
    max_new_tokens: int = Field(64, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    top_k: int = Field(0, ge=0, description="0 disables top-k filtering")
    seed: int | None = Field(None, ge=0)
    deadline_seconds: float | None = Field(
        None,
        gt=0,
        le=600,
        description=(
            "Drop this request instead of executing it if it is still waiting "
            "after this many seconds. Set it to your client timeout: past that "
            "point the work is wasted, and under load it displaces requests "
            "someone is still waiting for."
        ),
    )

    @field_validator("prompt")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        # min_length alone would accept a prompt of pure whitespace, which
        # occupies a queue slot and a batch row to produce nothing.
        if not value.strip():
            raise ValueError("prompt must contain non-whitespace characters")
        return value

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "Explain CUDA warps.",
                    "max_new_tokens": 64,
                    "temperature": 0.7,
                }
            ]
        }
    }


class GenerateResponse(BaseModel):
    """Generation result with its timing breakdown.

    The three timings are reported separately because they point at different
    problems: high ``queue_time_ms`` means the runtime is saturated or
    ``max_wait_us`` is too generous, while high ``inference_time_ms`` means the
    model or the batch size is the constraint. A single latency number cannot
    distinguish them.
    """

    request_id: str
    text: str
    prompt_tokens: int
    generated_tokens: int
    queue_time_ms: float
    inference_time_ms: float
    total_latency_ms: float
    batch_size: int


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    dtype: str
    custom_cuda_kernels: bool
    queue_depth: int
    uptime_seconds: float


class ReadinessResponse(BaseModel):
    """Whether this instance should receive traffic right now.

    Distinct from liveness: a process can be perfectly healthy and still be the
    wrong place to send a request — during warmup, or while the queue is already
    full. Conflating the two makes an orchestrator restart a container that only
    needed to be taken out of rotation for a few seconds.
    """

    ready: bool
    reason: str
    queue_depth: int
    queue_capacity: int
    saturation: float


class MetricsResponse(BaseModel):
    """Mirrors ``MetricsSnapshot``; see cudaforge.metrics for field meanings."""

    requests_received: int
    requests_completed: int
    requests_failed: int
    requests_rejected: int
    requests_expired: int

    batches_processed: int
    batches_closed_by_size: int
    batches_closed_by_timeout: int
    average_batch_size: float

    queue_depth: int
    tokens_generated: int

    uptime_seconds: float
    requests_per_second: float
    tokens_per_second: float

    queue_delay_p50_ms: float
    queue_delay_p95_ms: float
    queue_delay_p99_ms: float

    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_max_ms: float
    latency_mean_ms: float

    extra: dict[str, Any] = Field(default_factory=dict)

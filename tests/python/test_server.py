"""HTTP-level tests.

Run against the deterministic runner so nothing is downloaded and the
assertions are stable. What is being tested is the adapter — validation,
status codes, response shape — not generation quality.
"""

from __future__ import annotations

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")

from cudaforge.config import EngineConfig  # noqa: E402
from cudaforge.engine import InferenceEngine  # noqa: E402
from cudaforge.runners import EchoRunner  # noqa: E402
from inference import server  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    config = EngineConfig(
        max_batch_size=8,
        max_wait_us=2_000,
        queue_capacity=64,
        worker_threads=2,
        warmup_iterations=0,
    )
    monkeypatch.setattr(
        server, "build_engine", lambda *_, **__: InferenceEngine(config=config, runner=EchoRunner())
    )
    with fastapi_testclient.TestClient(server.app) as test_client:
        yield test_client


def test_health_reports_configuration(client):
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["device"] in {"cpu", "cuda", "mps"}
    assert isinstance(payload["custom_cuda_kernels"], bool)
    assert payload["queue_depth"] >= 0


def test_generate_returns_a_timing_breakdown(client):
    response = client.post("/generate", json={"prompt": "hello", "max_new_tokens": 8})
    assert response.status_code == 200

    payload = response.json()
    assert payload["text"]
    assert payload["generated_tokens"] == 8
    assert payload["request_id"]
    # The three timings are reported separately because they point at different
    # problems; the total must be their sum.
    assert payload["total_latency_ms"] == pytest.approx(
        payload["queue_time_ms"] + payload["inference_time_ms"], rel=1e-6
    )


def test_metrics_track_completed_requests(client):
    for index in range(5):
        client.post("/generate", json={"prompt": f"p{index}"})

    payload = client.get("/metrics").json()
    assert payload["requests_completed"] == 5
    assert payload["batches_processed"] >= 1
    assert payload["latency_p99_ms"] >= 0


@pytest.mark.parametrize(
    "body",
    [
        {"prompt": ""},
        {"prompt": "   "},
        {"prompt": "ok", "max_new_tokens": 0},
        {"prompt": "ok", "max_new_tokens": 100_000},
        {"prompt": "ok", "temperature": -1},
        {"prompt": "ok", "top_p": 0},
        {"prompt": "ok", "top_p": 2},
        {"prompt": "ok", "top_k": -1},
    ],
)
def test_invalid_requests_are_rejected_at_the_boundary(client, body):
    # 422 rather than 500: the request never reaches the engine and never
    # occupies a queue slot.
    assert client.post("/generate", json=body).status_code == 422


def test_a_missing_prompt_is_rejected(client):
    assert client.post("/generate", json={}).status_code == 422


def test_openapi_schema_is_served(client):
    schema = client.get("/openapi.json").json()
    assert "/generate" in schema["paths"]
    assert "/health" in schema["paths"]
    assert "/metrics" in schema["paths"]

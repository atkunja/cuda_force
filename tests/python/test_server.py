"""HTTP-level tests.

Run against the deterministic runner so nothing is downloaded and the
assertions are stable. What is being tested is the adapter — validation,
status codes, response shape — not generation quality.
"""

from __future__ import annotations

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")

from inference import server  # noqa: E402

from cudaforge.config import EngineConfig, GenerationConfig  # noqa: E402
from cudaforge.engine import InferenceEngine  # noqa: E402
from cudaforge.runners import EchoRunner  # noqa: E402


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


def test_a_deadline_is_accepted(client):
    response = client.post("/generate", json={"prompt": "hello", "deadline_seconds": 5.0})
    assert response.status_code == 200


@pytest.mark.parametrize("value", [0, -1, 601])
def test_an_invalid_deadline_is_rejected(client, value):
    response = client.post("/generate", json={"prompt": "hello", "deadline_seconds": value})
    assert response.status_code == 422


def test_metrics_expose_the_expiry_counter(client):
    client.post("/generate", json={"prompt": "hello"})
    payload = client.get("/metrics").json()
    assert payload["requests_expired"] == 0


def test_ready_reports_accepting_when_idle(client):
    response = client.get("/ready")
    assert response.status_code == 200

    payload = response.json()
    assert payload["ready"] is True
    assert payload["queue_depth"] == 0
    assert payload["queue_capacity"] > 0
    assert payload["saturation"] == 0.0


def test_ready_is_separate_from_health(client):
    # Liveness and readiness answer different questions: a full queue means
    # "take me out of rotation", not "restart me".
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200
    assert set(client.get("/ready").json()) != set(client.get("/health").json())


def test_readiness_is_in_the_openapi_schema(client):
    schema = client.get("/openapi.json").json()
    assert "/ready" in schema["paths"]
    assert "503" in schema["paths"]["/ready"]["get"]["responses"]


def test_prometheus_endpoint_serves_the_text_format(client):
    client.post("/generate", json={"prompt": "hello", "max_new_tokens": 4})
    response = client.get("/metrics/prometheus")

    assert response.status_code == 200
    assert "version=0.0.4" in response.headers["content-type"]
    assert "# TYPE cudaforge_requests_completed_total counter" in response.text
    assert "cudaforge_requests_completed_total 1" in response.text


def test_the_json_metrics_endpoint_is_unchanged_by_the_prometheus_one(client):
    # The JSON body is a documented response model that clients parse; adding
    # the text format must not alter it.
    payload = client.get("/metrics").json()
    assert isinstance(payload, dict)
    assert "requests_completed" in payload


def test_the_request_id_is_echoed_as_a_header(client):
    response = client.post("/generate", json={"prompt": "hello"})
    assert response.status_code == 200
    # Same id in the header and the body, so a client can correlate with a
    # server log line without parsing the payload.
    assert response.headers["X-Request-ID"] == response.json()["request_id"]


def test_request_ids_are_unique_across_requests(client):
    ids = {
        client.post("/generate", json={"prompt": f"p{i}"}).headers["X-Request-ID"]
        for i in range(10)
    }
    assert len(ids) == 10


def test_config_from_environment_reads_a_yaml_file(monkeypatch):
    monkeypatch.setenv("CUDAFORGE_CONFIG", "inference/configs/throughput.yaml")
    config = server.config_from_environment()
    assert config.model_name == "gpt2"
    assert config.max_batch_size == 64
    assert config.worker_threads == 8


def test_environment_variables_override_the_file(monkeypatch):
    # A container ships a config and is still adjustable per deployment without
    # rebuilding the image.
    monkeypatch.setenv("CUDAFORGE_CONFIG", "inference/configs/throughput.yaml")
    monkeypatch.setenv("CUDAFORGE_MAX_BATCH", "8")
    config = server.config_from_environment()
    assert config.max_batch_size == 8
    assert config.max_wait_us == 20_000  # unspecified, so from the file


def test_the_queue_is_widened_to_fit_an_overridden_batch(monkeypatch):
    monkeypatch.setenv("CUDAFORGE_CONFIG", "inference/configs/latency.yaml")
    monkeypatch.setenv("CUDAFORGE_MAX_BATCH", "2048")
    config = server.config_from_environment()
    assert config.queue_capacity >= 2048


def test_no_configuration_yields_the_defaults(monkeypatch):
    for name in (
        "CUDAFORGE_CONFIG",
        "CUDAFORGE_MODEL",
        "CUDAFORGE_DEVICE",
        "CUDAFORGE_MAX_BATCH",
        "CUDAFORGE_MAX_WAIT_US",
        "CUDAFORGE_QUEUE_CAPACITY",
        "CUDAFORGE_WORKER_THREADS",
    ):
        monkeypatch.delenv(name, raising=False)

    from cudaforge.config import EngineConfig

    assert server.config_from_environment() == EngineConfig()


def test_omitted_sampling_fields_fall_back_to_the_engine_defaults(monkeypatch):
    # Without this, the `generation:` block in a serving config would be dead
    # configuration: the schema's own defaults would win every time.
    config = EngineConfig(
        max_batch_size=4,
        max_wait_us=2_000,
        queue_capacity=64,
        warmup_iterations=0,
        generation=GenerationConfig(max_new_tokens=7),
    )
    monkeypatch.setattr(
        server,
        "build_engine",
        lambda *_, **__: InferenceEngine(config=config, runner=EchoRunner()),
    )
    with fastapi_testclient.TestClient(server.app) as client:
        payload = client.post("/generate", json={"prompt": "hello"}).json()
    assert payload["generated_tokens"] == 7


def test_an_explicit_sampling_field_overrides_the_engine_default(monkeypatch):
    config = EngineConfig(
        max_batch_size=4,
        max_wait_us=2_000,
        queue_capacity=64,
        warmup_iterations=0,
        generation=GenerationConfig(max_new_tokens=7),
    )
    monkeypatch.setattr(
        server,
        "build_engine",
        lambda *_, **__: InferenceEngine(config=config, runner=EchoRunner()),
    )
    with fastapi_testclient.TestClient(server.app) as client:
        payload = client.post("/generate", json={"prompt": "hello", "max_new_tokens": 12}).json()
    assert payload["generated_tokens"] == 12


def test_sampling_fields_are_still_range_checked_when_supplied(client):
    # Making them optional must not make them unvalidated.
    for body in (
        {"prompt": "ok", "max_new_tokens": 0},
        {"prompt": "ok", "temperature": 3.0},
        {"prompt": "ok", "top_p": 0.0},
        {"prompt": "ok", "top_k": -1},
    ):
        assert client.post("/generate", json=body).status_code == 422


def test_build_engine_falls_back_when_the_model_cannot_load(monkeypatch):
    # A startup crash gives no signal about which part failed; a degraded start
    # that says so in /health is more useful.
    from cudaforge.runners import TransformersRunner

    def explode(*_args, **_kwargs):
        raise RuntimeError("no such model")

    monkeypatch.setattr(server, "TransformersRunner", explode)
    monkeypatch.delenv("CUDAFORGE_ECHO_RUNNER", raising=False)

    engine = server.build_engine(EngineConfig(model_name="does-not-exist", warmup_iterations=0))
    try:
        assert "EchoRunner" in engine.snapshot().extra["runner"]
    finally:
        engine.shutdown()
    assert TransformersRunner is not None  # the real one is still importable


def test_the_echo_runner_environment_variable_short_circuits_loading(monkeypatch):
    monkeypatch.setenv("CUDAFORGE_ECHO_RUNNER", "1")

    def explode(*_args, **_kwargs):  # must never be reached
        raise AssertionError("model loading should have been skipped")

    monkeypatch.setattr(server, "TransformersRunner", explode)

    engine = server.build_engine(EngineConfig(warmup_iterations=0))
    try:
        assert "EchoRunner" in engine.snapshot().extra["runner"]
    finally:
        engine.shutdown()


def test_endpoints_report_503_before_the_engine_is_ready(monkeypatch):
    # /ready must say "not yet" rather than raising, so an orchestrator sees a
    # clean not-ready signal during startup.
    monkeypatch.setattr(server, "_engine", None)
    import asyncio

    response = asyncio.run(server.ready())
    assert response.status_code == 503


def test_the_prometheus_endpoint_carries_the_engine_configuration(client):
    text = client.get("/metrics/prometheus").text
    # Lets a dashboard group or filter by model and batching configuration.
    assert "cudaforge_build_info" in text
    assert 'max_batch_size="8"' in text


def test_build_engine_selects_the_continuous_scheduler(monkeypatch):
    """`CUDAFORGE_CONTINUOUS` is how `cudaforge-serve --continuous` reaches here."""
    from cudaforge.continuous_engine import ContinuousEngine

    monkeypatch.setenv("CUDAFORGE_ECHO_RUNNER", "1")
    monkeypatch.setenv("CUDAFORGE_CONTINUOUS", "1")

    engine = server.build_engine(EngineConfig(warmup_iterations=0))
    try:
        assert isinstance(engine, ContinuousEngine)
        assert engine.generate("hello").ok
    finally:
        engine.shutdown()


def test_build_engine_defaults_to_the_static_scheduler(monkeypatch):
    from cudaforge.engine import InferenceEngine

    monkeypatch.setenv("CUDAFORGE_ECHO_RUNNER", "1")
    monkeypatch.delenv("CUDAFORGE_CONTINUOUS", raising=False)

    engine = server.build_engine(EngineConfig(warmup_iterations=0))
    try:
        assert isinstance(engine, InferenceEngine)
    finally:
        engine.shutdown()


def test_the_continuous_engine_falls_back_when_the_model_cannot_load(monkeypatch):
    """A model failure must degrade the runner, not the scheduler choice."""
    from cudaforge.continuous_engine import ContinuousEngine

    monkeypatch.delenv("CUDAFORGE_ECHO_RUNNER", raising=False)
    monkeypatch.setenv("CUDAFORGE_CONTINUOUS", "1")

    engine = server.build_engine(
        EngineConfig(model_name="does-not-exist-anywhere", warmup_iterations=0)
    )
    try:
        assert isinstance(engine, ContinuousEngine)
        assert engine.generate("hello").ok
    finally:
        engine.shutdown()


@pytest.fixture
def continuous_client(monkeypatch):
    """The server driven by the iteration-level scheduler.

    Separate from `client` rather than parametrised over it: only a handful of
    endpoints behave differently, and running the whole file twice to check
    three of them would trade real time for little coverage.
    """
    from cudaforge.continuous_engine import ContinuousEngine
    from cudaforge.stepwise import EchoStepwiseRunner

    config = EngineConfig(
        max_batch_size=8,
        queue_capacity=64,
        warmup_iterations=0,
    )
    monkeypatch.setattr(
        server,
        "build_engine",
        lambda *_, **__: ContinuousEngine(config=config, runner=EchoStepwiseRunner()),
    )
    with fastapi_testclient.TestClient(server.app) as test_client:
        yield test_client


def test_the_continuous_engine_serves_generate(continuous_client):
    """End to end through HTTP, not just that build_engine returned the type."""
    response = continuous_client.post("/generate", json={"prompt": "hello"})
    assert response.status_code == 200

    payload = response.json()
    assert payload["text"]
    assert payload["generated_tokens"] > 0


def test_the_continuous_engine_serves_health_and_metrics(continuous_client):
    assert continuous_client.get("/health").json()["status"] == "ok"
    assert continuous_client.get("/metrics").json()["requests_completed"] >= 0

    prometheus = continuous_client.get("/metrics/prometheus")
    assert prometheus.status_code == 200
    # The scheduler is a label on the info metric, so a dashboard can tell the
    # two apart when they are being compared on the same traffic.
    assert "continuous" in prometheus.text


def test_the_continuous_engine_handles_concurrent_requests(continuous_client):
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(
            pool.map(
                lambda index: continuous_client.post("/generate", json={"prompt": f"p{index}"}),
                range(16),
            )
        )

    assert all(response.status_code == 200 for response in responses)
    assert all(response.json()["generated_tokens"] > 0 for response in responses)

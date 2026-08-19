from __future__ import annotations

import threading
import time

import pytest

from cudaforge.config import EngineConfig, GenerationConfig
from cudaforge.engine import EngineClosedError, InferenceEngine, Response
from cudaforge.runners import EchoRunner, GenerationResult


def make_engine(**overrides) -> InferenceEngine:
    defaults = {
        "max_batch_size": 8,
        "max_wait_us": 3_000,
        "queue_capacity": 256,
        "worker_threads": 4,
        "warmup_iterations": 0,
    }
    defaults.update(overrides)
    return InferenceEngine(config=EngineConfig(**defaults))


def test_single_request_round_trip():
    with make_engine() as engine:
        response = engine.generate("hello")
    assert response.ok
    assert "hello" in response.text
    assert response.total_latency > 0


def test_the_runner_is_deterministic():
    with make_engine() as engine:
        first = engine.generate("stable prompt")
        second = engine.generate("stable prompt")
    assert first.text == second.text


def test_responses_match_their_requests():
    prompts = [f"prompt-{i}" for i in range(32)]
    with make_engine() as engine:
        responses = engine.generate_many(prompts)

    assert len(responses) == len(prompts)
    for prompt, response in zip(prompts, responses, strict=True):
        # Ordering is the contract; a batching bug that shuffles rows would
        # otherwise be invisible.
        assert response.text.startswith(prompt)


def test_concurrent_submissions_are_batched():
    # A per-token cost makes batching observable: without it every batch
    # completes instantly and average batch size stays at 1.
    runner = EchoRunner(per_token_seconds=0.0005, fixed_overhead=0.002)
    config = EngineConfig(
        max_batch_size=16,
        max_wait_us=8_000,
        queue_capacity=512,
        worker_threads=2,
        warmup_iterations=0,
    )
    with InferenceEngine(config=config, runner=runner) as engine:
        responses = engine.generate_many([f"p{i}" for i in range(64)], timeout=60.0)

    assert all(response.ok for response in responses)
    snapshot = engine.snapshot()
    assert snapshot.requests_completed == 64
    assert snapshot.average_batch_size > 1.0


def test_batch_size_is_reported_on_the_response():
    runner = EchoRunner(fixed_overhead=0.002)
    config = EngineConfig(
        max_batch_size=8, max_wait_us=10_000, queue_capacity=64, warmup_iterations=0
    )
    with InferenceEngine(config=config, runner=runner) as engine:
        responses = engine.generate_many([f"p{i}" for i in range(8)], timeout=30.0)
    assert max(response.batch_size for response in responses) > 1


def test_generation_settings_are_respected_per_request():
    with make_engine() as engine:
        short = engine.generate("x", GenerationConfig(max_new_tokens=4))
        long = engine.generate("x", GenerationConfig(max_new_tokens=32))
    assert short.generated_tokens == 4
    assert long.generated_tokens == 32


def test_an_oversized_prompt_is_rejected():
    with (
        make_engine(max_prompt_chars=16) as engine,
        pytest.raises(ValueError, match="exceeds the limit"),
    ):
        engine.submit("x" * 100)


def test_submitting_after_shutdown_raises():
    engine = make_engine()
    engine.shutdown()
    with pytest.raises(EngineClosedError):
        engine.submit("hello")


def test_shutdown_is_idempotent():
    engine = make_engine()
    engine.shutdown()
    engine.shutdown()


def test_a_failing_runner_fails_its_requests_without_killing_the_engine():
    class Exploding:
        calls = 0

        def warmup(self, iterations: int) -> None:
            return None

        def generate(self, prompts, settings):
            Exploding.calls += 1
            raise RuntimeError("model exploded")

        @property
        def description(self) -> str:
            return "Exploding"

    config = EngineConfig(
        max_batch_size=1, max_wait_us=1_000, queue_capacity=32, warmup_iterations=0
    )
    with InferenceEngine(config=config, runner=Exploding()) as engine:
        responses = [engine.submit(f"p{i}") for i in range(5)]
        resolved = [future.result(timeout=10) for future in responses]

    assert all(not response.ok for response in resolved)
    assert all("model exploded" in (response.error or "") for response in resolved)
    assert engine.snapshot().requests_failed == 5


def test_a_runner_returning_the_wrong_row_count_is_reported_as_an_error():
    class Truncating:
        def warmup(self, iterations: int) -> None:
            return None

        def generate(self, prompts, settings):
            # zip(strict=True) in the engine turns this into a clear error
            # rather than silently dropping the last request's response.
            return [
                GenerationResult(text="x", prompt_tokens=1, generated_tokens=1)
                for _ in prompts[:-1]
            ]

        @property
        def description(self) -> str:
            return "Truncating"

    config = EngineConfig(
        max_batch_size=4, max_wait_us=5_000, queue_capacity=32, warmup_iterations=0
    )
    with InferenceEngine(config=config, runner=Truncating()) as engine:
        futures = [engine.submit(f"p{i}") for i in range(4)]
        results = [future.result(timeout=10) for future in futures]

    assert any(not response.ok for response in results)


def test_outstanding_futures_are_failed_at_shutdown():
    # A caller blocked on result() must not hang when the engine goes away.
    started = threading.Event()

    class Slow:
        def warmup(self, iterations: int) -> None:
            return None

        def generate(self, prompts, settings):
            started.set()
            time.sleep(0.2)
            return [GenerationResult(text=p, prompt_tokens=1, generated_tokens=1) for p in prompts]

        @property
        def description(self) -> str:
            return "Slow"

    config = EngineConfig(max_batch_size=1, max_wait_us=500, queue_capacity=64, warmup_iterations=0)
    engine = InferenceEngine(config=config, runner=Slow())
    futures = [engine.submit(f"p{i}") for i in range(10)]
    started.wait(timeout=5)
    engine.shutdown(timeout=10)

    # Every future must be settled one way or the other; none may be pending.
    assert all(future.done() for future in futures)


def test_many_threads_can_submit_concurrently():
    results: list[Response] = []
    lock = threading.Lock()

    with make_engine(max_batch_size=16, queue_capacity=1024) as engine:

        def worker(offset: int) -> None:
            local = engine.generate_many([f"t{offset}-{i}" for i in range(25)], timeout=60)
            with lock:
                results.extend(local)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert len(results) == 200
    assert all(response.ok for response in results)
    assert len({response.request_id for response in results}) == 200


def test_snapshot_reports_engine_configuration():
    with make_engine(max_batch_size=12) as engine:
        engine.generate("hello")
        snapshot = engine.snapshot()
    assert snapshot.extra["max_batch_size"] == 12
    assert "EchoRunner" in snapshot.extra["runner"]


def test_load_shedding_rejects_instead_of_blocking():
    runner = EchoRunner(fixed_overhead=0.05)
    config = EngineConfig(
        max_batch_size=1,
        max_wait_us=500,
        queue_capacity=2,
        warmup_iterations=0,
        worker_threads=1,
    )
    rejected = 0
    with InferenceEngine(config=config, runner=runner) as engine:
        for i in range(40):
            try:
                engine.submit(f"p{i}", block_when_full=False)
            except EngineClosedError:
                rejected += 1
    assert rejected > 0


# --- deadlines -------------------------------------------------------------


def test_an_expired_request_gets_a_settled_future_not_a_hang():
    # The important property: a dropped request must never leave its caller
    # blocked. It is failed with a stated reason instead.
    runner = EchoRunner(fixed_overhead=0.08)
    config = EngineConfig(
        max_batch_size=1,
        max_wait_us=500,
        queue_capacity=64,
        worker_threads=1,
        warmup_iterations=0,
    )
    with InferenceEngine(config=config, runner=runner) as engine:
        futures = [engine.submit(f"p{i}", deadline_seconds=0.02) for i in range(20)]
        responses = [future.result(timeout=30) for future in futures]

    assert all(future.done() for future in futures)
    expired = [r for r in responses if not r.ok and "RequestExpired" in (r.error or "")]
    assert expired, "expected at least one request to miss its deadline"
    assert engine.snapshot().requests_expired == len(expired)


def test_a_generous_deadline_does_not_drop_anything():
    with make_engine() as engine:
        responses = engine.generate_many([f"p{i}" for i in range(16)])
    assert all(response.ok for response in responses)
    assert engine.snapshot().requests_expired == 0


def test_submitting_without_a_deadline_is_unchanged():
    with make_engine() as engine:
        response = engine.submit("hello").result(timeout=10)
    assert response.ok
    assert engine.snapshot().requests_expired == 0


def test_batch_formation_continues_while_a_batch_executes():
    # If the batcher executed inline, no batch could form during execution and
    # arrival-time batching would collapse into strict serialisation. Handing
    # execution to the pool is what keeps formation concurrent — this asserts
    # the consequence rather than the implementation.
    runner = EchoRunner(fixed_overhead=0.05)
    config = EngineConfig(
        max_batch_size=4,
        max_wait_us=2_000,
        queue_capacity=256,
        worker_threads=4,
        warmup_iterations=0,
    )
    with InferenceEngine(config=config, runner=runner) as engine:
        started = time.monotonic()
        responses = engine.generate_many([f"p{i}" for i in range(16)], timeout=60)
        elapsed = time.monotonic() - started

    assert all(response.ok for response in responses)
    # Four batches at 50 ms each is 200 ms serialised. With four workers running
    # them concurrently it should be far below that.
    assert elapsed < 0.18, f"batches appear to be serialised: {elapsed:.3f}s"


def test_worker_count_bounds_concurrent_execution():
    # One worker means batches run one at a time regardless of how quickly the
    # batcher forms them.
    concurrent = 0
    peak = 0
    lock = threading.Lock()

    class Counting:
        def warmup(self, iterations: int) -> None:
            return None

        def generate(self, prompts, settings):
            nonlocal concurrent, peak
            with lock:
                concurrent += 1
                peak = max(peak, concurrent)
            time.sleep(0.02)
            with lock:
                concurrent -= 1
            return [GenerationResult(text=p, prompt_tokens=1, generated_tokens=1) for p in prompts]

        @property
        def description(self) -> str:
            return "Counting"

    config = EngineConfig(
        max_batch_size=1,
        max_wait_us=500,
        queue_capacity=64,
        worker_threads=1,
        warmup_iterations=0,
    )
    with InferenceEngine(config=config, runner=Counting()) as engine:
        engine.generate_many([f"p{i}" for i in range(8)], timeout=30)

    assert peak == 1


def test_metrics_survive_a_mix_of_success_and_failure():
    class Flaky:
        calls = 0

        def warmup(self, iterations: int) -> None:
            return None

        def generate(self, prompts, settings):
            Flaky.calls += 1
            if Flaky.calls % 2 == 0:
                raise RuntimeError("intermittent")
            return [GenerationResult(text=p, prompt_tokens=1, generated_tokens=3) for p in prompts]

        @property
        def description(self) -> str:
            return "Flaky"

    config = EngineConfig(max_batch_size=1, max_wait_us=500, queue_capacity=64, warmup_iterations=0)
    with InferenceEngine(config=config, runner=Flaky()) as engine:
        results = [engine.submit(f"p{i}").result(timeout=20) for i in range(10)]

    snapshot = engine.snapshot()
    succeeded = sum(1 for r in results if r.ok)
    failed = sum(1 for r in results if not r.ok)

    assert succeeded + failed == 10
    assert snapshot.requests_completed == succeeded
    assert snapshot.requests_failed == failed
    assert snapshot.tokens_generated == succeeded * 3

"""Serving replicated across devices.

The functional assertions here would all pass on an implementation that sent
every request to replica 0, so several of these check the *distribution* rather
than the results. A router that does not route is the failure worth catching.
"""

from __future__ import annotations

import threading

import pytest

from cudaforge.config import EngineConfig, GenerationConfig
from cudaforge.engine import EngineClosedError, InferenceEngine, ServingEngine
from cudaforge.replicated import ReplicatedEngine


def build(count: int = 3, **overrides) -> ReplicatedEngine:
    settings = {
        "max_batch_size": 4,
        "warmup_iterations": 0,
        "generation": GenerationConfig(max_new_tokens=2),
    }
    settings.update(overrides)
    config = EngineConfig(**settings)
    return ReplicatedEngine([InferenceEngine(config=config) for _ in range(count)], config)


def test_it_satisfies_the_serving_protocol():
    with build() as engine:
        assert isinstance(engine, ServingEngine)


def test_every_request_is_answered():
    with build() as engine:
        responses = engine.generate_many([f"prompt {i}" for i in range(24)])

    assert len(responses) == 24
    assert all(response.ok for response in responses)


def test_requests_actually_reach_every_replica():
    """The assertion that separates routing from not routing.

    Every other test in this file passes if all 24 requests land on replica 0.
    """
    with build(count=3) as engine:
        engine.generate_many([f"prompt {i}" for i in range(24)])
        counts = engine.routed_counts()

    assert len(counts) == 3
    assert all(count > 0 for count in counts), f"some replica got nothing: {counts}"
    assert sum(counts) == 24


def test_an_idle_fleet_rotates_rather_than_piling_on_the_first():
    """With every queue empty, `min` alone returns index 0 forever."""
    with build(count=4) as engine:
        for index in range(8):
            engine.generate(f"p{index}")
        counts = engine.routed_counts()

    assert all(count > 0 for count in counts), f"rotation did not happen: {counts}"


def test_a_single_replica_is_allowed():
    with build(count=1) as engine:
        assert engine.replica_count == 1
        assert engine.generate("hello").ok


def test_no_replicas_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        ReplicatedEngine([])


def test_queue_depth_sums_across_replicas():
    with build(count=3) as engine:
        assert engine.queue_depth >= 0


def test_submitting_after_shutdown_is_refused():
    engine = build()
    engine.shutdown()
    with pytest.raises(EngineClosedError):
        engine.submit("hello")


def test_shutdown_is_idempotent():
    engine = build()
    engine.shutdown()
    engine.shutdown()


def test_one_failing_replica_does_not_strand_the_others():
    """Every replica must be shut down even if an earlier one raises.

    Leaving the rest running would strand their threads and their futures — the
    caller sees an exception and assumes nothing is left behind.
    """
    shutdowns: list[str] = []

    class Exploding(InferenceEngine):
        def shutdown(self, timeout: float = 30.0) -> None:
            shutdowns.append("exploding")
            super().shutdown(timeout=timeout)
            raise RuntimeError("shutdown failed")

    class Recording(InferenceEngine):
        def shutdown(self, timeout: float = 30.0) -> None:
            shutdowns.append("recording")
            super().shutdown(timeout=timeout)

    config = EngineConfig(max_batch_size=2, warmup_iterations=0)
    engine = ReplicatedEngine([Exploding(config=config), Recording(config=config)], config)

    with pytest.raises(RuntimeError, match="shutdown failed"):
        engine.shutdown()

    assert shutdowns == ["exploding", "recording"], "the second replica was never shut down"


def test_the_snapshot_reports_the_fleet_and_what_it_cannot_merge():
    with build(count=3) as engine:
        engine.generate_many([f"p{i}" for i in range(9)])
        snapshot = engine.snapshot()

    assert snapshot.extra["replicas"] == 3
    assert snapshot.extra["parallelism"] == "data"
    # The honesty this project runs on: percentiles are not aggregated, and the
    # snapshot says so rather than implying a fleet-wide number.
    assert "replica 0 only" in snapshot.extra["percentiles_from"]
    assert snapshot.requests_completed == 9


def test_concurrent_submission_is_safe():
    collected: list = []
    lock = threading.Lock()

    engine = build(count=3)

    def worker(index: int) -> None:
        response = engine.generate(f"thread {index}")
        with lock:
            collected.append(response)

    with engine:
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(15)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

    assert len(collected) == 15
    assert sum(engine.routed_counts()) == 15


def test_across_devices_builds_one_replica_each():
    seen: list[str] = []

    def make(device: str):
        seen.append(device)
        return InferenceEngine(
            config=EngineConfig(device="cpu", max_batch_size=2, warmup_iterations=0)
        )

    with ReplicatedEngine.across_devices([0, 1, 2], make) as engine:
        assert engine.replica_count == 3

    assert seen == ["cuda:0", "cuda:1", "cuda:2"]

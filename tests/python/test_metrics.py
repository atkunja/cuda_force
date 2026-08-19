from __future__ import annotations

import threading

import pytest

from cudaforge.metrics import LatencyHistogram, MetricsRegistry


def test_empty_histogram_reports_zeroes():
    histogram = LatencyHistogram()
    assert histogram.count == 0
    assert histogram.percentile(0.5) == 0.0
    assert histogram.mean == 0.0
    assert histogram.maximum == 0.0


def test_histogram_rejects_a_non_positive_capacity():
    with pytest.raises(ValueError, match="capacity"):
        LatencyHistogram(capacity=0)


def test_percentiles_are_exact_within_the_window():
    histogram = LatencyHistogram()
    for value in range(1, 101):
        histogram.record(value / 1000)

    assert histogram.percentile(0.50) == pytest.approx(0.050, abs=0.002)
    assert histogram.percentile(0.99) == pytest.approx(0.100, abs=0.002)
    assert histogram.maximum == pytest.approx(0.100)
    assert histogram.minimum == pytest.approx(0.001)
    assert histogram.mean == pytest.approx(0.0505)


def test_percentiles_are_monotonic():
    histogram = LatencyHistogram()
    for value in range(1, 1000):
        histogram.record(value * 1e-4)
    assert histogram.percentile(0.5) <= histogram.percentile(0.9)
    assert histogram.percentile(0.9) <= histogram.percentile(0.99)
    assert histogram.percentile(0.99) <= histogram.percentile(1.0)


def test_out_of_range_quantiles_are_clamped():
    histogram = LatencyHistogram()
    histogram.record(0.5)
    assert histogram.percentile(-1.0) == pytest.approx(0.5)
    assert histogram.percentile(2.0) == pytest.approx(0.5)


def test_the_window_evicts_the_oldest_samples():
    # Keeping the most recent samples rather than a uniform reservoir is
    # deliberate: "what is p99 right now" is the question a serving system
    # asks, and stale samples from a different load regime mislead.
    histogram = LatencyHistogram(capacity=10)
    for _ in range(10):
        histogram.record(1.0)
    for _ in range(10):
        histogram.record(2.0)

    assert histogram.count == 20  # lifetime count is not windowed
    assert histogram.minimum == pytest.approx(2.0)
    assert histogram.maximum == pytest.approx(2.0)


def test_lifetime_mean_survives_eviction():
    histogram = LatencyHistogram(capacity=4)
    for value in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        histogram.record(value)
    assert histogram.mean == pytest.approx(3.5)


def test_reset_clears_everything():
    histogram = LatencyHistogram()
    for value in range(100):
        histogram.record(value / 100)
    histogram.reset()
    assert histogram.count == 0
    assert histogram.percentile(0.99) == 0.0


def test_concurrent_recording_loses_no_samples():
    histogram = LatencyHistogram()

    def worker(offset: int) -> None:
        for i in range(500):
            histogram.record((offset + i) / 10000)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert histogram.count == 4000


def test_fresh_registry_reports_zeroes():
    snapshot = MetricsRegistry().snapshot()
    assert snapshot.requests_received == 0
    assert snapshot.average_batch_size == 0.0
    assert snapshot.latency_p99_ms == 0.0


def test_counters_accumulate():
    metrics = MetricsRegistry()
    for _ in range(10):
        metrics.record_received()
    metrics.record_rejected()
    metrics.record_failed()
    metrics.record_completion(0.001, tokens=32)
    metrics.record_completion(0.002, tokens=16)

    snapshot = metrics.snapshot()
    assert snapshot.requests_received == 10
    assert snapshot.requests_rejected == 1
    assert snapshot.requests_failed == 1
    assert snapshot.requests_completed == 2
    assert snapshot.tokens_generated == 48


def test_average_batch_size_is_requests_over_batches():
    metrics = MetricsRegistry()
    metrics.record_batch(8, closed_by_timeout=False)
    metrics.record_batch(4, closed_by_timeout=True)

    snapshot = metrics.snapshot()
    assert snapshot.batches_processed == 2
    assert snapshot.average_batch_size == pytest.approx(6.0)
    assert snapshot.batches_closed_by_size == 1
    assert snapshot.batches_closed_by_timeout == 1


def test_a_tail_sample_moves_p99_but_not_p50():
    metrics = MetricsRegistry()
    for _ in range(98):
        metrics.record_completion(0.001, tokens=1)
    for _ in range(2):
        metrics.record_completion(0.5, tokens=1)

    snapshot = metrics.snapshot()
    assert snapshot.latency_p50_ms == pytest.approx(1.0, abs=0.5)
    assert snapshot.latency_p99_ms > snapshot.latency_p50_ms
    assert snapshot.latency_max_ms == pytest.approx(500.0, abs=1.0)


def test_rates_are_derived_from_uptime():
    metrics = MetricsRegistry()
    for _ in range(10):
        metrics.record_completion(0.001, tokens=5)

    snapshot = metrics.snapshot()
    assert snapshot.uptime_seconds > 0
    assert snapshot.requests_per_second > 0
    assert snapshot.tokens_per_second == pytest.approx(snapshot.requests_per_second * 5, rel=1e-6)


def test_reset_clears_every_counter():
    metrics = MetricsRegistry()
    metrics.record_received()
    metrics.record_completion(0.01, tokens=5)
    metrics.record_batch(4, closed_by_timeout=False)
    metrics.reset()

    snapshot = metrics.snapshot()
    assert snapshot.requests_received == 0
    assert snapshot.batches_processed == 0
    assert snapshot.tokens_generated == 0
    assert snapshot.latency_p99_ms == 0.0


def test_snapshot_serialises_to_a_plain_dict():
    metrics = MetricsRegistry()
    metrics.record_completion(0.005, tokens=12)
    payload = metrics.snapshot().to_dict()

    assert payload["requests_completed"] == 1
    assert payload["tokens_generated"] == 12
    assert "latency_p99_ms" in payload


def test_concurrent_recording_loses_no_counts():
    metrics = MetricsRegistry()

    def worker() -> None:
        for _ in range(1000):
            metrics.record_received()
            metrics.record_completion(0.001, tokens=2)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    snapshot = metrics.snapshot()
    assert snapshot.requests_received == 8000
    assert snapshot.requests_completed == 8000
    assert snapshot.tokens_generated == 16000

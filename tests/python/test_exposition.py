"""Prometheus exposition tests.

The format is picky in ways that fail silently at scrape time rather than
loudly here — a missing trailing newline reads as a truncated body, and a
counter declared as a gauge means every rate query over it is quietly wrong.
"""

from __future__ import annotations

from cudaforge.exposition import PROMETHEUS_CONTENT_TYPE, render_prometheus
from cudaforge.metrics import MetricsRegistry


def snapshot_with_activity():
    metrics = MetricsRegistry()
    for _ in range(7):
        metrics.record_received()
    metrics.record_rejected()
    metrics.record_expired()
    metrics.record_completion(0.005, tokens=12)
    metrics.record_batch(4, closed_by_timeout=False)
    metrics.set_queue_depth(3)
    return metrics.snapshot()


def parse(text: str) -> dict[str, float]:
    """Name-to-value map, discarding any label set.

    Split from the right: a labelled metric is `name{a="1",b="2"} 1`, and
    splitting on the first space would land inside the label set.
    """
    values = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, value = line.rpartition(" ")
        values[name.split("{")[0]] = float(value)
    return values


def test_output_ends_with_a_newline():
    # A body without one is rejected as truncated.
    assert render_prometheus(snapshot_with_activity()).endswith("\n")


def test_every_metric_is_declared_before_it_is_used():
    text = render_prometheus(snapshot_with_activity())
    declared_help = {line.split()[2] for line in text.splitlines() if line.startswith("# HELP")}
    declared_type = {line.split()[2] for line in text.splitlines() if line.startswith("# TYPE")}
    emitted = set(parse(text))

    assert emitted == declared_help
    assert emitted == declared_type


def test_counters_are_typed_as_counters():
    # Declaring a counter as a gauge makes every rate() query over it wrong,
    # and nothing reports an error.
    text = render_prometheus(snapshot_with_activity())
    for line in text.splitlines():
        if line.startswith("# TYPE") and line.endswith(" counter"):
            assert line.split()[2].endswith("_total")


def test_gauges_are_not_named_total():
    text = render_prometheus(snapshot_with_activity())
    for line in text.splitlines():
        if line.startswith("# TYPE") and line.endswith(" gauge"):
            assert not line.split()[2].endswith("_total")


def test_every_metric_is_namespaced():
    for name in parse(render_prometheus(snapshot_with_activity())):
        assert name.startswith("cudaforge_")


def test_values_reflect_the_snapshot():
    values = parse(render_prometheus(snapshot_with_activity()))
    assert values["cudaforge_requests_received_total"] == 7
    assert values["cudaforge_requests_completed_total"] == 1
    assert values["cudaforge_requests_rejected_total"] == 1
    assert values["cudaforge_requests_expired_total"] == 1
    assert values["cudaforge_tokens_generated_total"] == 12
    assert values["cudaforge_queue_depth"] == 3
    assert values["cudaforge_average_batch_size"] == 4.0


def test_an_empty_registry_renders_zeroes():
    values = parse(render_prometheus(MetricsRegistry().snapshot()))
    assert values["cudaforge_requests_completed_total"] == 0
    assert values["cudaforge_latency_p99_ms"] == 0.0


def test_every_value_parses_as_a_number():
    # A stray string or NaN literal makes the whole scrape fail, not just the
    # offending line.
    text = render_prometheus(snapshot_with_activity())
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        _, _, value = line.partition(" ")
        float(value)  # raises if malformed


def test_the_content_type_declares_the_format_version():
    assert "version=0.0.4" in PROMETHEUS_CONTENT_TYPE


# --- the info metric -------------------------------------------------------


def snapshot_with_configuration():
    metrics = MetricsRegistry()
    snapshot = metrics.snapshot()
    snapshot.extra.update(
        {"runner": "EchoRunner", "max_batch_size": 16, "max_wait_us": 5000}
    )
    return snapshot


def test_the_configuration_is_exposed_as_labels_on_a_constant():
    # The idiomatic shape for constant, non-numeric facts. Emitting a model name
    # as a metric value would be meaningless.
    text = render_prometheus(snapshot_with_configuration())
    assert "# TYPE cudaforge_build_info gauge" in text
    assert 'runner="EchoRunner"' in text
    assert 'max_batch_size="16"' in text
    assert text.rstrip().endswith("} 1")


def test_no_info_metric_is_emitted_without_configuration():
    # An info metric with no labels carries nothing and would only add noise.
    text = render_prometheus(MetricsRegistry().snapshot())
    assert "cudaforge_build_info" not in text


def test_label_values_are_escaped():
    # A raw quote would truncate the metric and fail the whole scrape body, not
    # just that line.
    snapshot = MetricsRegistry().snapshot()
    snapshot.extra["runner"] = 'Weird"Runner\\with\nnewline'

    text = render_prometheus(snapshot)
    info_line = next(
        line for line in text.splitlines() if line.startswith("cudaforge_build_info{")
    )
    assert '\\"' in info_line
    assert "\n" not in info_line.replace("\\n", "")
    # Exactly two unescaped quotes delimit the value.
    assert info_line.count('"') - info_line.count('\\"') == 2


def test_the_info_metric_does_not_break_the_rest_of_the_output():
    text = render_prometheus(snapshot_with_configuration())
    values = parse(text)
    assert values["cudaforge_build_info"] == 1.0
    assert "cudaforge_requests_completed_total" in values


def test_every_line_still_parses_with_the_info_metric_present():
    for line in render_prometheus(snapshot_with_configuration()).splitlines():
        if line.startswith("#") or not line.strip():
            continue
        _, _, value = line.rpartition(" ")
        float(value)

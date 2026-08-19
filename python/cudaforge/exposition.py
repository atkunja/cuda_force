"""Prometheus text exposition for a metrics snapshot.

The JSON endpoint is what a person reads. This is what a scraper reads, and the
distinction is not cosmetic: Prometheus needs type declarations to know that
`requests_completed` is a counter it may compute a rate over, and that
`latency_p99_ms` is a gauge it must not.

Written by hand rather than pulling in `prometheus_client`. The format is a
dozen lines of text, and the library brings a global registry with process-wide
state that would collide with the per-engine registries this project already
has.
"""

from __future__ import annotations

from cudaforge.metrics import MetricsSnapshot

#: Monotonically increasing totals. Prometheus may compute rates over these; by
#: convention their names end in `_total`.
_COUNTERS: dict[str, str] = {
    "requests_received": "Requests accepted at the ingress.",
    "requests_completed": "Requests that produced a response.",
    "requests_failed": "Requests whose execution raised.",
    "requests_rejected": "Requests refused because the queue was full.",
    "requests_expired": "Requests dropped after passing their deadline while queued.",
    "batches_processed": "Batches handed to the executor.",
    "batched_requests": "Requests contained in those batches.",
    "batches_closed_by_size": "Batches closed on reaching max_batch_size.",
    "batches_closed_by_timeout": "Batches closed on the oldest request's deadline.",
    "tokens_generated": "Tokens produced.",
}

#: Point-in-time values. A rate over any of these is meaningless.
_GAUGES: dict[str, str] = {
    "queue_depth": "Requests currently waiting to be batched.",
    "average_batch_size": "Requests per batch since start.",
    "uptime_seconds": "Seconds since the registry was created.",
    "requests_per_second": "Completed requests per second since start.",
    "tokens_per_second": "Tokens per second since start.",
    "queue_delay_p50_ms": "Median time spent waiting to be batched.",
    "queue_delay_p95_ms": "95th percentile queue delay.",
    "queue_delay_p99_ms": "99th percentile queue delay.",
    "latency_p50_ms": "Median end-to-end latency.",
    "latency_p95_ms": "95th percentile end-to-end latency.",
    "latency_p99_ms": "99th percentile end-to-end latency.",
    "latency_max_ms": "Longest end-to-end latency observed.",
    "latency_mean_ms": "Mean end-to-end latency.",
}

_PREFIX = "cudaforge"


def _format_value(value: float | int) -> str:
    # Prometheus wants a bare number. Python's repr of a float is already a
    # valid one, but integers should not gain a spurious `.0`.
    if isinstance(value, int):
        return str(value)
    return repr(float(value))


def _escape_label(value: str) -> str:
    """Escape a label value per the exposition format.

    Backslash, double quote and newline are the three characters that must be
    escaped; a raw quote in a model name would otherwise truncate the metric and
    the scrape would fail on the whole body, not just that line.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _render_info(snapshot: MetricsSnapshot) -> list[str]:
    """Render the configuration as an info metric.

    The idiomatic way to expose constant, non-numeric facts: a gauge fixed at 1
    carrying the values as labels, so a dashboard can group or filter by model
    and batching configuration. Emitting them as metric *values* would be wrong
    — a model name is not a number, and a batch size is a setting rather than a
    measurement.
    """
    labels = {
        key: str(snapshot.extra.get(key, ""))
        for key in ("runner", "max_batch_size", "max_wait_us")
        if key in snapshot.extra
    }
    if not labels:
        return []

    rendered = ",".join(f'{key}="{_escape_label(value)}"' for key, value in labels.items())
    name = f"{_PREFIX}_build_info"
    return [
        f"# HELP {name} Engine configuration, as labels on a constant.",
        f"# TYPE {name} gauge",
        f"{name}{{{rendered}}} 1",
    ]


def render_prometheus(snapshot: MetricsSnapshot) -> str:
    """Render a snapshot in the Prometheus text exposition format.

    The output ends with a newline, which the format requires — a scrape of a
    body without one is rejected as truncated.
    """
    payload = snapshot.to_dict()
    lines: list[str] = []

    for field, description in _COUNTERS.items():
        name = f"{_PREFIX}_{field}_total"
        lines.append(f"# HELP {name} {description}")
        lines.append(f"# TYPE {name} counter")
        lines.append(f"{name} {_format_value(payload.get(field, 0))}")

    for field, description in _GAUGES.items():
        name = f"{_PREFIX}_{field}"
        lines.append(f"# HELP {name} {description}")
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name} {_format_value(payload.get(field, 0))}")

    lines.extend(_render_info(snapshot))

    return "\n".join(lines) + "\n"


#: Content type Prometheus expects. Serving `text/plain` without the version
#: parameter works with most scrapers but is not what the specification asks for.
PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

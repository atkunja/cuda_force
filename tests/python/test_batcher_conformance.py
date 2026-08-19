"""Cross-implementation conformance for the two batchers.

`docs/concurrency.md` claims the C++ and Python batchers implement the same
policy. Each was tested only against its own expectations, which is exactly how
two implementations drift apart while both suites stay green.

These tests run the *same* scenario through both and compare. Exact equality is
not the goal and would be wrong to assert: the two are separately scheduled, so
batch-by-batch sizes legitimately differ. What must agree is the policy —
never exceeding the size limit, losing nothing, and closing batches for the same
reason under the same arrival pattern.

Skipped when the C++ harness has not been built, with a stated reason. Build it
with `./scripts/build.sh`.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

import pytest

from cudaforge.metrics import MetricsRegistry
from cudaforge.scheduler import Batch, DynamicBatcher, Request

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS = REPO_ROOT / "build" / "tests" / "cpp" / "batcher_scenario"

pytestmark = pytest.mark.skipif(
    not HARNESS.is_file(),
    reason=f"{HARNESS.relative_to(REPO_ROOT)} not built; run ./scripts/build.sh",
)


def run_cpp(max_batch: int, wait_us: int, producers: int, per_producer: int, gap_us: int):
    result = subprocess.run(
        [
            str(HARNESS),
            str(max_batch),
            str(wait_us),
            str(producers),
            str(per_producer),
            str(gap_us),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return json.loads(result.stdout)


def run_python(max_batch: int, wait_us: int, producers: int, per_producer: int, gap_us: int):
    sizes: list[int] = []
    lock = threading.Lock()
    metrics = MetricsRegistry()

    def handler(batch: Batch) -> None:
        with lock:
            sizes.append(len(batch))

    with DynamicBatcher(
        handler,
        max_batch_size=max_batch,
        max_wait_seconds=wait_us / 1e6,
        queue_capacity=max(1024, max_batch * 4),
        metrics=metrics,
    ) as batcher:

        def produce(offset: int) -> None:
            for i in range(per_producer):
                batcher.submit(Request(prompt=f"scenario-{offset}-{i}"))
                if gap_us > 0:
                    time.sleep(gap_us / 1e6)

        threads = [threading.Thread(target=produce, args=(p,)) for p in range(producers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    snapshot = metrics.snapshot()
    return {
        "implementation": "python",
        "max_batch_size": max_batch,
        "max_wait_us": wait_us,
        "submitted": producers * per_producer,
        "batched": sum(sizes),
        "batches": len(sizes),
        "largest_batch": max(sizes) if sizes else 0,
        "closed_by_size": snapshot.batches_closed_by_size,
        "closed_by_timeout": snapshot.batches_closed_by_timeout,
        "sizes": sizes,
    }


SCENARIOS = [
    # (name, max_batch, wait_us, producers, per_producer, gap_us)
    ("saturated", 8, 3_000, 4, 50, 0),
    ("trickle", 16, 2_000, 2, 40, 500),
    ("single_producer", 4, 1_000, 1, 40, 0),
    ("batch_of_one", 1, 1_000, 4, 25, 0),
]


@pytest.mark.parametrize(("name", "batch", "wait", "producers", "each", "gap"), SCENARIOS)
def test_neither_implementation_loses_a_request(name, batch, wait, producers, each, gap):
    cpp = run_cpp(batch, wait, producers, each, gap)
    python = run_python(batch, wait, producers, each, gap)

    expected = producers * each
    assert cpp["batched"] == expected, f"{name}: C++ lost requests"
    assert python["batched"] == expected, f"{name}: Python lost requests"


@pytest.mark.parametrize(("name", "batch", "wait", "producers", "each", "gap"), SCENARIOS)
def test_neither_implementation_exceeds_the_size_limit(name, batch, wait, producers, each, gap):
    cpp = run_cpp(batch, wait, producers, each, gap)
    python = run_python(batch, wait, producers, each, gap)

    assert cpp["largest_batch"] <= batch, name
    assert python["largest_batch"] <= batch, name
    assert all(1 <= size <= batch for size in cpp["sizes"]), name
    assert all(1 <= size <= batch for size in python["sizes"]), name


def test_a_saturated_queue_closes_on_size_in_both():
    # Four producers with no gap against a batch of 8: both should fill batches
    # and close on size, not sit out the deadline.
    cpp = run_cpp(8, 3_000, 4, 50, 0)
    python = run_python(8, 3_000, 4, 50, 0)

    for result in (cpp, python):
        closures = result["closed_by_size"] + result["closed_by_timeout"]
        assert closures > 0
        assert result["closed_by_size"] / closures > 0.5, result["implementation"]


def test_a_batch_size_of_one_never_aggregates_in_either():
    # The degenerate configuration: every batch must hold exactly one request.
    cpp = run_cpp(1, 1_000, 4, 25, 0)
    python = run_python(1, 1_000, 4, 25, 0)

    assert set(cpp["sizes"]) == {1}
    assert set(python["sizes"]) == {1}
    assert cpp["batches"] == 100
    assert python["batches"] == 100


def test_both_aggregate_more_than_one_request_when_saturated():
    # The property that makes batching worth having at all.
    cpp = run_cpp(16, 5_000, 8, 40, 0)
    python = run_python(16, 5_000, 8, 40, 0)

    assert cpp["batched"] / cpp["batches"] > 1.5
    assert python["batched"] / python["batches"] > 1.5

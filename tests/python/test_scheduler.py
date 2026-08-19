from __future__ import annotations

import threading
import time

import pytest

from cudaforge.config import GenerationConfig
from cudaforge.metrics import MetricsRegistry
from cudaforge.scheduler import Batch, BatchTrigger, DynamicBatcher, Request


class Collector:
    """Records batches so a test can assert after shutdown has joined the thread."""

    def __init__(self, delay: float = 0.0) -> None:
        self._lock = threading.Lock()
        self.batches: list[Batch] = []
        self._delay = delay

    def __call__(self, batch: Batch) -> None:
        if self._delay:
            time.sleep(self._delay)
        with self._lock:
            self.batches.append(batch)

    @property
    def sizes(self) -> list[int]:
        with self._lock:
            return [len(batch) for batch in self.batches]

    @property
    def total(self) -> int:
        with self._lock:
            return sum(len(batch) for batch in self.batches)

    @property
    def triggers(self) -> list[BatchTrigger]:
        with self._lock:
            return [batch.trigger for batch in self.batches]


def make_request(index: int = 0) -> Request:
    return Request(prompt=f"prompt-{index}")


def test_request_ids_are_unique():
    assert len({make_request(i).request_id for i in range(100)}) == 100


def test_queue_delay_is_zero_until_dequeued():
    assert make_request().queue_delay == 0.0


def test_batch_reports_the_longest_generation():
    batch = Batch(
        requests=[
            Request(prompt="a", generation=GenerationConfig(max_new_tokens=8)),
            Request(prompt="b", generation=GenerationConfig(max_new_tokens=32)),
        ],
        trigger=BatchTrigger.MAX_SIZE,
    )
    # The batch runs as long as its most demanding member; that is what the
    # executor must budget for.
    assert batch.max_new_tokens == 32
    assert batch.prompts == ["a", "b"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_batch_size": 0}, "max_batch_size"),
        ({"max_wait_seconds": -1}, "max_wait_seconds"),
        ({"max_batch_size": 32, "queue_capacity": 8}, "queue_capacity"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DynamicBatcher(handler=lambda _: None, **kwargs)


def test_an_idle_batcher_produces_no_batches():
    collector = Collector()
    with DynamicBatcher(collector, max_batch_size=16, max_wait_seconds=0.02):
        time.sleep(0.1)
    assert collector.total == 0


def test_a_lone_request_is_released_after_max_wait():
    collector = Collector()
    with DynamicBatcher(collector, max_batch_size=16, max_wait_seconds=0.02) as batcher:
        batcher.submit(make_request())
        time.sleep(0.15)

    assert collector.sizes == [1]
    assert collector.triggers[0] is BatchTrigger.TIMEOUT


def test_a_full_batch_is_released_without_waiting():
    # A long wait makes the assertion meaningful: without size-based closure
    # this would take five seconds.
    collector = Collector()
    start = time.monotonic()
    with DynamicBatcher(collector, max_batch_size=4, max_wait_seconds=5.0) as batcher:
        for i in range(4):
            batcher.submit(make_request(i))
        deadline = time.monotonic() + 2.0
        while collector.total < 4 and time.monotonic() < deadline:
            time.sleep(0.001)
    elapsed = time.monotonic() - start

    assert collector.sizes == [4]
    assert collector.triggers[0] is BatchTrigger.MAX_SIZE
    assert elapsed < 4.0


def test_no_batch_exceeds_max_batch_size():
    collector = Collector()
    with DynamicBatcher(
        collector, max_batch_size=8, max_wait_seconds=0.002, queue_capacity=512
    ) as batcher:
        for i in range(200):
            batcher.submit(make_request(i))

    assert collector.total == 200
    assert all(1 <= size <= 8 for size in collector.sizes)


def test_the_deadline_is_anchored_to_the_oldest_request():
    # A steady trickle must not postpone execution indefinitely. If the deadline
    # were reset on each arrival, no batch would ever close while the producer
    # keeps trickling.
    collector = Collector()
    with DynamicBatcher(
        collector, max_batch_size=1000, max_wait_seconds=0.03, queue_capacity=2000
    ) as batcher:
        stop = threading.Event()

        def trickle() -> None:
            index = 0
            while not stop.is_set():
                batcher.submit(make_request(index))
                index += 1
                time.sleep(0.002)

        producer = threading.Thread(target=trickle)
        producer.start()
        time.sleep(0.25)
        stop.set()
        producer.join()

    assert len(collector.sizes) >= 3
    assert collector.total > 0


def test_shutdown_drains_queued_requests():
    collector = Collector()
    batcher = DynamicBatcher(collector, max_batch_size=8, max_wait_seconds=0.5, queue_capacity=256)
    for i in range(50):
        batcher.submit(make_request(i))
    batcher.shutdown()

    assert collector.total == 50


def test_shutdown_is_idempotent():
    batcher = DynamicBatcher(lambda _: None, max_batch_size=4)
    batcher.shutdown()
    batcher.shutdown()
    assert not batcher.submit(make_request())


def test_try_submit_sheds_load_when_the_queue_is_full():
    # The handler is deliberately slow so the queue backs up. try_submit must
    # reject rather than block, which is the load-shedding contract.
    collector = Collector(delay=0.05)
    accepted = 0
    rejected = 0
    with DynamicBatcher(
        collector, max_batch_size=1, max_wait_seconds=0.001, queue_capacity=2
    ) as batcher:
        for i in range(50):
            if batcher.try_submit(make_request(i)):
                accepted += 1
            else:
                rejected += 1

    assert rejected > 0
    assert accepted + rejected == 50
    assert batcher.metrics.snapshot().requests_rejected == rejected


def test_a_failing_handler_does_not_stall_the_batcher():
    calls = []

    def failing(batch: Batch) -> None:
        calls.append(len(batch))
        raise RuntimeError("handler failure")

    with DynamicBatcher(failing, max_batch_size=1, max_wait_seconds=0.001) as batcher:
        for i in range(10):
            batcher.submit(make_request(i))

    assert len(calls) == 10


def test_metrics_reflect_batching_activity():
    metrics = MetricsRegistry()
    collector = Collector()
    with DynamicBatcher(
        collector,
        max_batch_size=8,
        max_wait_seconds=0.003,
        queue_capacity=256,
        metrics=metrics,
    ) as batcher:
        for i in range(100):
            batcher.submit(make_request(i))

    snapshot = metrics.snapshot()
    assert snapshot.requests_received == 100
    assert snapshot.batched_requests == 100
    assert snapshot.batches_processed == len(collector.sizes)
    assert snapshot.average_batch_size > 1.0
    assert (
        snapshot.batches_closed_by_size + snapshot.batches_closed_by_timeout
        <= snapshot.batches_processed
    )


def test_concurrent_producers_all_get_batched():
    collector = Collector()
    with DynamicBatcher(
        collector, max_batch_size=16, max_wait_seconds=0.003, queue_capacity=1024
    ) as batcher:
        threads = [
            threading.Thread(
                target=lambda offset=offset: [
                    batcher.submit(make_request(offset * 100 + i)) for i in range(100)
                ]
            )
            for offset in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert collector.total == 800


def test_queue_delay_is_recorded_for_every_request():
    metrics = MetricsRegistry()
    with DynamicBatcher(
        lambda _: None, max_batch_size=4, max_wait_seconds=0.005, metrics=metrics
    ) as batcher:
        for i in range(20):
            batcher.submit(make_request(i))

    assert metrics.queue_delay.count == 20


# --- deadline-aware admission ---------------------------------------------


def test_a_request_without_a_deadline_never_expires():
    assert not Request(prompt="x").expired()


def test_expiry_is_evaluated_against_the_deadline():
    request = Request(prompt="x", deadline=time.monotonic() - 1)
    assert request.expired()

    request = Request(prompt="x", deadline=time.monotonic() + 60)
    assert not request.expired()


def test_expired_requests_are_dropped_rather_than_executed():
    # Executing work nobody is waiting for spends capacity the live requests
    # need, which under overload deepens the backlog that caused the timeouts.
    collector = Collector()
    metrics = MetricsRegistry()
    dropped: list[Request] = []

    with DynamicBatcher(
        collector,
        max_batch_size=4,
        max_wait_seconds=0.01,
        queue_capacity=64,
        metrics=metrics,
        on_expired=dropped.append,
    ) as batcher:
        past = time.monotonic() - 1
        for i in range(6):
            batcher.submit(Request(prompt=f"stale-{i}", deadline=past))
        for i in range(4):
            batcher.submit(Request(prompt=f"fresh-{i}"))
        time.sleep(0.2)

    assert len(dropped) == 6
    assert collector.total == 4
    batched = [request for batch in collector.batches for request in batch.requests]
    assert all("fresh" in request.prompt for request in batched)
    assert metrics.snapshot().requests_expired == 6


def test_an_all_expired_queue_produces_no_batches():
    collector = Collector()
    with DynamicBatcher(
        collector, max_batch_size=4, max_wait_seconds=0.01, queue_capacity=64
    ) as batcher:
        past = time.monotonic() - 1
        for i in range(10):
            batcher.submit(Request(prompt=f"stale-{i}", deadline=past))
        time.sleep(0.15)

    assert collector.total == 0


def test_expiry_is_counted_separately_from_rejection():
    # Distinct causes with distinct remedies: rejection means the queue is full,
    # expiry means it is deeper than clients will wait for.
    metrics = MetricsRegistry()
    with DynamicBatcher(
        lambda _: None,
        max_batch_size=2,
        max_wait_seconds=0.005,
        queue_capacity=64,
        metrics=metrics,
    ) as batcher:
        batcher.submit(Request(prompt="stale", deadline=time.monotonic() - 1))
        time.sleep(0.1)

    snapshot = metrics.snapshot()
    assert snapshot.requests_expired == 1
    assert snapshot.requests_rejected == 0
    assert snapshot.requests_failed == 0


def test_a_throwing_expiry_callback_does_not_stop_the_batcher():
    metrics = MetricsRegistry()
    collector = Collector()

    def explode(_: Request) -> None:
        raise RuntimeError("callback failure")

    with DynamicBatcher(
        collector,
        max_batch_size=2,
        max_wait_seconds=0.005,
        queue_capacity=64,
        metrics=metrics,
        on_expired=explode,
    ) as batcher:
        batcher.submit(Request(prompt="stale", deadline=time.monotonic() - 1))
        batcher.submit(Request(prompt="live"))
        time.sleep(0.15)

    assert collector.total == 1

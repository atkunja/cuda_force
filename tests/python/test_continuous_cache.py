"""Continuous batching with admission bounded by KV cache, not by row count.

`max_batch_size` counts rows, and rows are the wrong unit: four sequences of
sixteen tokens and four of four thousand occupy the same number of rows and
wildly different amounts of memory. Only the second set can exhaust a device.
These tests pin the behaviour when a `KVCacheManager` is wired in.
"""

from __future__ import annotations

import threading

from cudaforge.config import GenerationConfig
from cudaforge.continuous import ContinuousBatcher
from cudaforge.kv_cache import KVCacheManager
from cudaforge.scheduler import Request
from cudaforge.stepwise import EchoStepwiseRunner


def drive(
    prompts: list[str],
    *,
    cache: KVCacheManager | None,
    max_batch_size: int = 16,
    tokens: int = 3,
):
    completed: list = []
    lock = threading.Lock()

    def collect(_request, state):
        with lock:
            completed.append(state)

    batcher = ContinuousBatcher(
        EchoStepwiseRunner(),
        collect,
        max_batch_size=max_batch_size,
        cache=cache,
    )
    with batcher:
        for prompt in prompts:
            batcher.submit(
                Request(prompt=prompt, generation=GenerationConfig(max_new_tokens=tokens))
            )
    # Read after the drain, not inside it: `shutdown` is what finishes the
    # backlog, so counters sampled before it describe a partial run.
    return completed, batcher.stats()


def test_without_a_cache_nothing_changes():
    """The cache is opt-in; the default path must behave exactly as before."""
    completed, stats = drive([f"p{i}" for i in range(12)], cache=None)

    assert len(completed) == 12
    assert stats.cache_refusals == 0
    assert stats.preempted == 0
    assert all(state.generated == 3 for state in completed)


def test_a_generous_cache_does_not_interfere():
    cache = KVCacheManager(block_count=256, block_size=16)
    completed, stats = drive([f"p{i}" for i in range(12)], cache=cache)

    assert len(completed) == 12
    assert stats.preempted == 0
    assert all(state.generated == 3 for state in completed)
    # Every sequence released its blocks on the way out.
    assert cache.free_blocks == cache.total_blocks
    assert cache.active_sequences == 0


def test_a_tight_cache_binds_before_the_row_limit():
    """The point of wiring a cache in at all.

    Sixteen rows of headroom, but only 32 tokens of cache. The observed batch
    must be governed by the cache, not by `max_batch_size`.
    """
    cache = KVCacheManager(block_count=8, block_size=4)
    prompts = ["a b c d e f g h"] * 12
    completed, stats = drive(prompts, cache=cache, max_batch_size=16)

    assert len(completed) == 12, "every request must be answered, even under pressure"
    assert stats.max_observed_batch < 16, (
        f"batch reached {stats.max_observed_batch} with only 32 tokens of cache; "
        "admission is still bounded by rows"
    )
    assert stats.preempted > 0, "a cache this tight must force preemption"
    assert cache.recomputed_tokens > 0
    assert cache.free_blocks == cache.total_blocks, "blocks leaked"


def test_blocks_are_returned_when_sequences_finish():
    cache = KVCacheManager(block_count=32, block_size=8)
    drive([f"prompt {i}" for i in range(20)], cache=cache)

    assert cache.free_blocks == cache.total_blocks
    assert cache.active_sequences == 0


def test_a_refused_request_is_deferred_rather_than_dropped():
    """Transient cache pressure must not turn into a failed request.

    The blocks a refused request needs are about to be freed by whatever
    finishes next, so it is held and retried rather than rejected.
    """
    # One block of four tokens: only one short sequence fits at a time.
    cache = KVCacheManager(block_count=1, block_size=4)
    completed, _ = drive(["ab"] * 6, cache=cache, max_batch_size=8, tokens=1)

    assert len(completed) == 6, "a deferred request must still be served"
    assert cache.free_blocks == cache.total_blocks


def test_the_token_estimate_can_be_supplied():
    """Admission decides before the runner has tokenised anything, so the
    prompt length is necessarily an estimate. Callers with a tokeniser should
    be able to supply the real one."""
    seen: list[str] = []

    def estimate(prompt: str) -> int:
        seen.append(prompt)
        return 1

    cache = KVCacheManager(block_count=64, block_size=8)
    completed: list = []
    with ContinuousBatcher(
        EchoStepwiseRunner(),
        lambda _r, state: completed.append(state),
        max_batch_size=4,
        cache=cache,
        estimate_tokens=estimate,
    ) as batcher:
        for index in range(5):
            batcher.submit(
                Request(prompt=f"p{index}", generation=GenerationConfig(max_new_tokens=2))
            )

    assert len(completed) == 5
    assert seen, "the supplied estimator was never consulted"


def test_a_request_larger_than_the_whole_cache_is_rejected():
    """The pathological case: a prompt no amount of eviction can fit.

    It must not be deferred. Deferring assumes something running will free the
    blocks, and when nothing is running that assumption spins the scheduler at
    full tilt until shutdown — measured at 31 seconds before this was fixed.

    Rejection is the honest answer: the request is bigger than the cache, not
    merely unlucky in its timing.
    """
    cache = KVCacheManager(block_count=2, block_size=4)
    rejected: list[tuple] = []
    completed: list = []

    batcher = ContinuousBatcher(
        EchoStepwiseRunner(),
        lambda _r, state: completed.append(state),
        max_batch_size=4,
        cache=cache,
        on_rejected=lambda request, reason: rejected.append((request, reason)),
        # Claim every prompt needs more than the pool holds.
        estimate_tokens=lambda _prompt: 1000,
    )
    with batcher:
        for index in range(3):
            batcher.submit(
                Request(prompt=f"p{index}", generation=GenerationConfig(max_new_tokens=1))
            )
    stats = batcher.stats()

    assert len(rejected) == 3, "every impossible request must be settled, not dropped"
    assert completed == [], "nothing should have been admitted"
    assert stats.rejected == 3
    assert "KV cache" in rejected[0][1]
    assert cache.free_blocks == cache.total_blocks


def test_a_rejected_request_is_settled_even_without_a_callback():
    """No callback must still mean no caller left waiting forever."""
    cache = KVCacheManager(block_count=2, block_size=4)
    completed: list = []

    with ContinuousBatcher(
        EchoStepwiseRunner(),
        lambda _r, state: completed.append(state),
        max_batch_size=4,
        cache=cache,
        estimate_tokens=lambda _prompt: 1000,
    ) as batcher:
        batcher.submit(Request(prompt="p", generation=GenerationConfig(max_new_tokens=1)))

    assert len(completed) == 1, "the request must be settled somehow"
    assert completed[0].generated == 0

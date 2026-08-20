"""A paged KV cache manager, mirroring the C++ one in `cpp/src/kv_cache_manager.cpp`.

The scheduler in `continuous.py` bounds admission by `max_batch_size` alone,
which is the wrong quantity: what actually runs out is cache. Two sequences of
sixteen tokens and two of four thousand occupy the same number of rows and wildly
different amounts of memory, and only the second pair can exhaust the device.

This is the Python side of that. It exists rather than binding the C++ manager
because the engine must keep working with no compiled extension — the same
reason `ops.py` carries reference implementations — and because two
implementations of one specification can be checked against each other.
`tests/python/test_kv_cache_conformance.py` drives both through identical
sequences and requires identical decisions, the same arrangement
`test_batcher_conformance.py` uses for the batcher.

Semantics are the C++ ones exactly, including the parts that are easy to get
subtly different:

* A sequence that cannot fit in the *entire* cache fails immediately rather than
  evicting everything and failing anyway.
* Feasibility is decided before anything is evicted, so a doomed admission never
  destroys a sequence on its way to failing.
* The requester is never its own victim, which is what stops a large admission
  livelocking by evicting itself.
* A preempted sequence stays *known* — its table is emptied, not erased — so it
  can be re-admitted and its prompt recomputed.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum


class PreemptionPolicy(Enum):
    """Which sequence to evict when the cache is full."""

    #: Evict the most recently admitted sequence. Older sequences are closer to
    #: finishing, so protecting them means invested work is not discarded.
    #: Evicting the *oldest* instead repeatedly kills the sequence nearest
    #: completion, and throughput approaches zero while every admission still
    #: reports success.
    NEWEST = "newest"

    #: Evict whichever sequence holds the most blocks. Frees the most memory per
    #: eviction, at the cost of discarding the most invested work.
    LARGEST = "largest"


class AdmissionResult(Enum):
    ADMITTED = "admitted"
    PREEMPTED_OTHERS = "preempted_others"
    INSUFFICIENT_CACHE = "insufficient_cache"


@dataclass
class AdmissionOutcome:
    result: AdmissionResult = AdmissionResult.ADMITTED
    #: Sequences evicted to make room, in the order they were chosen.
    preempted: list[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.result is not AdmissionResult.INSUFFICIENT_CACHE


class BlockAllocator:
    """A fixed pool of reference-counted blocks."""

    def __init__(self, block_count: int, block_size: int) -> None:
        if block_count <= 0:
            raise ValueError(f"block_count must be positive, got {block_count}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self._total = block_count
        self._block_size = block_size
        # Handed out from the end, so the first allocation is block 0 and the
        # order matches the C++ implementation's free list.
        self._free: list[int] = list(range(block_count - 1, -1, -1))
        self._references = [0] * block_count

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def total_blocks(self) -> int:
        return self._total

    @property
    def free_blocks(self) -> int:
        return len(self._free)

    @property
    def used_blocks(self) -> int:
        return self._total - len(self._free)

    @property
    def utilisation(self) -> float:
        return self.used_blocks / self._total if self._total else 0.0

    def allocate(self) -> int | None:
        """Take a free block, or None when exhausted.

        None rather than an exception: running out of cache is expected under
        load, and the scheduler's answer is to preempt, not to unwind.
        """
        if not self._free:
            return None
        block = self._free.pop()
        self._references[block] = 1
        return block

    def add_reference(self, block: int) -> None:
        self._references[block] += 1

    def release(self, block: int) -> None:
        if self._references[block] == 0:
            return
        self._references[block] -= 1
        if self._references[block] == 0:
            self._free.append(block)

    def reference_count(self, block: int) -> int:
        return self._references[block]

    def is_writable(self, block: int) -> bool:
        """False when another sequence shares this block, so appending to it
        would corrupt theirs."""
        return self._references[block] <= 1


class SequenceBlockTable:
    """One sequence's logical-to-physical block mapping."""

    def __init__(self, sequence_id: int, block_size: int) -> None:
        self.sequence_id = sequence_id
        self._block_size = block_size
        self._tokens = 0
        self._blocks: list[int] = []

    @property
    def blocks(self) -> list[int]:
        return self._blocks

    @property
    def token_count(self) -> int:
        return self._tokens

    @property
    def capacity(self) -> int:
        return len(self._blocks) * self._block_size

    @property
    def slack(self) -> int:
        """Unused slots in the final block — the entire internal fragmentation
        of a paged cache, at most `block_size - 1` tokens per sequence."""
        return self.capacity - self._tokens

    def append_block(self, block: int) -> None:
        self._blocks.append(block)

    def add_tokens(self, count: int) -> None:
        if self._tokens + count > self.capacity:
            raise ValueError(
                f"sequence {self.sequence_id}: {count} tokens exceed capacity "
                f"{self.capacity} (holding {self._tokens}) — allocate first"
            )
        self._tokens += count

    def locate(self, token_index: int) -> tuple[int, int]:
        """Physical block holding a token, and its offset within it."""
        if token_index >= self._tokens:
            raise IndexError(f"token {token_index} beyond the {self._tokens} this sequence holds")
        return self._blocks[token_index // self._block_size], token_index % self._block_size


class KVCacheManager:
    """Admission, extension and preemption over a block pool.

    Thread-safe: the scheduler admits from its own thread while metrics are read
    from another.
    """

    def __init__(
        self,
        block_count: int,
        block_size: int,
        policy: PreemptionPolicy = PreemptionPolicy.NEWEST,
    ) -> None:
        self._allocator = BlockAllocator(block_count, block_size)
        self._block_size = block_size
        self._policy = policy
        self._tables: dict[int, SequenceBlockTable] = {}
        self._admission_order: list[int] = []
        self._preemptions = 0
        self._recomputed_tokens = 0
        self._lock = threading.Lock()

    # -- reporting ----------------------------------------------------------

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def total_blocks(self) -> int:
        return self._allocator.total_blocks

    @property
    def free_blocks(self) -> int:
        with self._lock:
            return self._allocator.free_blocks

    @property
    def utilisation(self) -> float:
        with self._lock:
            return self._allocator.utilisation

    @property
    def preemption_count(self) -> int:
        with self._lock:
            return self._preemptions

    @property
    def recomputed_tokens(self) -> int:
        """Tokens that will have to be generated again because their sequence
        was preempted. The bill for choosing recompute over swapping."""
        with self._lock:
            return self._recomputed_tokens

    @property
    def active_sequences(self) -> int:
        with self._lock:
            return len(self._tables)

    def is_admitted(self, sequence: int) -> bool:
        with self._lock:
            return sequence in self._tables

    def tokens_held(self, sequence: int) -> int:
        with self._lock:
            table = self._tables.get(sequence)
            return table.token_count if table else 0

    def blocks_held(self, sequence: int) -> int:
        with self._lock:
            table = self._tables.get(sequence)
            return len(table.blocks) if table else 0

    def locate(self, sequence: int, token_index: int) -> tuple[int, int]:
        with self._lock:
            table = self._tables.get(sequence)
            if table is None:
                raise KeyError(f"locate on unknown sequence {sequence}")
            return table.locate(token_index)

    # -- the scheduler's interface ------------------------------------------

    def admit(self, sequence: int, tokens: int) -> AdmissionOutcome:
        with self._lock:
            return self._reserve(sequence, tokens, creating=True)

    def extend(self, sequence: int, tokens: int) -> AdmissionOutcome:
        """Grow an admitted sequence, evicting others if needed.

        Separate from `admit` because the failure differs: a sequence that
        cannot grow is already running and holding blocks, where one that cannot
        be admitted has none.
        """
        with self._lock:
            return self._reserve(sequence, tokens, creating=False)

    def preempt(self, sequence: int) -> int:
        """Return a sequence's blocks, keeping it known so it can be re-admitted."""
        with self._lock:
            table = self._tables.get(sequence)
            if table is None or not table.blocks:
                return 0
            reclaimed = len(table.blocks)
            for block in table.blocks:
                self._allocator.release(block)
            self._recomputed_tokens += table.token_count
            self._preemptions += 1
            self._tables[sequence] = SequenceBlockTable(sequence, self._block_size)
            return reclaimed

    def release(self, sequence: int) -> None:
        """Finish with a sequence entirely."""
        with self._lock:
            table = self._tables.pop(sequence, None)
            if table is None:
                return
            for block in table.blocks:
                self._allocator.release(block)
            self._admission_order = [s for s in self._admission_order if s != sequence]

    # -- internals ----------------------------------------------------------

    def _additional_blocks_for(self, sequence: int, tokens: int) -> int:
        table = self._tables.get(sequence)
        if table is None:
            return -(-tokens // self._block_size)
        wanted = table.token_count + tokens
        if wanted <= table.capacity:
            # Fits in the last block's slack, which is the common case while
            # decoding: one token at a time into a block holding `block_size`.
            return 0
        return -(-(wanted - table.capacity) // self._block_size)

    def _choose_victim(self, requester: int) -> int | None:
        if self._policy is PreemptionPolicy.NEWEST:
            for candidate in reversed(self._admission_order):
                if candidate == requester:
                    continue
                table = self._tables.get(candidate)
                if table is not None and table.blocks:
                    return candidate
            return None

        victim: int | None = None
        most = 0
        for candidate in self._admission_order:
            if candidate == requester:
                continue
            table = self._tables.get(candidate)
            if table is None:
                continue
            # Strictly greater, so ties go to the earliest admitted — matching
            # the C++ scan, which a `>=` here would silently diverge from.
            if len(table.blocks) > most:
                most = len(table.blocks)
                victim = candidate
        return victim

    def _reclaimable_blocks(self, requester: int) -> int:
        return sum(
            len(table.blocks) for sequence, table in self._tables.items() if sequence != requester
        )

    def _evict_until(self, needed: int, requester: int) -> list[int]:
        # Feasibility first. Discovering halfway through that the demand cannot
        # be met would leave sequences destroyed for an admission that fails
        # anyway — unrecoverable, since their blocks are already back in the pool.
        if self._allocator.free_blocks + self._reclaimable_blocks(requester) < needed:
            return []

        evicted: list[int] = []
        while self._allocator.free_blocks < needed:
            victim = self._choose_victim(requester)
            if victim is None:
                # Unreachable given the check above, which counted the same
                # sequences this scan walks.
                return evicted
            table = self._tables[victim]
            for block in table.blocks:
                self._allocator.release(block)
            self._recomputed_tokens += table.token_count
            self._preemptions += 1
            self._tables[victim] = SequenceBlockTable(victim, self._block_size)
            evicted.append(victim)
        return evicted

    def _reserve(self, sequence: int, tokens: int, creating: bool) -> AdmissionOutcome:
        outcome = AdmissionOutcome()
        if tokens == 0:
            return outcome

        if creating and sequence not in self._tables:
            self._tables[sequence] = SequenceBlockTable(sequence, self._block_size)
            self._admission_order.append(sequence)

        table = self._tables.get(sequence)
        if table is None:
            raise KeyError(f"sequence {sequence} is not admitted")

        needed = self._additional_blocks_for(sequence, tokens)
        if needed > self._allocator.total_blocks:
            # Larger than the entire cache. No amount of eviction helps, and
            # trying would empty the cache and still fail.
            outcome.result = AdmissionResult.INSUFFICIENT_CACHE
            return outcome

        if self._allocator.free_blocks < needed:
            outcome.preempted = self._evict_until(needed, sequence)
            if self._allocator.free_blocks < needed:
                outcome.result = AdmissionResult.INSUFFICIENT_CACHE
                return outcome
            outcome.result = AdmissionResult.PREEMPTED_OTHERS

        for _ in range(needed):
            block = self._allocator.allocate()
            if block is None:
                raise RuntimeError("allocation failed after the free count was checked")
            table.append_block(block)
        table.add_tokens(tokens)
        return outcome


__all__ = [
    "AdmissionOutcome",
    "AdmissionResult",
    "BlockAllocator",
    "KVCacheManager",
    "PreemptionPolicy",
    "SequenceBlockTable",
]

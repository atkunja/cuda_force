# Paged KV Cache

The single largest available win in inference serving, and the reason it is
listed first on the roadmap. What exists here is the **block allocator** — the
part that is a systems problem rather than a kernel, and therefore the part that
can be built and tested on a machine with no GPU.

## Why a contiguous cache is the problem

During generation, every layer's keys and values for every past token must be
kept. A naive implementation reserves a contiguous buffer per sequence, sized
for the longest sequence the server will accept:

```
slot 0: [====used====|················ reserved, unused ················]
slot 1: [==used==|······················ reserved, unused ··············]
```

A server configured for 2048 tokens and serving 100-token requests wastes 95%
of its cache. That waste is what caps concurrency, because concurrency in
inference is cache-bound long before it is compute-bound: the number of
sequences you can serve at once is the cache divided by the *reserved* size, not
the used size.

Two further costs follow from contiguity: a sequence that outgrows its
reservation must be reallocated and copied, and freed reservations leave holes
that only fit sequences of the same size.

## What paging changes

The cache becomes a pool of fixed-size **blocks**, each holding `block_size`
tokens. A sequence holds a **block table** mapping logical block index to
physical block, and grows one block at a time:

```
sequence A tokens 0..35, block_size 16

  logical:   [ block 0 ][ block 1 ][ block 2 ]
  physical:  [   #7    ][   #3    ][   #12   ]   ← wherever there was space
                                        ^^^^ 4 tokens used, 12 free
```

Internal fragmentation collapses to **at most one partly-filled block per
sequence** — under 16 tokens, against thousands. `tests/cpp/test_kv_cache.cpp`
asserts that bound directly, across sequence lengths from 1 to 511.

The indirection through the block table is what an attention kernel would follow
to gather keys and values; it is the mechanism behind PagedAttention.

## Reference counting and prefix sharing

Blocks are reference counted, which buys two things that matter in practice:

**Shared prefixes.** Many requests begin with the same system prompt. Those
tokens produce identical keys and values, so the blocks holding them can be
shared rather than duplicated. With a long system prompt and many concurrent
requests, that is most of the cache.

**Forking.** Beam search and parallel sampling fork a sequence. The fork shares
every block of the common prefix and diverges only from the point it differs.

The rule that makes sharing safe: **a block with more than one referent is not
writable.** Appending to it would corrupt the other referent, so the caller must
copy the block first and swap it into its own table — copy-on-write, at block
granularity. `is_writable` reports this, and `replace_block` performs the swap.

## Choosing the block size

A genuine tradeoff, not a tuning constant:

| Smaller blocks | Larger blocks |
| --- | --- |
| less waste in the final block | fewer block-table entries to index |
| finer-grained sharing | better locality within a block |
| more per-token indirection | more waste in the final block |

16 is the common choice, and is what the tests use. The waste is bounded by
`block_size - 1` tokens per sequence either way.

## Failure is expected, not exceptional

`allocate()` returns `std::nullopt` when the pool is exhausted rather than
throwing. Running out of cache is a normal condition under load, and the
scheduler's response is to **preempt** a sequence — evict its blocks, return it
to the queue, and recompute later — not to unwind the call stack.

Making that an exception would push a routine scheduling decision into an error
path, where it would either be caught and ignored or crash a request that could
have been served a moment later.

## What is not implemented

Deliberately, and stated here rather than implied:

* **No attention kernel reads these blocks.** The allocator is complete; the
  gather that makes it useful is not written.
* **No preemption policy.** The allocator supports it — release the blocks, keep
  the table — but nothing decides *which* sequence to evict.
* **No automatic prefix detection.** Sharing is possible and tested; nothing
  hashes incoming prompts to find the common prefix automatically.
* **No device memory.** The allocator is host-side bookkeeping over block
  *indices*. Wiring it to real device memory is the `MemoryPool` work, and is
  the straightforward part.

## Why build the allocator first

Every hard part of paging is bookkeeping: reference counts, copy-on-write,
eviction, the block table. None of it needs a GPU to be correct, and all of it
is difficult to debug once entangled with device memory — a refcount bug there
surfaces as garbled model output rather than as a failed assertion.

Building it host-side means it is exhaustively tested, including under
ThreadSanitizer, before any device memory is involved. See
[testing.md](testing.md) for the same argument applied to launch geometry.

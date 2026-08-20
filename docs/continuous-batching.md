# Continuous Batching and Preemption

Two roadmap items, built together because they are the same problem seen from
two sides: what to do when a row frees up, and what to do when the cache fills.

## The waste static batching leaves

`DynamicBatcher` forms a batch and runs it until **every** member finishes. A
batch asking for 8, 8, 8 and 256 tokens holds four rows for 256 steps, three of
them idle from step 9:

```
static      [====][====][====][==============================]
                                ^ three rows idle for 248 steps

continuous  [====][====][====][===]
            [new ][new ][new ][new ]   ← admitted into the freed rows
```

This is not a rounding error. Real traffic has a long tail of generation
lengths, so a batch's longest member is routinely many times its median, and
utilisation falls in proportion.

## Iteration-level scheduling

The scheduler regains control after every decode step:

```python
while running or queued:
    admit whatever fits into free rows     # the whole point
    decode_step(running)                   # one token, whole batch
    retire finished sequences
```

A row freed at step 9 is filled at step 10, not at step 256. The batch is never
drained and refilled; it is continuously topped up.

### One token for the whole batch

`decode_step` advances *every* active sequence by one token, because a decode
step is a single forward pass and its cost is dominated by reading the weights
once. Advancing sequences individually would forfeit exactly the amortisation
batching exists for.

That is why the one-shot `ModelRunner.generate` cannot support this, and why
[`stepwise.py`](../python/cudaforge/stepwise.py) introduces a separate protocol
rather than bending the existing one.

## Measured

200 sequences, `max_batch_size` 16, on the development host. No GPU involved:
the runner is simulated, so these measure the **scheduler** — decode steps and
batch occupancy — not model throughput.

| Workload | Median | Max | Ideal | Static | Continuous | Fewer steps | Utilisation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Long-tailed (log-normal) | 11 | 100 | 156 | 524 | 201 | **62%** | 30% → 78% |
| Bimodal, 10% long | 4 | 200 | 283 | 1,804 | 435 | **76%** | 16% → 65% |
| Uniform | 60 | 120 | 563 | 1,078 | 624 | **42%** | 52% → 90% |
| Constant | 32 | 32 | 300 | 320 | 325 | **−2%** | 94% → 92% |

`Ideal` is the lower bound: total work spread across the batch, or the longest
single sequence, whichever is larger. Continuous batching approaches it and
never beats it.

**The constant row is the honest floor.** When every sequence is the same
length, no row is ever freed early, there is nothing to recover, and the extra
per-step scheduling costs about 2%. The gain is entirely a function of how
*spread* the length distribution is — which is why the benchmark sweeps four
rather than reporting one number.

Reproduce with `python benchmarks/benchmark_continuous.py`.

## Preemption: what to do when the cache fills

Continuous batching admits sequences aggressively, which makes running out of
KV cache a routine event rather than an exceptional one. `KVCacheManager` is the
half that decides whom to evict.

### Newest-first, and why

| Policy | Behaviour |
| --- | --- |
| `Newest` (default) | evict the most recently admitted sequence |
| `Largest` | evict whichever holds the most blocks |

Evicting the **newest** protects the sequences closest to finishing, so the work
already invested is not thrown away and requests keep completing.

The failure mode this avoids is worth naming: evicting the *oldest* repeatedly
kills the sequence nearest completion, so tokens are recomputed forever and
throughput approaches zero while every individual admission still "succeeds".
`tests/cpp/test_kv_cache_manager.cpp` asserts the property directly — after ten
rounds of admission against a full cache, the first sequence admitted still
holds its tokens.

`Largest` frees the most per eviction, so fewer sequences are disturbed to
satisfy a given demand, at the cost of recomputing the most work. Useful when
admissions are large and preemption is rare.

### Two rules that are not obvious

**A sequence is never evicted to satisfy its own request.** Otherwise admission
livelocks: the requester is the newest, so newest-first evicts it to make room
for itself, forever.

**Feasibility is decided before anything is evicted.** Discovering halfway
through that the demand cannot be met would leave sequences destroyed for an
admission that then fails anyway — and their blocks cannot be given back once
they are in the pool. The manager sums what is reclaimable first and refuses
without touching anything if it is not enough.

### Recompute, not swap

A preempted sequence's blocks return to the pool and its table is cleared, but
the sequence stays *known*: it can be re-admitted and its prompt re-run.

The alternative — copying its blocks to host memory and back — resumes faster
but needs a device-to-host transfer per eviction, which competes with the copies
the stream scheduler works to overlap. Recompute trades wasted compute for not
touching the copy engines, and is the right default when preemption is rare.
`recomputed_tokens()` reports the bill, so the trade is visible rather than
implicit.

Swapping is not implemented. Saying so is more useful than implying eviction is
free.

## What is still missing

* **The two halves are not wired together.** `KVCacheManager` implements
  admission and preemption in the C++ runtime; `ContinuousBatcher` schedules in
  Python. The Python path has no KV cache to page, so admission is bounded by
  `max_batch_size` alone. Connecting them needs the attention gather that reads
  through the block table, which needs a GPU.
* **No attention kernel reads the block table.** See
  [kv-cache.md](kv-cache.md#what-is-not-implemented).
* **`TransformersRunner` is not step-wise.** `EchoStepwiseRunner` implements the
  protocol; adapting a real model means driving `generate` one token at a time
  with an explicit past-key-value cache, which is straightforward and unwritten.

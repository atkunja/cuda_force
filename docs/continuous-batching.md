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

## Measured on a real model

`EchoStepwiseRunner` sleeps instead of computing, so the numbers above describe
the scheduler with a per-step cost the benchmark chose.
`TransformersStepwiseRunner` drives an actual transformer — real attention over a
real KV cache, resized as sequences join and leave — so the per-step cost is
whatever a forward pass costs.

    python benchmarks/benchmark_continuous_model.py --requests 128 --batch 16 \
        --layers 6 --heads 6 --width 384

128 requests, one in eight generating 60-100 tokens and the rest 4-12, against a
6-layer/384-wide GPT-2 with random weights on CPU:

| batch | decode steps          | occupancy      | decode time | wall-clock |
| ----: | :-------------------- | :------------- | :---------- | :--------- |
|     4 | 914 -> 478  (-47.7%)  | 46.9% -> 89.7% | -6.6%       | **0.83x**  |
|     8 | 710 -> 262  (-63.1%)  | 30.2% -> 81.8% | -37.8%      | **1.06x**  |
|    16 | 502 -> 157  (-68.7%)  | 21.4% -> 68.3% | -51.5%      | **1.26x**  |
|    32 | 360 -> 108  (-70.0%)  | 14.9% -> 49.6% | -53.1%      | **1.44x**  |

The weights are random, so the generated text is meaningless. Nothing in the
table depends on it: step counts, occupancy and timings come from real forward
passes, and token *identity* never enters the measurement.

### The step saving is not the whole story

Continuous batching cuts decode steps by 47-70%, and that saving is a property
of the schedule alone — it would hold on any hardware. Wall-clock does not
follow it, and at batch 4 the policy is a net **loss**.

The cause is visible in the benchmark's own output: the prefill call count.

| batch | prefill calls (static -> continuous) | prefill time     |
| ----: | :----------------------------------- | :--------------- |
|     4 | 32 -> 126                            | 0.113s -> 0.434s |
|    32 | 4 -> 33                              | 0.033s -> 0.138s |

Static batching prefills a full batch at once. Continuous batching refills rows
as they free, and rows free in ones and twos — so the same prompts arrive as
many narrow prefills instead of a few wide ones. Each carries a fixed
framework cost that no amount of scheduling removes.

That tax is roughly constant while the decode saving grows with batch size,
which is why the two cross over around batch 8 on this hardware. Production
systems attack it directly — chunked prefill, or separating prefill from decode
entirely — and this implementation does neither.

Two honest limits on the table. The model is small enough that fixed per-call
overhead is a large share of every forward pass, which inflates the prefill tax
relative to a production-sized model. And CPU decode is compute-bound where GPU
decode is memory-bandwidth-bound, so a wider batch is closer to free on a GPU
than it is here — meaning these figures likely *understate* the gain. Both
directions are stated because neither has been measured on a GPU.

## What is still missing

* **The two halves are not wired together.** `KVCacheManager` implements
  admission and preemption in the C++ runtime; `ContinuousBatcher` schedules in
  Python. The Python path has no KV cache to page, so admission is bounded by
  `max_batch_size` alone. Connecting them needs the attention gather that reads
  through the block table, which needs a GPU.
* **No attention kernel reads the block table.** See
  [kv-cache.md](kv-cache.md#what-is-not-implemented).
* **The runner's cache is contiguous, not paged.** Ragged rows are left-padded
  to a common length, and that padding is wasted memory. Admission also copies
  the whole cache to concatenate a row, where a paged implementation would
  update a block table. This is the interim: correct, and wasteful in exactly
  the way `KVCacheManager` exists to fix.
* **Sequence length is bounded by the padded maximum.** One long sequence sets
  the cache width for every row sharing the batch.

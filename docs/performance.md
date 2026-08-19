# Performance Engineering

A record of every optimisation in this project, in one shape:

> baseline → bottleneck → change → expected impact → **measured impact**

The last field is the one that matters, and for anything requiring a GPU it
reads *not measured*. The development host has no NVIDIA hardware
([environment.md](environment.md)), and inventing a number would make every
other number in the repository worthless.

## Measured on the development host

These ran on an Apple M5 Pro, 18 cores, macOS 26.5, Apple clang 21.

### Dynamic batching throughput

| Field | |
| --- | --- |
| Baseline | `max_batch_size = 1` — every request executed alone |
| Bottleneck | a fixed per-batch cost paid once per *request* instead of once per *batch* |
| Change | aggregate concurrent arrivals up to 16 per batch |
| Expected | throughput rises toward the fixed/variable cost ratio; latency rises by at most `max_wait` |
| **Measured** | **3.52x throughput at unchanged p50 and p99** (16 clients x 10 requests, simulated runner with 4 ms fixed and 0.2 ms/token cost) |

Latency did not rise because the queue was deep enough that batches filled on
size rather than waiting out the deadline. Reproduce with
`python examples/concurrent_requests.py`.

The same sweep at one client shows the cost side honestly: a lone client *loses*
throughput (112 → 87 req/s) because it pays the full `max_wait` with nobody to
batch with. Both directions are real, and which one you get depends entirely on
arrival concurrency.

### Batching under a fixed client count

From `benchmarks/benchmark_batching.py`, 8 concurrent clients:

| `max_batch_size` | req/s | avg batch | p99 (ms) |
| --- | --- | --- | --- |
| 1 | 499 | 1.00 | 12.89 |
| 16 | 722 | 8.00 | 7.88 |

Average batch size saturates at 8, not 16, because eight blocking clients can
have at most eight requests in flight. That is the batcher behaving correctly,
and it is why `average_batch_size` must be read alongside client concurrency
rather than against the configured limit.

### Memory pool reuse

| Field | |
| --- | --- |
| Baseline | `malloc`/`free` per allocation |
| Bottleneck | allocator work per batch |
| Change | size-class free lists |
| Expected | backend calls fall to the number of distinct size classes |
| **Measured** | **2,020 pool allocations served by 5 backend calls; reuse rate 0.9975** |

The wall-clock saving on the host backend is small, because `malloc` already
caches. The number that transfers is the call count: on the device backend each
avoided call is a `cudaMalloc` that would have synchronised the device. That
part is **not measured**.

### Latency histogram accuracy

| Field | |
| --- | --- |
| Baseline | storing every sample and sorting — exact, but unbounded memory |
| Bottleneck | memory growth precisely under the sustained load where percentiles matter most |
| Change | log-linear buckets, 16 sub-buckets per magnitude, O(1) memory |
| Expected | worst-case relative error of 1/16 = 6.25% |
| **Measured** | **worst observed error 4.95%**, across four distributions, at 76–120M records/second |

Errors by distribution, 200k samples each, against exact sorted percentiles:

| Distribution | Worst relative error |
| --- | --- |
| Uniform over 1 ns – 10 ms | 4.95% |
| Log-normal (the shape real latency takes) | 4.53% |
| Bimodal, 2% slow tail | 4.28% |
| Constant | 0.86% |

The bimodal case is the one that matters — it is why percentile reporting exists
at all, and it is where a naive bucketing would do worst. Reproduce with
`./build/benchmarks/bench_histogram`.

### HTTP overhead against runtime time

| Field | |
| --- | --- |
| Baseline | — this is a decomposition, not an optimisation |
| **Measured** | client p99 **279.22 ms** against server p99 **1.03 ms**, 300 requests at concurrency 32 with the deterministic runner |

The runtime is doing essentially nothing in that run, so the difference is
transport and client-side contention. Recorded because it is the reason latency
attributed to "the batcher" is so often not the batcher — and because it is the
argument for `cudaforge-bench` and `benchmark_server.py` being separate tools.

### Correctness under sanitizers

| Field | |
| --- | --- |
| Baseline | 160 C++ tests, 49,866 assertions |
| **Measured** | clean under ThreadSanitizer, AddressSanitizer and UndefinedBehaviorSanitizer |

ASan surfaced a real defect during development: concurrency tests were calling
Catch2's `REQUIRE` from worker threads, which is not thread-safe and aborted
non-deterministically. Fixed by routing worker-thread checks through an atomic
counter and asserting on the main thread after joining.

## Implemented, not measured — requires an NVIDIA GPU

### Reduction: atomics → shared memory → warp shuffle

| Field | |
| --- | --- |
| Baseline | one `atomicAdd` to global memory per element |
| Bottleneck | every thread targets one address; the memory subsystem serialises conflicting atomics, so the grid degenerates to sequential updates |
| Change | grid-stride loads, warp-shuffle reduction, one atomic per block |
| Expected | global atomics fall from N to N/blockDim — about 256x fewer at the default block size; barriers fall from log2(blockDim) to 2 |
| **Measured** | **not measured** |

### Softmax: three passes → online

| Field | |
| --- | --- |
| Baseline | separate max, sum and normalise passes, each reading global memory |
| Bottleneck | 3 reads + 1 write, three times the minimum traffic on a bandwidth-bound kernel |
| Change | online (max, sum) recurrence; 2 reads + 1 write, no capacity limit |
| Expected | approaching a 1.5x reduction in traffic versus naive |
| **Measured** | **not measured** |

### RMSNorm: scalar → `float4`

| Field | |
| --- | --- |
| Baseline | one 32-bit load and store per element |
| Bottleneck | four times more memory instructions than necessary for the same bytes |
| Change | `float4` loads and stores, with an alignment check and scalar fallback |
| Expected | a quarter of the memory instructions; sectors-per-request should drop correspondingly in Nsight Compute |
| **Measured** | **not measured** |

### LoRA: unfused → fused

| Field | |
| --- | --- |
| Baseline | three launches — `XW`, `XA`, then `(XA)B` accumulated |
| Bottleneck | the `batch x rank` intermediate makes a round trip through global memory, and `X` is re-read for the adapter path |
| Change | one kernel holding `XA` in shared memory and consuming it immediately |
| Expected | two fewer launches, one fewer global write and read, one fewer read of `X` |
| **Measured** | **not measured** |

### Matmul: naive → shared-memory tiles

| Field | |
| --- | --- |
| Baseline | each element of A re-read `n` times from global memory |
| Bottleneck | global traffic proportional to the output size times K |
| Change | 16x16 shared-memory tiles, with `[kTile][kTile + 1]` padding to break bank conflicts |
| Expected | global traffic falls by roughly `kTile`; the padding removes a 16-way conflict on the column reads |
| **Measured** | **not measured** — and note this is not competitive with cuBLAS by design |

### Stream overlap

| Field | |
| --- | --- |
| Baseline | copy, compute, copy issued to one stream |
| Bottleneck | copies sit on the critical path; the copy engines and SMs never work simultaneously |
| Change | K streams, pinned staging buffers, event-based cross-stream ordering, zero device-wide synchronisation |
| Expected | throughput approaching the ratio of total work to compute-only work |
| **Measured** | **not measured** — this one is best verified with an Nsight Systems timeline rather than a throughput number |

### Device allocation

| Field | |
| --- | --- |
| Baseline | `cudaMalloc`/`cudaFree` per batch |
| Bottleneck | both synchronise the device, so every allocation drains the pipeline the stream scheduler works to fill |
| Change | the same size-class pool, with a device backend |
| Expected | allocator-induced synchronisations fall to near zero after warmup |
| **Measured** | **not measured** — the host-backend reuse rate above is the closest available proxy |

## Techniques and where each is used

| Technique | Where | Why it applies |
| --- | --- | --- |
| Memory coalescing | every row-wise kernel | rows are contiguous; adjacent threads read adjacent floats |
| Shared-memory tiling | `matmul_tiled`, `softmax_shared` | reuse a loaded tile `kTile` times instead of re-reading |
| Bank-conflict padding | `tile_b[kTile][kTile + 1]` | breaks the power-of-two stride on column reads |
| Warp primitives | all block reductions | last 5 steps in registers; no shared memory, no barrier |
| Reduced synchronisation | two-stage block reduction | 2 barriers instead of log2(blockDim) |
| Kernel fusion | fused LoRA | removes an intermediate round trip and two launches |
| Vectorised access | `rmsnorm_vectorised` | 128-bit transactions instead of 32-bit |
| Occupancy tuning | 256-thread default | four warps per block without register spills |
| Launch overhead | grid-stride loops | fixed grid instead of one block per tile |
| Asynchronous execution | `GpuScheduler` | copies overlap compute on separate hardware |
| Pinned memory | `PinnedBuffer` | the precondition for a copy to overlap at all |
| Memory reuse | `MemoryPool` | avoids the device synchronisation `cudaMalloc` forces |
| Avoiding copies | move semantics through the queue | requests are moved, never copied |
| Batching | `DynamicBatcher` | amortises fixed per-batch cost |
| Lock contention | relaxed atomics for counters | avoids serialising workers that never interact |
| Cache locality | `std::deque` ring in the queue | contiguous chunks rather than per-node allocation |
| Backpressure | bounded queue | converts overload into rejection rather than unbounded latency |

## False sharing

Not observed, and not preemptively worked around.

The counters in `ThreadPool` and `Metrics` are adjacent atomics and are
therefore candidates: if two land on the same 64-byte cache line, the line
ping-pongs between cores and throughput *falls* as threads are added. Padding
each to its own line would fix it, at a cost in memory and clarity.

That change has not been made, because the queue benchmark shows no negative
scaling on this host. Adding padding on suspicion would be an unmeasured
optimisation, which is the thing this document exists to avoid. `perf c2c` is
the tool if the symptom appears; the diagnosis procedure is in
[profiling.md](profiling.md).

## Optimisations deliberately not made

| Not done | Why |
| --- | --- |
| Lock-free queue | the queue is crossed once per request, not once per kernel; the mutex is not on a hot path, and blocked producers still need somewhere to wait |
| Tensor cores in the matmul | cuBLAS already does this far better; the tiled kernel exists to demonstrate the shared-memory argument |
| Splitting and coalescing in the pool | `cudaMallocAsync` and PyTorch's allocator solve it; duplicating them would add risk without insight |
| Persistent kernels | large complexity increase, and the batching layer already addresses launch overhead |
| Custom attention | FlashAttention exists and is better than anything written here would be |

Each of these is a real technique that would help some workload. None was
adopted, because the cost is certain and the benefit here is not.

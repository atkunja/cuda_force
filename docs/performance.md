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

### Correctness under sanitizers

| Field | |
| --- | --- |
| Baseline | 70 C++ tests, 19,430 assertions |
| **Measured** | clean under ThreadSanitizer, AddressSanitizer and UndefinedBehaviorSanitizer |

ASan surfaced a real defect during development: concurrency tests were calling
Catch2's `REQUIRE` from worker threads, which is not thread-safe and aborted
non-deterministically. Fixed by routing worker-thread checks through an atomic
counter and asserting on the main thread after joining.

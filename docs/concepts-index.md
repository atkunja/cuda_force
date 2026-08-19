# Concepts Index

Every claim this repository makes, mapped to the code that backs it. If a
concept is listed here, there is a file and a line you can open.

Ordered by the question someone would actually ask.

## Concurrency

| Question | Code | Discussion |
| --- | --- | --- |
| Why is a mutex needed here at all? | [`concurrent_queue.hpp`](../cpp/include/cudaforge/concurrent_queue.hpp) — class comment | [concurrency.md](concurrency.md#why-a-mutex-not-atomics) |
| Why not atomics? | same — the composite `closed_` + `buffer_` invariant | same |
| Why condition variables over polling? | `push` / `pop` predicate waits | [concurrency.md](concurrency.md#why-condition-variables-not-polling) |
| What is a spurious wakeup and how is it handled? | every `wait` uses the predicate overload | same |
| Why `unique_lock` and not `lock_guard`? | `pop` — `wait` must release the mutex | same |
| What happens on shutdown? | `shutdown()`, and the drain rule in `pop` | [concurrency.md](concurrency.md#shutdown) |
| How does backpressure work? | `push` blocks; `try_push` returns `Full` | [concurrency.md](concurrency.md#backpressure) |
| Where are atomics used, and why relaxed? | [`thread_pool.hpp`](../cpp/include/cudaforge/thread_pool.hpp), [`metrics.hpp`](../cpp/include/cudaforge/metrics.hpp) | [architecture.md](architecture.md#threading-model-summary) |
| How do you know there are no races? | TSan-clean across 70 cases | [testing.md](testing.md) |
| What if a task throws? | `ThreadPool::run` catch block | [architecture.md](architecture.md#failure-isolation) |
| How do futures get their values? | `submit_with_result`, `packaged_task` behind a `shared_ptr` | — |

## Batching and serving

| Question | Code | Discussion |
| --- | --- | --- |
| How are requests dynamically batched? | [`dynamic_batcher.cpp`](../cpp/src/dynamic_batcher.cpp) `collect_batch` | [concurrency.md](concurrency.md#dynamic-batching) |
| Why does batching help at all? | — | [concurrency.md](concurrency.md#dynamic-batching) — weight movement, not arithmetic |
| What stops a steady stream from starving a request? | the deadline is anchored to the first request | [concurrency.md](concurrency.md#the-deadline-is-anchored-not-sliding) |
| Why is batch formation single-threaded? | `DynamicBatcher` owns one thread | same |
| Why does the batcher not execute batches? | `_dispatch` hands off to the pool | [architecture.md](architecture.md#request-lifecycle) |
| Throughput versus latency — which knob? | `max_batch_size` vs `max_wait_us` | [config.hpp](../cpp/include/cudaforge/config.hpp) class comment |
| How do you tell a starved batcher from a saturated one? | `batches_closed_by_size` / `_by_timeout` | [concurrency.md](concurrency.md#reading-the-trigger-counters) |
| What is p99 and how is it computed? | [`latency_histogram.hpp`](../cpp/include/cudaforge/latency_histogram.hpp) | [testing.md](testing.md) |
| Why bucket latencies instead of storing them? | O(1) memory under sustained load | class comment |
| How is load shed? | `try_submit`, and the server's `block_when_full=False` | [concurrency.md](concurrency.md#backpressure) |

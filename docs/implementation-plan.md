# Implementation Plan

## Objective

A GPU-native LLM fine-tuning and concurrent inference runtime. The custom
engineering lives in two places: **CUDA kernels** and the **C++20 concurrency
runtime that feeds them**. Model definitions, tokenizers and training loops
delegate to established libraries — rewriting Transformers would add volume
without adding signal.

## Layering

```
python/cudaforge   ← user-facing API, dispatches CUDA ext or PyTorch fallback
        │
        ├── cpp/    ← portable C++20: queue, pool, batcher, metrics  (runs anywhere)
        │
        └── cuda/   ← kernels + stream scheduler + device allocator  (needs NVIDIA)
```

The portable/CUDA split is deliberate and load-bearing. Everything in `cpp/`
compiles and is tested on any host with a C++20 compiler. Everything under
`cuda/` is guarded behind `CUDAFORGE_ENABLE_CUDA` and is never a build
prerequisite for the portable targets.

## Order of work

Dependencies dictate the order; each stage is testable before the next begins.

| # | Stage | Gate before moving on |
| --- | --- | --- |
| 1 | `ConcurrentQueue` — bounded MPMC, mutex + condvar, shutdown | unit + stress tests green under TSan |
| 2 | `ThreadPool` — futures, atomics, graceful drain | many-producer stress test green |
| 3 | `Metrics` — counters + latency histogram | percentile correctness tests |
| 4 | `DynamicBatcher` — size/time-triggered aggregation | deterministic timing tests |
| 5 | Host `MemoryPool` — size-class free lists | reuse + fragmentation tests |
| 6 | CUDA RAII layer — stream, event, device buffer | compiles under nvcc |
| 7 | `GpuScheduler` — stream assignment, async copies | compiles; runtime pending GPU |
| 8 | Kernels — reduction, softmax, RMSNorm, LoRA, quant | reference parity tests (GPU) |
| 9 | PyTorch bindings + fallback dispatch | Python tests pass on CPU-only host |
| 10 | Inference engine + server | end-to-end on CPU fallback |
| 11 | LoRA fine-tuning pipeline | tiny-model smoke run |
| 12 | Benchmarks, Docker, CI, profiling, docs | CI green |

## Design decisions taken up front

**Bounded queue, not unbounded.** An unbounded request queue converts overload
into unbounded latency and eventual OOM. A bounded queue applies backpressure:
producers block or are rejected, and p99 stays meaningful.

**Condition variables, not polling.** A spin/poll loop burns a core per waiter
and adds latency quantised to the poll interval. Condvars park the thread and
the OS wakes it on the state change.

**One batcher thread per model replica.** Batch formation is a serial decision —
parallelising it would need a lock that serialises it anyway.

**Streams over `cudaDeviceSynchronize`.** Device-wide sync is a barrier across
every stream, which destroys the overlap the scheduler exists to create. All
synchronisation is stream- or event-scoped.

**Fallback dispatch in Python, not C++.** The `cudaforge.ops` layer picks
between the compiled extension and a pure-PyTorch reference at call time. This
keeps the package importable — and its tests runnable — on a machine with no
GPU, which is where it was developed.

## Correctness strategy

Every kernel ships with a PyTorch reference implementation. The reference is
what the tests compare against, with dtype-appropriate tolerances, over shapes
that include non-power-of-two, very small, and large cases. The reference also
*is* the CPU fallback, so it is exercised continuously rather than rotting.

## Honesty constraints

The development host has no NVIDIA GPU (see [environment.md](environment.md)).
Anything requiring one is implemented and documented but explicitly marked as
unmeasured. No benchmark number is written by hand into any file.

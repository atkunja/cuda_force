<div align="center">

# CudaForge

**A GPU-native LLM fine-tuning and concurrent inference runtime built with CUDA C++, C++20, and PyTorch.**

[![python](https://github.com/atkunja/cuda_force/actions/workflows/python.yml/badge.svg)](https://github.com/atkunja/cuda_force/actions/workflows/python.yml)
[![cpp](https://github.com/atkunja/cuda_force/actions/workflows/cpp.yml/badge.svg)](https://github.com/atkunja/cuda_force/actions/workflows/cpp.yml)
[![lint](https://github.com/atkunja/cuda_force/actions/workflows/lint.yml/badge.svg)](https://github.com/atkunja/cuda_force/actions/workflows/lint.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

---

> **On measurement.** This project was developed on Apple Silicon, which has no
> CUDA path. Every GPU-dependent component is implemented, structurally checked
> and unit-tested against a reference — but **no GPU performance number appears
> anywhere in this repository**, because none was measured. Host-side results
> are labelled with the machine that produced them. See
> [PROJECT_STATUS.md](PROJECT_STATUS.md) for the exact split.

## What this is

Two things that usually live in separate projects:

1. **Custom CUDA kernels** — reduction, softmax, RMSNorm, LoRA linear and
   block-wise INT8 quantisation, each in a naive and an optimised form, exposed
   to PyTorch through the dispatcher.
2. **A concurrent runtime that feeds them** — a bounded MPMC queue, a thread
   pool, a deadline-anchored dynamic batcher, a CUDA stream scheduler and a
   caching device allocator.

Plus the parts that make those usable: a LoRA/QLoRA fine-tuning pipeline, an
inference engine with an HTTP front end, benchmarks, and documentation that
explains *why* each thing is shaped the way it is.

## Architecture

```
                        Client requests
                              │
                              ▼
                  ┌───────────────────────┐
                  │   ConcurrentQueue     │   bounded · mutex + condvar
                  │   backpressure here   │   rejects or blocks; never grows
                  └───────────┬───────────┘
                              ▼
                  ┌───────────────────────┐
                  │   DynamicBatcher      │   closes on size OR on the
                  │   single thread       │   oldest request's deadline
                  └───────────┬───────────┘
                              ▼
                  ┌───────────────────────┐
                  │   Worker pool         │   formation and execution
                  │   N workers           │   stay concurrent
                  └───────────┬───────────┘
                              ▼
                  ┌───────────────────────┐
                  │   GpuScheduler        │   round-robin over K streams
                  └───────────┬───────────┘
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
          Stream 0        Stream 1    …   Stream K-1
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                  ┌───────────────────────┐
                  │   Custom CUDA ops     │   RMSNorm · softmax · LoRA
                  │   + MemoryPool        │   quantise · reduce
                  └───────────┬───────────┘
                              ▼
                  ┌───────────────────────┐
                  │   Transformer model   │
                  └───────────┬───────────┘
                              ▼
                          Responses
```

Each stage decouples a rate mismatch. Details in
[docs/architecture.md](docs/architecture.md).

## Technical highlights

<table>
<tr><td valign="top" width="50%">

**CUDA**

- Custom kernels in naive + optimised forms
- Warp shuffles (`__shfl_down_sync`) with explicit masks
- Shared-memory tiling with bank-conflict padding
- Memory coalescing and `float4` vectorised access
- Two-stage block reductions (2 barriers, not log N)
- Kernel fusion for the LoRA adapter path
- Numerically stable softmax and FP32 accumulators
- CUDA streams and events; **zero** `cudaDeviceSynchronize`
- Pinned host memory for genuine async copies
- Caching device allocator over size classes
- RAII for every stream, event and allocation
- Typed errors distinguishing recoverable from sticky

</td><td valign="top" width="50%">

**C++20 / systems**

- Bounded MPMC queue: mutex + condition variables
- Predicate waits — no bare `wait`, no spurious-wakeup bugs
- Thread pool with futures and graceful drain
- Relaxed atomics for hot-path counters
- Backpressure, and explicit load shedding
- Idempotent shutdown that never loses accepted work
- Fixed-memory log-linear latency histogram
- Clean under TSan, ASan and UBSan

**ML**

- LoRA from scratch + PEFT integration
- QLoRA via bitsandbytes NF4 (scope stated, not overclaimed)
- Block-wise INT8 quantisation with a proven error bound
- Gradient accumulation, checkpointing, mixed precision
- Dynamic batching, DDP/NCCL data parallelism

</td></tr>
</table>

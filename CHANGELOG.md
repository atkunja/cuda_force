# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

## [0.1.0] — 2026-08-19

First release. Everything below is implemented; what has and has not been
*executed* is recorded in [PROJECT_STATUS.md](PROJECT_STATUS.md), because the
development host has no NVIDIA GPU.

### Added — CUDA

- Reduction kernels: naive atomic, shared-memory tree, warp-shuffle, plus a
  row-wise variant.
- Softmax: naive, shared-memory, online recurrence, and FP16, with a
  capacity-aware fallback for long rows.
- RMSNorm: scalar, `float4` vectorised with alignment checks and a scalar
  fallback, and FP16 with FP32 accumulation.
- LoRA linear: tiled matmul with bank-conflict padding, plus fused and unfused
  adapter paths.
- Activations: SiLU, GELU (tanh form) and fused SwiGLU, in scalar, `float4`
  and FP16 paths.
- Fused residual add + RMSNorm, emitting both the normalised output and the
  carried residual in one pass.
- Block-wise symmetric INT8 quantise, dequantise and fake-quantise.
- `GpuScheduler`: round-robin stream leases, event-based cross-stream chaining,
  asynchronous copies, per-stream accounting.
- RAII wrappers for streams, events, device buffers and pinned buffers.
- `CudaError` carrying the status code, with sticky-vs-recoverable
  classification, and check macros on every call site.
- Optional NVTX range annotations, compiled out by default.

### Added — C++ runtime

- `ConcurrentQueue<T>`: bounded MPMC, mutex + condition variables, predicate
  waits, idempotent shutdown that drains before reporting closed.
- `ThreadPool`: futures, relaxed-atomic counters, per-task exception isolation,
  graceful drain.
- `DynamicBatcher`: deadline anchored to the oldest request; drops requests past
  their own deadline.
- `MemoryPool<Backend>`: size-class caching allocator over a concept-constrained
  backend, with host, device and pinned backends.
- `LatencyHistogram`: fixed-memory log-linear buckets, measured worst-case error
  of 4.95% against a documented 6.25% bound.
- `Metrics`: counters plus queue-delay and latency percentiles, JSON output.
- Paged KV cache block allocator: reference-counted blocks, per-sequence block
  tables, copy-on-write for shared prefixes.

### Added — Python

- Operators dispatching to the compiled extension or to reference PyTorch
  implementations, with `backend_report()` reporting which path is active.
- An explicit autograd guard: differentiating through a kernel raises rather
  than producing silently incorrect gradients, and calls that need gradients are
  routed to the differentiable reference automatically.
- `InferenceEngine`: concurrent submission, dynamic batching, future-based
  results, load shedding, deadline-aware dropping, graceful shutdown that
  settles every outstanding future.
- FastAPI server with `/health`, `/ready`, `/metrics`, `/metrics/prometheus`
  and `/generate`, request-id echo, and 503 on shed or expired requests.
- LoRA/QLoRA fine-tuning pipeline with an explicit training loop, packed causal
  LM datasets, token-weighted perplexity, and adapter-only checkpoints.
- Prometheus text exposition with correct counter/gauge typing.

### Added — tooling

- `check_cuda_sources.py`: structural CUDA rules enforceable without a toolkit.
- `check_docs.py`: Markdown link and anchor validation.
- Benchmarks for the queue, batcher, memory pool, latency histogram, operators,
  end-to-end batching, the HTTP server, and CUDA kernels, plus a dependency-free
  Markdown summariser for their output.
- CI across three workflows, a multi-stage CUDA Dockerfile, compose services,
  pre-commit hooks, and a Makefile.
- Fourteen documents, including a concepts index mapping every claim to the code
  that implements it.

### Known limitations

Recorded in full in [PROJECT_STATUS.md](PROJECT_STATUS.md). The significant
ones: no KV cache, batches are static once formed, the memory pool is not
stream-ordered, the tiled matmul is not competitive with cuBLAS, and no GPU
performance number has been measured.

[Unreleased]: https://github.com/atkunja/cuda_force/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/atkunja/cuda_force/releases/tag/v0.1.0

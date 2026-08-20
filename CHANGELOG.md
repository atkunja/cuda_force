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
- BF16 paths for RMSNorm, softmax and SwiGLU, sharing one templated kernel body
  with the FP16 versions — the dtype `EngineConfig` prefers on Ampere and later.
- Optional NVTX range annotations for Nsight Systems, compiled out by default.
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
- `KVCacheManager`: admission, extension and recompute-based preemption, with
  newest-first and largest-first eviction policies.
- `ContinuousBatcher`: iteration-level scheduling that refills rows freed by
  finished sequences, and the step-wise runner protocol it needs.

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
- Prometheus text exposition with correct counter/gauge typing, on its own path
  so the JSON response model is unchanged.
- `/ready` as a readiness signal distinct from `/health` liveness.
- Deadline-aware admission: requests past their deadline are dropped rather than
  executed, checked both at dequeue and immediately before execution.
- Serving configs (latency, balanced, throughput) loadable by the CLI and by the
  server, with flag and environment overrides.

### Added — tooling

- `check_cuda_sources.py`: structural CUDA rules enforceable without a toolkit.
- `check_docs.py`: Markdown link and anchor validation.
- `check_references.py`: validation of file paths named in prose.
- Benchmarks for the queue, batcher, memory pool, latency histogram, operators,
  end-to-end batching, the HTTP server, and CUDA kernels, plus a dependency-free
  Markdown summariser for their output.
- CI across three workflows, a multi-stage CUDA Dockerfile, compose services,
  pre-commit hooks, and a Makefile.
- Fourteen documents, including a concepts index mapping every claim to the code
  that implements it.

### Measured on the development host

No GPU was involved in any of these:

- Dynamic batching: 3.52x throughput at unchanged p50/p99 (16 clients); 10.9x
  throughput under saturation with p99 queue delay falling 671 ms to 65 ms.
- Bounded queue: 2.39M items/s at one producer and consumer, 774k at eight of
  each — where the single mutex becomes the bottleneck.
- Paged KV cache: 13.2x more concurrent sequences than a contiguous cache on
  chat-shaped traffic; 25.8x on short prompts; 1.0x when every sequence reaches
  the limit.
- Latency histogram: 4.95% worst-case error against a documented 6.25% bound.
- Metrics overhead: 16.38 → 1.76 µs per record after retuning the window.
- HTTP: client p99 279 ms against server p99 1.03 ms — the gap is transport.

### Known limitations

Recorded in full in [PROJECT_STATUS.md](PROJECT_STATUS.md). The significant
ones: no KV cache, batches are static once formed, the memory pool is not
stream-ordered, the tiled matmul is not competitive with cuBLAS, and no GPU
performance number has been measured.

[Unreleased]: https://github.com/atkunja/cuda_force/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/atkunja/cuda_force/releases/tag/v0.1.0

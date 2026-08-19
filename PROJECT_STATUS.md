# Project Status

A precise account of what was built, what was executed, and what was not — so
that nothing in this repository has to be taken on trust.

**Development host:** Apple M5 Pro (arm64), macOS 26.5.2, 18 cores, 48 GB,
Apple clang 21, Python 3.12.14, PyTorch 2.13.0.
**No NVIDIA GPU. No CUDA toolkit.** Full report:
[docs/environment.md](docs/environment.md).

---

## Completed

### Portable C++20 runtime — `cpp/`

| Component | Description |
| --- | --- |
| `ConcurrentQueue<T>` | Bounded MPMC queue. Mutex + two condition variables, predicate waits, blocking and non-blocking push/pop, bounded-wait pop, idempotent shutdown that drains before reporting closed. |
| `ThreadPool` | Fixed workers over a bounded task queue. Futures via `packaged_task`, relaxed-atomic counters, exception isolation per task, graceful drain on destruction. |
| `DynamicBatcher` | Single-threaded batch formation. Closes on `max_batch_size` or on the *oldest* request's deadline. Handler exceptions isolated. |
| `MemoryPool<Backend>` | Size-class caching allocator, concept-constrained backend, allocation accounting, `trim()`, foreign-pointer rejection. |
| `LatencyHistogram` | Fixed-memory log-linear buckets, 16 sub-buckets per magnitude, bounded 6.25% relative error, exact mean. |
| `Metrics` | Counters plus queue-delay and latency percentiles, JSON serialisation. |
| `RuntimeConfig` | Validation at construction, including the queue-smaller-than-batch trap. |

### CUDA — `cuda/`

| Component | Description |
| --- | --- |
| Reduction | Naive atomic, shared-memory tree, warp-shuffle; plus a row-wise variant. |
| Softmax | Naive, shared-memory, online recurrence, FP16; capacity-aware fallback. |
| RMSNorm | Scalar, `float4` vectorised with alignment checks and fallback, FP16 with FP32 accumulation. |
| LoRA linear | Tiled matmul with bank-conflict padding; unfused and fused paths. |
| Quantisation | Block-wise symmetric INT8 quantise, dequantise and fake-quantise. |
| `GpuScheduler` | Round-robin stream leases, event-based cross-stream chaining, async copies, per-stream accounting. |
| RAII layer | `CudaStream`, `CudaEvent` (timing vs ordering), `DeviceBuffer`, `PinnedBuffer`. |
| Error handling | `CudaError` with status code and sticky-vs-recoverable classification; `CUDAFORGE_CHECK` on every call. |
| Backends | `DeviceAllocatorBackend`, `PinnedAllocatorBackend` for `MemoryPool`. |

### PyTorch integration — `cpp/src/bindings.cpp`, `python/cudaforge/`

Operators registered through `TORCH_LIBRARY` with separate CPU and CUDA
implementations, so the fallback is a real dispatch path rather than a Python
if-statement. Shape, dtype, device and contiguity validation with actionable
messages. `backend_report()` reports which path is active.

### Training — `training/`

Explicit loop with gradient accumulation, mixed precision, correct
unscale-then-clip ordering, OneCycle scheduling, deterministic seeding,
adapter-only checkpointing with run metadata. From-scratch `LoRALinear` plus
PEFT integration. Packed causal-LM dataset with EOS separators. Token-weighted
perplexity evaluation. Three configs (CPU-runnable, single-GPU, QLoRA).

### Inference — `python/cudaforge/`, `inference/`

Concurrent engine with future-based results, load shedding, warmup, graceful
shutdown that settles every outstanding future. FastAPI server with `/health`,
`/metrics`, `/generate`, boundary validation, and an event loop that stays free
during generation.

### Supporting

Benchmarks (4 C++/Python harnesses plus a CUDA one), 4 examples, 7 scripts, a
custom CUDA structural linter, 3 CI workflows, multi-stage CUDA Dockerfile with
5 compose services, pre-commit config, and 11 documents.

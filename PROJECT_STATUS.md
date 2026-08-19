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
| `DynamicBatcher` | Single-threaded batch formation. Closes on `max_batch_size` or on the *oldest* request's deadline. Drops requests past their own deadline. Handler exceptions isolated. |
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

Concurrent engine with future-based results, load shedding, deadline-aware
dropping checked both at dequeue and before execution, warmup, and a graceful
shutdown that settles every outstanding future. FastAPI server with `/health`,
`/metrics`, `/generate`, boundary validation, and an event loop that stays free
during generation.

### Supporting

Benchmarks (5 C++/Python harnesses plus a CUDA one), 5 examples, 7 scripts, a
custom CUDA structural linter, a Markdown link checker, 3 CI workflows, a
multi-stage CUDA Dockerfile with 5 compose services, pre-commit config, a
Makefile, and 14 documents.

---

## Tested locally — actually executed on this machine

| Check | Result |
| --- | --- |
| C++ build (clang 21, C++20, `-Werror`) | **pass**, zero warnings |
| C++ test suite | **114 cases, 28,567 assertions — pass** |
| C++ under ThreadSanitizer | **pass**, no races reported |
| C++ under AddressSanitizer | **pass** |
| C++ under UndefinedBehaviorSanitizer | **pass** |
| Python test suite | **288 tests — pass** |
| PyTorch extension build (CPU-only) | **pass** — the extension compiles and loads |
| C++ CPU operators vs Python references | **exact match** on rmsnorm, softmax, lora, sum, quantise |
| INT8 round trip | max error 0.01104 against a bound of 0.01114 — **within `scale/2`** |
| LoRA fine-tune, end to end | **pass** — 64/102,778 trainable (0.062%), 2 steps, eval, checkpoint |
| C++ benchmarks (queue, scheduler, memory, histogram) | **run** |
| Python benchmarks (operators, batching) | **run** |
| Engine under concurrent load | **run** — 160 requests, avg batch 8.0 |
| `ruff check` / `ruff format --check` | **clean** |
| `mypy` | **clean**, 8 source files |
| `clang-format --dry-run --Werror` | **clean**, 46 files |
| `shellcheck scripts/*.sh` | **clean** |
| CUDA structural checks | **clean**, 22 files |

### Measured results

Apple M5 Pro, simulated executor. These describe the **scheduler**, not model
throughput.

| Measurement | Result |
| --- | --- |
| Batching, 16 clients, `max_batch_size` 1 → 16 | 361 → 1,269 req/s (**3.52×**), p50 and p99 unchanged |
| Batching, 8 clients, `max_batch_size` 1 → 16 | 499 → 722 req/s, p99 12.89 → 7.88 ms |
| Batching, 1 client, `max_batch_size` 1 → 16 | 112 → 87 req/s — the latency cost when there is nothing to batch |
| Memory pool | 2,020 allocations, 5 backend calls, reuse rate 0.9975 |
| Latency histogram | worst error **4.95%** against a documented 6.25% bound, over four distributions |
| Histogram record rate | 76–120M samples/second |

### Bugs found and fixed during development

Recorded because each was found by a test that was written to look for it:

1. **Futures never settled on a row-count mismatch.** A runner returning fewer
   results than requests raised outside the engine's guard, so no future was
   completed and every caller blocked until timeout.
2. **Catch2 assertions on worker threads.** Not thread-safe; produced a sporadic
   `SIGABRT` under ASan whose timing depended on the scheduler.
3. **`min_block_bytes` ignored** by the pool's size-class selection.
4. **Blocking on a full queue while posting the shutdown sentinel**, which could
   deadlock the only thread able to drain it.
5. **`O(n)` histogram eviction** from a list front, on the per-request path.
6. **19 late-binding closures** in the benchmark harness, each capturing the
   loop variable rather than its value.
7. **Missing includes** for `<chrono>`, `<cstdint>`, `<cstddef>`, `<stdexcept>`.
8. **Deadline checks in the wrong place.** Checking only at batcher dequeue
   dropped nothing under load, because the backlog accumulates behind the worker
   pool rather than in the request queue. Found by a test that saturated the
   executor and expected drops.
9. **Wrong anchor slugs in the documentation checker**, which collapsed runs of
   whitespace where GitHub emits one hyphen per space.
10. **Line numbers off in the CUDA checker**, which attributed a statement to a
    preceding blank or comment line.

---

## Requires NVIDIA hardware — implemented, not executed

Nothing below has run. It is written, reviewed, structurally checked, and unit
tested against references that *did* run — but no line of it has been compiled
by `nvcc` or executed on a GPU as part of this work.

| Component | What is unverified |
| --- | --- |
| All `.cu` kernels | compilation and execution |
| `tests/cuda/*` (30+ cases) | every assertion in them |
| `GpuScheduler` | runtime stream overlap and event ordering |
| `MemoryPool<DeviceAllocatorBackend>` | device allocation behaviour |
| `PinnedBuffer` | page-locked allocation and DMA overlap |
| PyTorch CUDA extension | the `CUDAExtension` build path |
| `benchmarks/benchmark_kernels.cu` | every number it would produce |
| Nsight profiles | all of them |
| QLoRA / bitsandbytes | the 4-bit path |
| `examples/distributed_train.py` | DDP and NCCL |
| Dockerfile and compose services | image build and GPU passthrough |

**No GPU performance number was fabricated.** Where one would normally appear,
this repository states that it was not measured.

## Known limitations

Real ones, not hedges:

1. **The tiled matmul is not competitive with cuBLAS.** It exists to make the
   shared-memory argument concrete. Production code should call cuBLAS.
2. **The memory pool is not stream-ordered.** A buffer must not be freed while a
   kernel using it is in flight. `cudaMallocAsync` solves this; this pool does
   not.
3. **The INT8 kernel is not NF4.** It is uniform symmetric INT8. The QLoRA path
   calls bitsandbytes directly.
4. **No KV cache.** Each request re-runs its prompt. This is the largest
   available performance win and is not implemented.
5. **Batches are static once formed.** No continuous batching, so a batch runs
   until its longest member finishes.
6. **Single-device serving.** DDP covers training only.
7. **`benchmark_kernels.py` on a CPU host compares PyTorch to PyTorch.** The
   output says so, in the results file.
8. **Two `type: ignore` comments** in `runners.py`, both for inconsistencies in
   the transformers 5.x stubs, both documented at the site.

## Recommended GPU validation

On a Linux machine with an NVIDIA GPU and CUDA 12.x:

```bash
git clone https://github.com/atkunja/cuda_force.git && cd cuda_force
./scripts/setup.sh && source .venv/bin/activate

# 1. Does it build?
./scripts/build.sh --cuda

# 2. Are the kernels correct? This is the important one — every kernel is
#    compared against a double-precision host reference over awkward shapes.
./build-cuda/tests/cuda/cudaforge_cuda_tests

# 3. Does the extension build and dispatch to CUDA?
pip install -e . --no-build-isolation
python -c "from cudaforge.ops import backend_report; print(backend_report())"
python examples/kernel_parity.py   # every operator against its reference
pytest tests/python -q             # cuda-marked tests now run instead of skipping

# 4. What are the numbers?
./build-cuda/benchmarks/bench_kernels > benchmarks/results/cuda-kernels.json

# 5. Where does the time actually go?
./scripts/profile.sh

# 6. Does it train?
python -m training.train --config training/configs/lora_gpt2.yaml
```

Step 2 is the one that matters. If those assertions pass, the kernels compute
what they claim to compute; everything after that is performance.

## Future improvements

Ordered by expected value:

1. **Paged KV cache** — the largest single win, and the main thing between this
   and a serious serving runtime.
2. **Continuous batching** — admit new requests mid-generation.
3. **Stream-ordered allocation** — adopt `cudaMallocAsync` semantics and remove
   limitation 2.
4. **Tensor-core matmul** — via CUTLASS rather than by hand.
5. **Speculative decoding.**
6. **Tensor-parallel inference** across GPUs.
7. **Persistent kernels** for the small elementwise ops.
8. **GPU-measured benchmarks** — the harness is complete; only hardware is
   missing.

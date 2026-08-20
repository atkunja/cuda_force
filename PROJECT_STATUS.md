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
| `BlockAllocator` / `SequenceBlockTable` | Paged KV cache bookkeeping: reference-counted blocks, per-sequence block tables, copy-on-write for shared prefixes, exhaustion as a value rather than an exception. |
| `RuntimeConfig` | Validation at construction, including the queue-smaller-than-batch trap. |

### CUDA — `cuda/`

| Component | Description |
| --- | --- |
| Reduction | Naive atomic, shared-memory tree, warp-shuffle; plus a row-wise variant. |
| Softmax | Naive, shared-memory, online recurrence, FP16 and BF16; capacity-aware fallback. |
| RMSNorm | Scalar, `float4` vectorised with alignment checks and fallback, plus FP16 and BF16 with FP32 accumulation. |
| LoRA linear | Tiled matmul with bank-conflict padding; unfused and fused paths. |
| Activations | SiLU, GELU (tanh form), and SwiGLU in scalar, vectorised, FP16 and BF16 paths. |
| Fused residual + RMSNorm | Single-pass residual add and normalisation, emitting both the normalised output and the carried residual. |
| Quantisation | Block-wise symmetric INT8 quantise, dequantise and fake-quantise. |
| `GpuScheduler` | Round-robin stream leases, event-based cross-stream chaining, async copies, per-stream accounting. |
| RAII layer | `CudaStream`, `CudaEvent` (timing vs ordering), `DeviceBuffer`, `PinnedBuffer`. |
| Error handling | `CudaError` with status code and sticky-vs-recoverable classification; `CUDAFORGE_CHECK` on every call. |
| Backends | `DeviceAllocatorBackend`, `PinnedAllocatorBackend` for `MemoryPool`. |

### PyTorch integration — `cpp/src/bindings.cpp`, `python/cudaforge/`

Nine operators registered through `TORCH_LIBRARY` with separate CPU and CUDA
implementations, so the fallback is a real dispatch path rather than a Python
if-statement. An explicit Autograd registration turns "silently wrong
gradients" into an immediate error, and the Python layer routes to the
differentiable reference whenever a backward pass is expected. Shape, dtype, device and contiguity validation with actionable
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

Ten benchmark harnesses (C++, Python and CUDA) plus a Markdown result
summariser, 6 examples, 8 scripts, a custom CUDA structural linter, a
Markdown link checker, 3 CI workflows, a multi-stage CUDA Dockerfile with 5
compose services, pre-commit config, a Makefile, and 17 documents.

---

## Tested locally — actually executed on this machine

| Check | Result |
| --- | --- |
| C++ build (clang 21, C++20, `-Werror`) | **pass**, zero warnings |
| C++ test suite | **164 cases, 49,876 assertions — pass** |
| C++ under ThreadSanitizer | **pass**, no races reported |
| C++ under AddressSanitizer | **pass** |
| C++ under UndefinedBehaviorSanitizer | **pass** |
| Python test suite | **450 tests — pass**, 91% statement / 88% branch coverage |
| PyTorch extension build (CPU-only) | **pass** — the extension compiles and loads |
| C++ CPU operators vs Python references | **exact match** on rmsnorm, softmax, lora, sum, quantise |
| INT8 round trip | max error 0.01104 against a bound of 0.01114 — **within `scale/2`** |
| LoRA fine-tune, end to end | **pass** — 64/102,778 trainable (0.062%), 2 steps, eval, checkpoint |
| C++ benchmarks (queue, scheduler, memory, histogram) | **run** |
| Python benchmarks (operators, batching) | **run** |
| Engine under concurrent load | **run** — 160 requests, avg batch 8.0 |
| HTTP server under load | **run** — 300 requests at concurrency 32, no failures |
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
| C++ scheduler under saturation, batch 1 → 32 | 1,542 → 16,844 req/s (10.9×), p99 queue delay 671 → 65 ms |
| Memory pool | 2,020 allocations, 5 backend calls, reuse rate 0.9975 |
| HTTP end to end | 300 requests at concurrency 32: 420 req/s, 0 failures; client p99 279 ms against server p99 1.03 ms |
| Paged KV cache occupancy | 13.2× more sequences than contiguous on chat-shaped traffic (25.8× on short prompts, 1.0× when every sequence hits the limit) |
| Block allocator throughput | 163–180M operations/second |
| Bounded queue scaling | 2.39M items/s at 1×1, falling to 774k at 8×8 — where the single mutex becomes the bottleneck |
| Latency histogram (C++) | worst error **4.95%** against a documented 6.25% bound, over four distributions |
| Latency histogram (Python) | 16.38 µs → 1.76 µs per record after retuning the window; 3.25 µs per request total |
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
11. **No autograd kernel registered for the custom operators.** PyTorch's
    default is to warn once and then produce silently incorrect gradients — the
    worst available failure mode, since training converges to the wrong thing
    with nothing pointing at the cause. Surfaced by a warning in a test that
    called `.backward()`. Now an explicit error, with the Python layer routing
    gradient-requiring calls to the differentiable reference.
12. **`serve` ignored its own `--config` file**, passing the flag defaults to
    the server instead of the resolved values — so a config file changed the
    benchmark's behaviour but not the server's.
13. **Test pollution through the process environment.** `cli.serve` configures
    the server by writing `CUDAFORGE_*` variables, and those leaked into later
    tests; the failure appeared only when the whole suite ran, not when the
    test did.
14. **A shared-memory race in the block reductions.** `block_reduce_max`
    returned `shared[0]` directly, so a caller reusing the array for a second
    reduction — softmax does exactly that, max then sum — could have one warp
    overwrite the slot while another had not yet read it. Found by re-reading
    the CUDA sources, not by a test: the code cannot be compiled on this host,
    and the symptom would have been a silently wrong row rather than a crash.
    Fixed by reading into a register with a trailing barrier, and guarded by a
    new structural rule.
15. **Shared-memory capacity checks ignored the static scratch array**, so a
    launch could be accepted at exactly the row width where the dynamic request
    alone just fit.
16. **A 32-bit index overflow in the elementwise kernels.** The index was
    computed as `blockIdx.x * blockDim.x + threadIdx.x` and cast to `int`; near
    `INT_MAX` elements the product exceeds `INT_MAX` and the cast is undefined.
    Now computed in `size_t` throughout.
17. **Unbounded device buffer copies.** `copy_from_host` took a count and did
    not check it against the allocation. A device-side overrun does not fault
    at the copy — it corrupts the next allocation and surfaces elsewhere.
18. **`inference` and `training` were never installed as packages.** Only
    `cudaforge` was, so `cudaforge-serve` — which starts uvicorn with the import
    string `"inference.server:app"` — worked solely when the current directory
    happened to be the repository root. The same gap made a bare `pytest` fail
    while `python -m pytest` passed, because the latter puts the working
    directory on `sys.path`.
19. **Three undeclared dependencies.** `pyyaml`, `httpx` and `httpx2` were
    present on the development machine and missing from the extras that needed
    them, so a clean install failed 16 tests that passed locally.
20. **An unused variable that only Linux clang rejects.** The local build did
    not run with `-Werror`, and Apple clang does not warn for an unused object
    with a non-trivial constructor.
21. **Three different clang-format versions** in play — Homebrew's locally,
    apt's in CI, and a third pinned in pre-commit — which disagreed about the
    same file. Now one pinned version everywhere.
22. **A CMake ordering bug reachable only with CUDA enabled.** `tests/cuda` was
    added from `cuda/CMakeLists.txt`, before `tests/cpp` had fetched Catch2 and
    defined `catch_discover_tests`.
23. **`tomllib` used on Python 3.10**, where it is not in the standard library,
    despite the package declaring 3.10 as its floor.
24. **A half-applied template conversion.** The BF16 change converted the SwiGLU
    kernel's *body* to use the `Convert` alias but not its signature, leaving
    `Convert::` with nothing behind it. It compiles nowhere, but the development
    host has no nvcc, so only the CUDA CI job caught it. Now covered by a
    structural rule.
25. **The CUDA targets were built as C++17** while the portable headers they
    include are C++20. The errors pointed at `memory_pool.hpp` — a file that
    compiles fine in the portable build — and mentioned concepts and
    `std::bit_ceil` rather than the standard, which made it read as a header bug
    rather than a flag one.
26. **The sanitizer was not applied to the benchmark targets**, so they linked
    an instrumented runtime without the sanitizer runtime and the whole
    ThreadSanitizer build failed to link. The test target applied it and kept
    passing, which is why this only surfaced when `scripts/test.sh` was run
    end to end rather than stage by stage.

---

## Requires NVIDIA hardware — implemented, not executed

Nothing below has run. It is written, reviewed, structurally checked, and unit
tested against references that *did* run — but no line of it has been compiled
by `nvcc` or executed on a GPU as part of this work.

| Component | What is unverified |
| --- | --- |
| All `.cu` kernels | compilation and execution |
| `tests/cuda/*` (69 cases) | every assertion in them |
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
4. **No KV cache in the serving path.** The paged block allocator is built and
   tested, but nothing reads through it: the attention gather that would make
   it useful is not written, and no preemption policy decides which sequence to
   evict. Each request still re-runs its prompt.
5. **Batches are static once formed.** No continuous batching, so a batch runs
   until its longest member finishes.
6. **Single-device serving.** DDP covers training only.
7. **`benchmark_kernels.py` on a CPU host compares PyTorch to PyTorch.** The
   output says so, in the results file.
8. **Two `type: ignore` comments** in `runners.py`, both for inconsistencies in
   the transformers 5.x stubs, both documented at the site.
9. **BF16 needs compute capability 8.0.** Below it bfloat16 is emulated, so the
   BF16 path would be slower than the FP16 one it replaces. The default
   architecture list starts at 80 and CMake warns if it is lowered.

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

The suite covers seven kernel families — reduction, softmax, RMSNorm, LoRA
linear, quantisation, activations (SiLU/GELU/SwiGLU) and the fused residual +
RMSNorm — plus the stream scheduler, the device memory pool and the RAII
wrappers. Each is compared against a double-precision host reference over shapes
that include non-power-of-two, very small, very large, and widths that defeat
the vectorised paths so the scalar fallbacks are exercised rather than assumed.

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

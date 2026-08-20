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
> CUDA path, so for most of its life nothing CUDA had ever executed. On
> 2026-08-20 the whole suite was run on a rented **RTX 3090** (driver 580.126.09,
> CUDA 12.8): every kernel executed, **19,511 assertions passed across 69 test
> cases**, and the figures in
> [GPU kernel measurements](#gpu-kernel-measurements) are real. Everything else
> is host-side and labelled with the machine that produced it. See
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

That is the **static** path: a batch is formed, run to completion, and replaced.
The runtime also ships an **iteration-level** path, where `ContinuousBatcher`
stands in for `DynamicBatcher` and refills a row the moment a sequence finishes
rather than waiting for the slowest member of its batch. Both sit behind one
`ServingEngine` protocol, so the HTTP server takes either — `cudaforge-serve
--continuous` picks the second. See
[docs/continuous-batching.md](docs/continuous-batching.md).

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
- FP16 *and* BF16 paths from one templated kernel body (BF16 needs SM 8.0+)
- Explicit autograd guard: no silently wrong gradients
- CUDA streams and events; **zero** `cudaDeviceSynchronize`
- Pinned host memory for genuine async copies
- Caching device allocator over size classes
- RAII for every stream, event and allocation
- Typed errors distinguishing recoverable from sticky

</td><td valign="top" width="50%">

**C++20 / systems**

- Paged KV cache: block allocator, refcounted prefix sharing, eviction policy
- Continuous batching: iteration-level scheduling — **70% fewer decode
  steps, 1.44x wall-clock** on a real transformer at batch 32
- Bounded MPMC queue: mutex + condition variables
- Predicate waits — no bare `wait`, no spurious-wakeup bugs
- Thread pool with futures and graceful drain
- Relaxed atomics for hot-path counters
- Backpressure, load shedding, and deadline-aware dropping
- Idempotent shutdown that never loses accepted work
- Fixed-memory log-linear latency histogram
- Clean under TSan, ASan and UBSan

**ML**

- Speculative decoding: draft-and-verify, **lossless** — the accepted tokens
  come from the target's own distribution, tested against it
- LoRA from scratch + PEFT integration
- QLoRA via bitsandbytes NF4 (scope stated, not overclaimed)
- Block-wise INT8 quantisation with a proven error bound
- Gradient accumulation, checkpointing, mixed precision
- Dynamic batching, DDP/NCCL data parallelism

</td></tr>
</table>

## The kernels

Each ships with a reference implementation, correctness tests over awkward
shapes, and a benchmark harness.

| Kernel | Variants | The interesting part |
| --- | --- | --- |
| **Reduction** | atomic · shared-memory tree · warp shuffle | why one `atomicAdd` per element serialises the whole grid |
| **Softmax** | naive · shared-memory · online | the online (max, sum) recurrence — no row-length limit, one fewer pass |
| **RMSNorm** | scalar · `float4` · FP16 | 128-bit transactions, plus alignment checks with a real fallback |
| **LoRA linear** | unfused · fused | keeping the `batch × rank` intermediate out of global memory |
| **Activations** | SiLU · GELU · SwiGLU, scalar and `float4` | fusing `silu(gate) * up` to the minimum possible memory traffic |
| **Residual + RMSNorm** | fused | the highest-frequency fusion in inference — twice per layer, every layer |
| **Quantise** | block-wise INT8 | round-trip error provably under `scale / 2` |

Full reasoning in [docs/cuda-kernels.md](docs/cuda-kernels.md).

## Quick start

```bash
git clone https://github.com/atkunja/cuda_force.git
cd cuda_force
./scripts/setup.sh
source .venv/bin/activate
```

`setup.sh` installs a CPU-only PyTorch if no CUDA toolkit is present, and skips
bitsandbytes off Linux. The package remains importable and fully testable
either way.

```bash
# What this machine can actually build and run
python scripts/environment_report.py

# Which implementation path is active
python -c "from cudaforge.ops import backend_report; print(backend_report())"

# One request end to end
python examples/simple_inference.py --echo-runner

# What dynamic batching buys
python examples/concurrent_requests.py

# The same load under iteration-level scheduling, for comparison
python -c "from cudaforge.cli import bench; bench(['--echo-runner','--continuous'])"

# A real LoRA fine-tune, CPU, under a minute, no downloads beyond the model
python examples/fine_tune.py

# Every kernel against its reference — the first thing to run on new GPU hardware
python examples/kernel_parity.py

# The operators composed into a LLaMA-style block, checked against PyTorch
python examples/transformer_block.py

# Speculative decoding against ordinary decoding — same tokens, fewer target calls
python examples/speculative_decoding.py
```

### Using the operators

```python
import torch
import cudaforge

x = torch.randn(8, 4096, device="cuda")
w = torch.ones(4096, device="cuda")

y = cudaforge.rmsnorm(x, w, eps=1e-6)
p = cudaforge.softmax(x)
z = cudaforge.lora_linear(x, weight, lora_a, lora_b, scale=2.0)

q, scales = cudaforge.quantize_int8(x)
restored = cudaforge.dequantize_int8(q, scales)
```

On a machine without CUDA these dispatch to reference PyTorch implementations
and return identical results. `cudaforge.backend_report()` says which path ran —
worth checking before drawing any conclusion from a timing.

### Serving

```bash
cudaforge-serve --model gpt2 --max-batch-size 16 --max-wait-us 5000

# Or from a config, with flags overriding individual values
cudaforge-serve --config inference/configs/balanced.yaml
```

| Config | Shape |
| --- | --- |
| [`latency.yaml`](inference/configs/latency.yaml) | small batch, short wait — interactive traffic where p99 is what users feel |
| [`throughput.yaml`](inference/configs/throughput.yaml) | large batch, generous wait — batch traffic where tokens/second is what matters |
| [`balanced.yaml`](inference/configs/balanced.yaml) | where to start before measuring |

```bash
curl -s localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"prompt": "Explain CUDA warps.", "max_new_tokens": 64, "temperature": 0.7}'
```

```json
{
  "request_id": "3f9a1c0e7b2d4a58",
  "text": "...",
  "prompt_tokens": 4,
  "generated_tokens": 64,
  "queue_time_ms": 1.83,
  "inference_time_ms": 42.11,
  "total_latency_ms": 43.94,
  "batch_size": 7
}
```

Pass `deadline_seconds` to have the runtime drop the request instead of
executing it if it is still queued past that point — under load, work nobody is
waiting for displaces work someone is. Dropped requests return 503, and are
counted separately from rejections and failures.

Queue time and inference time are reported separately because they point at
different problems: high queue time means the runtime is saturated or the wait
is too generous, high inference time means the model or the batch size is the
constraint.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | liveness — a failure here means restart the process |
| `GET /ready` | readiness — 503 when the queue is near capacity, meaning take it out of rotation, not restart it |
| `GET /metrics` | counters, batching statistics and latency percentiles, as JSON |
| `GET /metrics/prometheus` | the same snapshot in the Prometheus text exposition format |
| `POST /generate` | generation |

## Building the C++ and CUDA targets

```bash
./scripts/build.sh                      # portable runtime; no CUDA needed
./scripts/build.sh --cuda               # adds the kernels (requires nvcc)
./scripts/build.sh --sanitizer thread   # TSan build
./scripts/test.sh                       # everything runnable here, skips reported
```

The portable targets never depend on CUDA. That split is deliberate: the
concurrency runtime, where most of the subtle correctness lives, gets full test
and sanitizer coverage regardless of what hardware is present.

### On CUDA hardware

Everything that needs a GPU, in one command — preflighted, logged, and written
up as a Markdown report:

```bash
./scripts/validate_gpu.sh
```

Or the stages individually:

```bash
./scripts/build.sh --cuda
./build-cuda/tests/cuda/cudaforge_cuda_tests      # the one that matters
./build-cuda/benchmarks/bench_kernels > benchmarks/results/cuda-kernels.json
./scripts/profile.sh
```

Or through Docker, which needs an NVIDIA driver and the NVIDIA Container
Toolkit:

```bash
docker compose run --rm test
docker compose run --rm bench
docker compose up serve
```

### Developing on macOS

CUDA kernels cannot compile or run on Apple Silicon, and nothing here pretends
otherwise. What does work locally:

| Works | Does not |
| --- | --- |
| The whole C++20 concurrency runtime, plus sanitizers | `nvcc`, any `.cu` compilation |
| The Python package and its reference operators | CUDA kernel execution |
| The inference engine, server and batching benchmarks | GPU benchmarks and Nsight profiles |
| LoRA fine-tuning on CPU or MPS | QLoRA (bitsandbytes is Linux/CUDA only) |
| CUDA structural checks | — |

The PyTorch extension still builds — as a **CPU-only** extension — so the
dispatcher registration and every CPU operator implementation are exercised
locally.

## Fine-tuning

```bash
python -m training.train --config training/configs/tiny.yaml      # CPU, ~1 min
python -m training.train --config training/configs/lora_gpt2.yaml # 16 GB GPU
python -m training.train --config training/configs/qlora_7b.yaml  # 24 GB GPU
```

The training loop is written out rather than delegated to `Trainer`, so
gradient accumulation, loss-scaling order and scheduler timing are explicit.
`training/lora.py` also contains a from-scratch `LoRALinear` used as the
reference the CUDA kernel is validated against.

Details, including why `alpha` is not a second learning rate and why `B` must
start at zero, in [docs/fine-tuning.md](docs/fine-tuning.md).

## Benchmarks

```bash
./scripts/benchmark.sh          # everything runnable here; skips are reported
```

Results are written to `benchmarks/results/` and are **not** committed —
committed numbers would be numbers from someone else's machine.

What has been measured, on an Apple M5 Pro. Execution is simulated except
where a row says otherwise:

| Measurement | Result |
| --- | --- |
| Batching, 16 clients, batch 1 → 16 | **3.52× throughput at unchanged p50 and p99** |
| Batching under saturation, batch 1 → 32 | **10.9× throughput**, and p99 queue delay 671 ms → 65 ms |
| Bounded queue | 2.39M items/s at 1×1, 774k at 8×8 — where the single mutex binds |
| Batching, 8 clients, batch 1 → 16 | 499 → 722 req/s, p99 12.9 → 7.9 ms |
| Batching, 1 client, batch 1 → 16 | 112 → 87 req/s — the latency cost, shown honestly |
| Memory pool | 2,020 allocations served by 5 backend calls; reuse rate 0.9975 |
| Latency histogram | worst error **4.95%** vs its documented 6.25% bound, at 76–120M records/s |
| HTTP end to end | 300 requests at concurrency 32: 420 req/s, client p99 279 ms vs server p99 1.03 ms |
| Paged KV cache | **13.2× more concurrent sequences** than contiguous on chat-shaped traffic; waste 93% → 4.8% |
| Continuous batching | **62% fewer decode steps** on long-tailed traffic; utilisation 30% → 78% (−2% when lengths are constant) |
| Continuous batching, **real transformer** | 70% fewer decode steps and **1.44× wall-clock** at batch 32 — and 0.83× at batch 4, where refilling rows one at a time fragments prefill |
| Speculative decoding | Tokens per target call tracks the closed form `(1 − a^(k+1))/(1 − a)` across acceptance 0.3–0.9; lossless by construction, checked against the target's own distribution |
| C++ suite | 183 cases, 49,983 assertions, clean under TSan / ASan / UBSan |
| Python suite | 570 tests, 95% statement / 85% branch coverage |

Most of these measure the **scheduler**, not model throughput: execution is
simulated so the variable under study is isolated and the benchmark runs without
a GPU. The two rows marked *real transformer* are the exception — they drive an
actual model, one token at a time, over a KV cache that is resized as sequences
join and leave. Its weights are random, so the generated text is meaningless;
nothing measured depends on it.

**No CUDA kernel number exists anywhere.** The harness is complete and runs
unchanged on NVIDIA hardware — `scripts/validate_gpu.sh` is the one command.
See [docs/benchmarking.md](docs/benchmarking.md).

## GPU kernel measurements

Run on a rented **NVIDIA RTX 3090** (compute capability 8.6, 82 SMs, theoretical
bandwidth 936 GB/s), driver 580.126.09, CUDA 12.8, on 2026-08-20 — the first and
so far only execution of this CUDA code. Reproduce with
`./scripts/validate_gpu.sh`.

`of peak` is the number that matters. A kernel near the device's bandwidth is
finished; getting further needs an algorithmic change, not more tuning.

### The reduction hierarchy, which is the whole argument

| elements | naive | shared memory | warp shuffle |
| -------: | ----: | ------------: | -----------: |
|      64K | 0.126 ms | 0.0082 ms | 0.0082 ms |
|       1M | 1.913 ms | 0.0113 ms | 0.0102 ms |
|      16M | **26.67 ms** | 0.0932 ms | **0.0922 ms** |

**289x** at 16M, and the optimised variants reach **78% of peak**. One
`atomicAdd` per element serialises the entire grid; that is what the first
column costs.

The shuffle beats shared memory by **1%** at 16M and 9.7% at 1M. Both are
bandwidth-bound, so the shuffle is nearly free and — at scale — nearly
pointless. It is in the repository because the technique is worth knowing, not
because it wins here.

### Everything else

| kernel | variant | shape | median ms | GB/s | of peak |
| --- | --- | --- | --- | --- | --- |
| softmax | naive | 128x8192 | 0.0164 | 512.0 | 55% |
| softmax | online | 128x8192 | 0.0154 | 546.1 | **58%** |
| rmsnorm | scalar | 4096x1024 | 0.0911 | 368.2 | 39% |
| rmsnorm | float4 | 4096x1024 | 0.0440 | 762.0 | **81%** |
| rmsnorm | scalar | 2048x4096 | 0.0829 | 809.1 | 86% |
| rmsnorm | float4 | 2048x4096 | 0.0829 | 809.1 | 86% |
| silu | scalar | 16M | 0.1567 | 856.8 | 92% |
| swiglu | scalar | 16M | 0.2294 | 877.7 | **94%** |
| fused_residual_rmsnorm | fused | 2048x4096 | 0.1597 | 840.2 | 90% |
| fused_residual_rmsnorm | separate | 2048x4096 | 0.1997 | 840.2 | 90% |
| quantize_int8 | blockwise | 16M | 0.1915 | 438.1 | 47% |
| dequantize_int8 | blockwise | 16M | 0.1126 | 744.7 | 80% |

Three things worth reading off that table.

**`float4` is shape-dependent, not a free win.** RMSNorm at 4096x1024 goes 39% ->
81% of peak, a genuine 2.07x. At 2048x4096 the two variants are *identical*: once
rows are wide enough for scalar access to saturate, vectorising buys nothing.

**The activation kernels are done.** SwiGLU at 94% of theoretical bandwidth has
no tuning left in it.

**Fusion pays, modestly.** Residual+RMSNorm fused is 1.23-1.45x faster than the
separate pair. Both sit at the same GB/s; the fused kernel simply moves less.

### The fused LoRA kernel is slower. Much slower.

| shape | unfused | fused | |
| --- | ---: | ---: | ---: |
| 32x1024x1024 r8 | 0.0850 ms | 1.8493 ms | **21.8x slower** |
| 64x2048x2048 r16 | 0.3640 ms | 11.759 ms | **32.3x slower** |
| 128x4096x4096 r16 | 2.0562 ms | 46.666 ms | **22.7x slower** |

The fusion does what it claims — the `batch x rank` intermediate never reaches
global memory. It still loses, and not for the reason it first appears.

Neither path uses cuBLAS. The unfused one calls this project's own
`matmul_tiled`, which stages 16x16 tiles in shared memory and reuses every
loaded element sixteen times. The fused kernel's second phase does **no tiling
at all**: each thread walks the whole of `in_features`, reading `x` and `w`
straight from global memory with no staging and no reuse.

So the comparison is not fused against unfused. It is untiled against tiled, and
the fusion is what forced the untiling — holding `X A` in shared memory left no
room to tile the frozen path beside it. The intermediate saved is a few
`batch x rank` floats; the reuse given up is 16x on a `batch x in x out` matmul.

Use the unfused path. The fix is to tile the fused kernel's second phase, not to
reach for tensor cores.

## Testing

```bash
./scripts/test.sh
```

| Suite | Scope |
| --- | --- |
| `tests/cpp` | queue, pool, batcher, metrics, histogram, config — including stress and shutdown |
| `tests/cpp` under 3 sanitizers | TSan, ASan, UBSan — all clean |
| `tests/python` | operators, batching, engine, HTTP, LoRA, dataset, training |
| `tests/cuda` | kernel parity against double-precision references (needs a GPU) |
| `scripts/check_cuda_sources.py` | structural CUDA rules, no toolkit required |

A few of these exist to pin down properties that would otherwise regress
silently: that RMSNorm survives FP16 magnitudes which overflow when squared,
that an INT8 outlier degrades only its own block, that merging a LoRA layer is
*exact*, and that gradient accumulation produces the same gradient as one large
step. More in [docs/testing.md](docs/testing.md).

Since a machine without a GPU cannot compile the kernels,
`scripts/check_cuda_sources.py` enforces what can be checked statically:
unchecked launches, discarded CUDA statuses, `cudaDeviceSynchronize`, maskless
warp shuffles, and conditionally-reached `__syncthreads()`.

## Documentation

| Document | Covers |
| --- | --- |
| [api.md](docs/api.md) | the public surface and what each part guarantees |
| [architecture.md](docs/architecture.md) | layering, request lifecycle, failure isolation, threading model |
| [concurrency.md](docs/concurrency.md) | mutexes vs atomics, condvars vs polling, backpressure, batching policy |
| [cuda-kernels.md](docs/cuda-kernels.md) | every kernel, its variants, and what each optimisation removes |
| [gpu-execution.md](docs/gpu-execution.md) | streams, events, the three conditions for overlap, warp execution |
| [memory-management.md](docs/memory-management.md) | why `cudaMalloc` stalls, size classes, pinned memory, allocator comparison |
| [fine-tuning.md](docs/fine-tuning.md) | LoRA memory arithmetic, packing, mixed precision, QLoRA scope |
| [benchmarking.md](docs/benchmarking.md) | method, what to measure, how to read each suite |
| [profiling.md](docs/profiling.md) | Nsight Systems and Compute, what each counter means |
| [performance.md](docs/performance.md) | every optimisation as baseline → bottleneck → change → measured |
| [kv-cache.md](docs/kv-cache.md) | why contiguous caches waste memory, and what paging changes |
| [continuous-batching.md](docs/continuous-batching.md) | iteration-level scheduling, eviction policy, and what each recovers |
| [speculative-decoding.md](docs/speculative-decoding.md) | why draft-and-verify is lossless, and what tokens-per-target-call does and does not promise |
| [testing.md](docs/testing.md) | what each suite asserts and why |
| [deployment.md](docs/deployment.md) | containers, probes, graceful shutdown, sizing, alerts |
| [troubleshooting.md](docs/troubleshooting.md) | symptoms, diagnoses and fixes |
| [concepts-index.md](docs/concepts-index.md) | every concept mapped to the file and function that implements it |
| [environment.md](docs/environment.md) | the development host and exactly what it could not run |

## Repository layout

```
cpp/       portable C++20 runtime — queue, pool, batcher, metrics, memory pool
cuda/      kernels, stream scheduler, device allocator backends
python/    cudaforge package — ops dispatch, engine, batcher, metrics, CLI
training/  LoRA/QLoRA pipeline, configs, from-scratch reference layer
inference/ FastAPI server and request schemas
tests/     cpp · python · cuda
benchmarks/ C++ and Python harnesses; results are generated, never committed
scripts/   setup · build · test · benchmark · profile · lint · CUDA checks
docs/      the reasoning behind all of the above
```

## Roadmap

Honest about what is not here:

Legend: `[x]` done and verified, `[~]` written and compiling in CI but **not
yet executed on a GPU**, `[ ]` not started.

- [x] **Paged KV cache — block allocator.** Reference-counted blocks, per-sequence
      block tables, copy-on-write for shared prefixes. Host-side and fully
      tested; see [kv-cache.md](docs/kv-cache.md).
- [~] **Paged KV cache — attention gather.** Written and compiling:
      `launch_paged_attention` reads K and V through the block table, with
      grouped-query support and an online softmax so context is bounded by the
      cache rather than by shared memory. Tests include one that swaps two
      block-table entries and requires the output to change — without it a
      kernel ignoring the indirection would pass. **Not yet run on a GPU.**
- [x] **Preemption policy.** Newest-first and largest-first eviction, livelock-free
      admission, feasibility decided before anything is destroyed. Host-side and
      fully tested; see [continuous-batching.md](docs/continuous-batching.md).
- [x] **Continuous batching.** Iteration-level scheduling: rows freed by finished
      sequences are refilled at the next decode step. **62% fewer decode steps**
      on long-tailed traffic, utilisation 30% → 78%.
- [ ] **Wire the two together.** The cache manager is C++, the scheduler is
      Python, and connecting them needs the attention gather below.
- [x] **A step-wise runner over a real model.**
      `TransformersStepwiseRunner` drives a transformer one token at a time and
      owns the KV cache, adding and removing rows as sequences join and leave.
      On a 6-layer GPT-2 at batch 32 it turns the scheduling win into **70%
      fewer decode steps and 1.44x wall-clock** — and at batch 4 into a net
      loss, because refilling rows one at a time fragments prefill. Both are in
      [continuous-batching.md](docs/continuous-batching.md#measured-on-a-real-model).
- [ ] **Tensor-core matmul.** The tiled kernel is a teaching implementation and
      is not competitive with cuBLAS, by design.
- [ ] **FP8.** Hopper and later. The `ReducedPrecision` traits are the seam a
      third format would slot into.
- [~] **Stream-ordered allocation.** Written and compiling:
      `StreamOrderedAllocatorBackend` over `cudaMallocAsync`/`cudaFreeAsync`
      orders the free within the stream, so returning a buffer a running kernel
      still reads costs no device-wide stall. **Not yet run on a GPU.**
- [x] **Speculative decoding.** A draft model proposes `k` tokens and the target
      verifies them in one pass. Lossless by construction — greedy matches the
      target token for token, and sampling uses the `min(1, p/q)` rule with a
      residual draw, checked against the target's own distribution. Throughput
      tracks the closed form `(1 - a^(k+1))/(1 - a)`; batch size 1 only. See
      [speculative-decoding.md](docs/speculative-decoding.md).
- [ ] **Multi-GPU inference.** Training has a DDP example; serving is
      single-device.
- [x] **GPU-measured benchmarks.** Run on an RTX 3090 on 2026-08-20: every
      stage passed, 19,511 CUDA assertions, and real kernel numbers are in
      [GPU kernel measurements](#gpu-kernel-measurements). Only hardware was
      missing.

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) has the rules that are not negotiable — no
fabricated measurements, every CUDA call checked, no `cudaDeviceSynchronize`,
portable code stays portable, and new kernels ship with a reference
implementation.

Release history is in [CHANGELOG.md](CHANGELOG.md).

## Licence

MIT — see [LICENSE](LICENSE).

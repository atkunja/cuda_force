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
- Explicit autograd guard: no silently wrong gradients
- CUDA streams and events; **zero** `cudaDeviceSynchronize`
- Pinned host memory for genuine async copies
- Caching device allocator over size classes
- RAII for every stream, event and allocation
- Typed errors distinguishing recoverable from sticky

</td><td valign="top" width="50%">

**C++20 / systems**

- Paged KV cache block allocator with refcounted prefix sharing
- Bounded MPMC queue: mutex + condition variables
- Predicate waits — no bare `wait`, no spurious-wakeup bugs
- Thread pool with futures and graceful drain
- Relaxed atomics for hot-path counters
- Backpressure, load shedding, and deadline-aware dropping
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
# Which implementation path is actually active
python -c "from cudaforge.ops import backend_report; print(backend_report())"

# One request end to end
python examples/simple_inference.py --echo-runner

# What dynamic batching buys
python examples/concurrent_requests.py

# A real LoRA fine-tune, CPU, under a minute, no downloads beyond the model
python examples/fine_tune.py

# Every kernel against its reference — the first thing to run on new GPU hardware
python examples/kernel_parity.py

# The operators composed into a LLaMA-style block, checked against PyTorch
python examples/transformer_block.py
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

```bash
./scripts/build.sh --cuda
./build-cuda/tests/cuda/cudaforge_cuda_tests
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

What has been measured, on an Apple M5 Pro with a simulated executor:

| Measurement | Result |
| --- | --- |
| Batching, 16 clients, batch 1 → 16 | **3.52× throughput at unchanged p50 and p99** |
| Batching, 8 clients, batch 1 → 16 | 499 → 722 req/s, p99 12.9 → 7.9 ms |
| Batching, 1 client, batch 1 → 16 | 112 → 87 req/s — the latency cost, shown honestly |
| Memory pool | 2,020 allocations served by 5 backend calls; reuse rate 0.9975 |
| Latency histogram | worst error **4.95%** vs its documented 6.25% bound, at 76–120M records/s |
| HTTP end to end | 300 requests at concurrency 32: 420 req/s, client p99 279 ms vs server p99 1.03 ms |
| Paged KV cache | **13.2× more concurrent sequences** than contiguous on chat-shaped traffic; waste 93% → 4.8% |
| C++ suite | 160 cases, 49,866 assertions, clean under TSan / ASan / UBSan |
| Python suite | 418 tests, 91% statement coverage |

These measure the **scheduler**, not model throughput — execution is simulated
so the variable under study is isolated and the benchmark runs without a GPU.
**No CUDA kernel number exists**; the harness is complete and runs unchanged on
NVIDIA hardware. See [docs/benchmarking.md](docs/benchmarking.md).

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

- [x] **Paged KV cache — block allocator.** Reference-counted blocks, per-sequence
      block tables, copy-on-write for shared prefixes. Host-side and fully
      tested; see [kv-cache.md](docs/kv-cache.md).
- [ ] **Paged KV cache — attention gather.** The allocator is done; the kernel
      that reads through the block table is not, so nothing uses it yet.
- [ ] **Preemption policy.** The allocator supports evicting a sequence; nothing
      decides which one.
- [ ] **Continuous batching.** Batches are static once formed. Admitting new
      requests mid-generation would raise utilisation substantially.
- [ ] **Tensor-core matmul.** The tiled kernel is a teaching implementation and
      is not competitive with cuBLAS, by design.
- [ ] **Stream-ordered allocation.** The pool frees immediately; adopting
      `cudaMallocAsync` semantics would remove the "do not free in-flight
      buffers" constraint.
- [ ] **Speculative decoding.**
- [ ] **Multi-GPU inference.** Training has a DDP example; serving is
      single-device.
- [ ] **GPU-measured benchmarks.** Everything is in place; only hardware is
      missing.

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) has the rules that are not negotiable — no
fabricated measurements, every CUDA call checked, no `cudaDeviceSynchronize`,
portable code stays portable, and new kernels ship with a reference
implementation.

Release history is in [CHANGELOG.md](CHANGELOG.md).

## Licence

MIT — see [LICENSE](LICENSE).

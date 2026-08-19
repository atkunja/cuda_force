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

## The kernels

Each ships with a reference implementation, correctness tests over awkward
shapes, and a benchmark harness.

| Kernel | Variants | The interesting part |
| --- | --- | --- |
| **Reduction** | atomic · shared-memory tree · warp shuffle | why one `atomicAdd` per element serialises the whole grid |
| **Softmax** | naive · shared-memory · online | the online (max, sum) recurrence — no row-length limit, one fewer pass |
| **RMSNorm** | scalar · `float4` · FP16 | 128-bit transactions, plus alignment checks with a real fallback |
| **LoRA linear** | unfused · fused | keeping the `batch × rank` intermediate out of global memory |
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
```

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

Queue time and inference time are reported separately because they point at
different problems: high queue time means the runtime is saturated or the wait
is too generous, high inference time means the model or the batch size is the
constraint. `GET /health` and `GET /metrics` are also served.

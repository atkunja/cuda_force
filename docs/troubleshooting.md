# Troubleshooting

Symptoms, in the order they are usually hit.

## "It's slow" — but is it even using the custom kernels?

```python
from cudaforge.ops import backend_report

print(backend_report())
```

| Output | Meaning | Fix |
| --- | --- | --- |
| `custom CUDA kernels on NVIDIA ...` | working as intended | — |
| `PyTorch reference on cpu (extension not built...)` | no compiled extension | rebuild: `pip install -e . --no-build-isolation` |
| `PyTorch reference on cpu (extension loaded, compiled without CUDA)` | built without a toolkit | install CUDA, then rebuild |
| `PyTorch reference on ...` with `cuda_compiled=True` | built for CUDA, no device visible | check `nvidia-smi` and `CUDA_VISIBLE_DEVICES` |

A silent fallback looks exactly like a very slow custom kernel. Check this
before investigating anything else.

## The extension will not build

**`nvcc` not found.** `setup.py` gates on `torch.utils.cpp_extension.CUDA_HOME`,
not on a visible GPU. If `CUDA_HOME` is unset the build produces a CPU-only
extension without failing, which is intentional — the package must stay
installable on a machine with no toolkit.

**Compiler too old.** The bindings need C++20. `g++ --version` should be 10 or
newer, `clang++` 12 or newer.

**Architecture mismatch at runtime.** A kernel that launches with
`cudaErrorNoKernelImageForDevice` was compiled for a different compute
capability. Set `TORCH_CUDA_ARCH_LIST` to match the card and rebuild.

## The orchestrator keeps restarting the container

Check which probe is failing. `/health` is liveness — a failure there should
mean the process is broken. `/ready` is readiness, and returns 503 whenever the
queue is above 90% of capacity.

Wiring a liveness probe to `/ready` makes an orchestrator restart an instance
that is merely busy, discarding the queued work it was draining and pushing that
load onto its peers — which then also become busy. Point liveness at `/health`
and readiness at `/ready`.

## Throughput is lower than expected

Read four metrics together, not one:

```bash
curl -s localhost:8000/metrics | python -m json.tool
```

| Pattern | Diagnosis | Action |
| --- | --- | --- |
| `average_batch_size` near 1, `timeout_closure_fraction` near 1 | arrivals never fill a batch | lower `max_wait_us`; it is pure added latency |
| `average_batch_size` at the limit, `queue_depth` rising | saturated | raise `max_batch_size`, or add capacity |
| `average_batch_size` well below the limit with many clients | clients are blocking on responses | they can have at most one request in flight each — that is the client's concurrency, not a batcher fault |
| `requests_rejected` climbing | queue full, load being shed | raise `queue_capacity` only if latency allows; otherwise this is working correctly |

`average_batch_size` saturating at the client count rather than at
`max_batch_size` is normal and not a bug: N blocking clients can have at most N
requests outstanding.

## p99 latency is high but p50 is fine

A long tail with a healthy median usually means one of:

* **`max_wait_us` too generous.** It is the direct upper bound on
  batching-induced queue delay. Compare `queue_delay_p99_ms` against it.
* **A slow member dominating its batch.** Batches are static once formed, so a
  batch runs as long as its longest generation. This is a known limitation; see
  continuous batching in the roadmap.
* **Allocation stalls.** Check the pool's `reuse_rate`. If it is well below 1.0
  after warmup, shapes are varying more than expected and every miss is a
  device-synchronising `cudaMalloc`.

## Copies and kernels are not overlapping

Confirm it with an Nsight Systems timeline before changing anything. If they are
genuinely serialised, all three of these must hold and one of them will not:

1. work is on **different** streams — same-stream work is ordered by definition;
2. host memory is **pinned** — a copy from pageable memory is staged
   synchronously and cannot overlap. This is the usual cause;
3. no **device-wide synchronisation** intervenes — `cudaDeviceSynchronize` is a
   barrier across every stream. `scripts/check_cuda_sources.py` rejects it in
   this codebase, but a library you call might not.

Also check the streams were created with `cudaStreamNonBlocking`: a default
stream implicitly synchronises with the legacy stream, which serialises
everything.

## Training

**`no trainable parameters; adapters were not attached`.** `target_modules` does
not match any module name in the model. Print them:

```python
print([name for name, _ in model.named_modules()])
```

Names differ by architecture — GPT-2 uses `c_attn`, LLaMA-family models use
`q_proj`, `k_proj`, `v_proj`, `o_proj`.

**`load_in_4bit requires CUDA`.** bitsandbytes has no CPU or MPS backend. This
is a deliberate refusal rather than a silent fallback: training in a different
dtype would not fit the memory budget the config was written for.

**Loss is NaN.** In fp16, check that loss scaling is enabled — `GradScaler` is
active only when `mixed_precision` is on *and* the device is CUDA *and* bf16 is
unsupported. Prefer bf16 where available; it has fp32's exponent range and does
not need scaling at all.

**Two runs with the same config diverge.** Something is unseeded. `set_seed`
covers `random`, NumPy, torch CPU and all CUDA devices; anything outside those
needs its own seed.

## Tests

**Sporadic `SIGABRT` in a concurrency test.** Almost certainly a Catch2
assertion macro on a worker thread — they are not thread-safe. Use
`tests/cpp/thread_assert.hpp` and assert on the main thread after joining.

**A CUDA test "passes" without a GPU.** It did not run.
`tests/python/conftest.py` skips `cuda`-marked tests with a stated reason. Check
the summary line for skips.

**A tolerance failure that comes and goes.** An unseeded input. Every test in
this repository seeds explicitly for exactly this reason.

## Docker

**`could not select device driver "nvidia"`.** The NVIDIA Container Toolkit is
not installed or the daemon was not restarted after installing it.

**The image builds but the GPU is invisible.** `--gpus all` is required at run
time; the compose services declare it under
`deploy.resources.reservations.devices`.

**Building on Apple Silicon fails.** `nvidia/cuda` images are linux/amd64 and
there is no GPU to pass through. This is expected; the image is authored for
Linux and is not exercised on macOS.

# Development Environment

```bash
python scripts/environment_report.py
```

That prints the same facts for whatever machine you are on, including a direct
answer to "what can this machine actually build and run". The record below is
the development host, and it is the basis for every "not measured here" claim in
this repository.

This report records the machine CudaForge was developed on. It matters because
the development host has **no NVIDIA GPU**, which determines exactly which parts
of this repository were executed locally and which are hardware-pending.

## Host

| Property | Value |
| --- | --- |
| OS | macOS 26.5.2 (Darwin 25.5.0, build 25F84) |
| Architecture | arm64 (Apple Silicon) |
| CPU | Apple M5 Pro, 18 logical cores |
| Memory | 48 GB unified |
| Shell | zsh |

## Toolchain

| Tool | Version | Notes |
| --- | --- | --- |
| Apple clang | 21.0.0 (clang-2100.1.1.101) | Full C++20 including `<barrier>`, `<semaphore>`, `<format>` |
| CMake | 3.31+ (Homebrew) | Used for the portable C++ targets |
| Ninja | Homebrew | Default CMake generator |
| GNU Make | 3.81 (Apple) | Present but unused; Ninja preferred |
| Python | 3.12.14 (Homebrew) | System Python 3.9.6 is too old for the pinned deps |
| PyTorch | 2.13.0 | CPU/MPS build; `torch.cuda.is_available()` is False |
| transformers / peft | 5.15.1 / 0.20.0 | |
| shellcheck | Homebrew | |
| Docker | 29.6.1 | Engine present; see CUDA caveat below |
| Git | 2.50.1 | |
| gh | Homebrew | |

## CUDA availability

| Probe | Result |
| --- | --- |
| `nvcc` | **not found** |
| `nvidia-smi` | **not found** |
| CUDA toolkit | **not installed** |
| NVIDIA GPU | **none present** |

Apple Silicon has no CUDA path. There is no emulation layer that would make
`nvcc` output executable here, and none is attempted in this repository.

The Docker engine is present, but `nvidia/cuda` images are linux/amd64 and
depend on GPU passthrough via the NVIDIA Container Toolkit. Neither is available
on this host, so the CUDA container is authored for Linux and is not exercised
locally.

## Consequences for this repository

**Executed locally and verified:**

- The entire portable C++20 concurrency runtime (queue, thread pool, batcher,
  metrics, host-side memory pool) — compiled, unit tested, stress tested.
- C++ sanitizer runs (ASan, UBSan, TSan) on the concurrency targets.
- The Python package, its PyTorch reference operators, configuration layer, and
  batching logic — tested with pytest.
- CPU-side concurrency and batching benchmarks.

**Implemented but not executed here (requires an NVIDIA GPU):**

- All `.cu` kernels and their `nvcc` compilation.
- The PyTorch CUDA extension build and its bindings.
- The CUDA stream scheduler's runtime behaviour and the device memory pool.
- Every GPU benchmark number and every Nsight profile.

CUDA sources are still checked structurally in CI (see
[`.github/workflows/cpp.yml`](../.github/workflows/cpp.yml)) and are compiled on
an NVIDIA runner or in the CUDA container.

**No GPU performance number in this repository was fabricated.** Benchmark
result files are generated on the machine that runs them; the committed tree
contains harnesses, not invented measurements.

## Reproducing on Linux + NVIDIA

See [`docs/benchmarking.md`](benchmarking.md) for the exact command sequence, and
[`PROJECT_STATUS.md`](../PROJECT_STATUS.md) for the validation checklist.

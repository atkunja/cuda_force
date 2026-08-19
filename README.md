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

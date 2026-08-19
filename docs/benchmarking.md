# Benchmarking

## The one rule

**No performance number in this repository was written by hand.** Every figure
comes from a harness on the machine that ran it. The committed tree contains
benchmark code and no results; `benchmarks/results/` is gitignored, because
committed numbers would be numbers from someone else's machine, which is worse
than no numbers at all.

The development host has no NVIDIA GPU, so **no GPU measurement exists in this
project**. The CUDA harness is complete and runs unchanged on CUDA hardware.
Where a GPU comparison would normally appear, this repository says so instead.

## What can be measured where

| Suite | Needs | Measured on the dev host? |
| --- | --- | --- |
| `bench_queue` | C++ compiler | yes |
| `bench_scheduler` | C++ compiler | yes |
| `bench_memory` | C++ compiler | yes |
| `benchmark_batching.py` | Python | yes |
| `benchmark_kernels.py` | Python | yes, but see the caveat below |
| `bench_kernels` (CUDA) | NVIDIA GPU | **no** |

`benchmark_kernels.py` on a CPU-only host compares the reference
implementations against PyTorch's — which is nearly a tautology for the ops that
delegate. The output says so, in the results file itself:

```json
"caveat": "Custom CUDA kernels were not used. Both columns are PyTorch
           reference implementations, so the comparison shows dispatch
           overhead only, not kernel performance."
```

Always check `backend.using_custom_kernels` before reading a result file. A
silent fallback to the reference path looks exactly like a very slow custom
kernel.

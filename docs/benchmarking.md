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

## Running everything

```bash
./scripts/benchmark.sh
```

Writes timestamped JSON into `benchmarks/results/` and prints which suites were
skipped and why. Individually:

```bash
./build/benchmarks/bench_queue                  # producer/consumer scaling
./build/benchmarks/bench_scheduler              # batching parameter sweep
./build/benchmarks/bench_memory                 # pooled vs direct allocation
python benchmarks/benchmark_batching.py         # end-to-end batching sweep
python benchmarks/benchmark_kernels.py          # operators vs PyTorch
cudaforge-bench --echo-runner --clients 16      # engine under concurrent load
```

On an NVIDIA machine, add:

```bash
./scripts/build.sh --cuda
./build-cuda/benchmarks/bench_kernels > benchmarks/results/cuda-kernels.json
```

## Method

Each of these exists because omitting it produces a plausible, wrong number.

**Warm up.** The first call pays for CUDA context creation, kernel autotuning,
cuDNN algorithm selection and lazy module loading. A benchmark that includes
those is measuring initialisation. 20 warmup runs for CUDA kernels, 10 for the
Python operators, 2–3 engine iterations.

**Time the GPU with CUDA events.** A kernel launch is asynchronous, so a host
timer around it measures the launch. Adding a synchronise to fix that measures
the synchronisation as well. Events are recorded on the device timeline.

**Report the median.** The mean is dragged around by occasional scheduling
interference. Min and p95 are reported alongside so a suspiciously wide spread
is visible rather than averaged away.

**Prevent dead-code elimination.** A benchmark whose result is unused can be
deleted wholesale by the optimiser and will report an impossibly good number.
`bench::keep()` is an empty asm block with a memory clobber that stops it.

**Start producers simultaneously.** The concurrency benchmarks spin on a start
flag so every producer begins at the same instant; staggered starts understate
contention.

**Report effective bandwidth for memory-bound kernels.** Milliseconds mean
nothing without a reference. GB/s can be compared against the device's
theoretical peak, which the CUDA harness reads from device properties and
includes in its output. A kernel at 80% of peak is essentially done, regardless
of how its time compares to another implementation.

## What to look at

### Concurrency (`bench_queue`)

Sweeps producers x consumers x capacity. The interesting quantity is not peak
throughput but **where it stops scaling** — one mutex serialises every push and
pop, so beyond some thread count the queue becomes the bottleneck and adding
threads makes things worse. If that point sits below your target concurrency,
the fix is sharding the queue, not removing the lock.

### Batching (`bench_scheduler`, `benchmark_batching.py`)

Sweeps arrival concurrency against `max_batch_size` and `max_wait_us`. Four
numbers matter together:

| Column | Reading |
| --- | --- |
| `requests_per_second` | throughput |
| `average_batch_size` | how much aggregation is actually happening |
| `timeout_closure_fraction` | near 1.0 means arrivals never fill a batch — the wait is pure added latency |
| `queue_delay_p99_ms` | the batcher's own contribution to tail latency |

Throughput that has flattened while p99 keeps climbing means the batch is
already large enough. A `timeout_closure_fraction` near 1.0 with small batches
means lowering `max_wait_us` costs nothing and saves latency.

Execution is simulated with a sleep unless `--model` is passed, so these
describe the **scheduler**, not model throughput. That separation is deliberate:
it isolates the variable under study and keeps the benchmark runnable without a
GPU. The output says so.

### Memory (`bench_memory`)

On the host backend the wall-clock saving is modest, because `malloc` already
caches. The numbers that carry over are `reuse_rate` and the ratio of
`backend_allocations` to `pool_allocations`: on the device backend each avoided
call is a `cudaMalloc` that would have synchronised the device. That is the
actual argument for the pool, and it cannot be measured on this host.

### CUDA kernels (`bench_kernels`)

Compares naive against optimised for every kernel, with effective bandwidth
against the device's theoretical peak. Requires an NVIDIA GPU; no such run has
been performed for this repository.

## Reporting results

If you run these on real hardware, include:

* GPU model and driver version (the harness records both),
* CUDA toolkit version,
* the full JSON, not a selected figure,
* whether `using_custom_kernels` was true.

A speedup quoted without the baseline's configuration is not a measurement.

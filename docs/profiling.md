# Profiling

Benchmarks tell you *that* something is slow. Profilers tell you *why*. The two
NVIDIA tools answer different questions and are not interchangeable.

| Tool | Scope | Question it answers |
| --- | --- | --- |
| Nsight Systems (`nsys`) | whole process, timeline | Is the GPU busy? Do copies overlap compute? Where are the gaps? |
| Nsight Compute (`ncu`) | one kernel, hardware counters | Why is *this* kernel slow? Memory- or compute-bound? Coalesced? |

Use them in that order. Optimising a kernel that occupies 5% of the timeline is
wasted effort, and only `nsys` can tell you that.

> Neither tool has been run against this project. The development host has no
> NVIDIA GPU — see [environment.md](environment.md).

```bash
./scripts/profile.sh
```

## Nsight Systems: is the GPU busy?

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=cpu \
  --cuda-memory-usage=true \
  --output=timeline \
  ./build-cuda/benchmarks/bench_kernels

nsys stats --report cuda_gpu_kern_sum timeline.nsys-rep
```

### What to look for

**Gaps in the GPU row.** The GPU is idle. Almost always one of:

| Cause | Signature in the timeline |
| --- | --- |
| Host-side bottleneck | CPU rows busy while the GPU row is empty |
| Synchronous copies | copy and kernel rows never overlap; copies sit on the critical path |
| Device-wide sync | every stream stops at the same instant |
| Allocation | a gap immediately before a kernel, aligned with a `cudaMalloc` |
| Launch overhead | many tiny kernels with gaps between them |

**Copy and compute on separate rows, running simultaneously.** This is the
single most useful thing the timeline shows, because it is the direct test of
whether the stream scheduler is working. If they are serialised, check the three
conditions in [gpu-execution.md](gpu-execution.md) — in practice the cause is
almost always pageable host memory.

**Streams not overlapping.** If all work sits on one row, either everything was
issued to one stream, or a `cudaDeviceSynchronize` is acting as a barrier
between batches. The structural checks reject the latter, so suspect the former.

### Annotating with NVTX

NVTX ranges label regions in the timeline so a gap can be attributed to a phase
rather than guessed at. They are built in, behind a CMake option:

```bash
cmake -S . -B build-cuda -G Ninja \
  -DCUDAFORGE_ENABLE_CUDA=ON -DCUDAFORGE_ENABLE_NVTX=ON
```

`scripts/profile.sh` enables it automatically. Adding your own:

```cpp
#include "cudaforge/nvtx.cuh"

void execute(Batch&& batch) {
    CUDAFORGE_NVTX_RANGE("batch_execute", NvtxCategory::Compute);
    // ...
}                                   // the range closes here, exceptions included
```

The RAII form matters: pushing and popping by hand is exactly the bookkeeping an
early return gets wrong, and an unbalanced range makes the whole timeline
unreadable rather than merely losing one label.

Categories are colour-coded — ingress blue, batching orange, transfers green,
compute red, sampling purple — so the timeline is readable without reading every
label. The scheduler's copies and its shutdown drain are annotated already; a
long `synchronize_all` bar anywhere other than shutdown means the pipeline is
being drained when it should not be.

Annotate **phases, not launches**. Each range costs a few hundred nanoseconds,
which is nothing against per-batch work and considerable against a single
kernel launch. Everything is compiled out when the option is off.

## Nsight Compute: why is this kernel slow?

```bash
ncu \
  --set full \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section Occupancy \
  --section WarpStateStats \
  --kernel-name-base demangled \
  --launch-count 3 \
  --export kernels \
  ./build-cuda/benchmarks/bench_kernels
```

`ncu` replays each kernel many times to collect counters, so it is far slower
than `nsys`. Point it at a small set of kernels — `--launch-count 3` and a
`--kernel-name` filter — rather than a whole run.

Build with `-lineinfo`, which the CMake configuration already does. It maps
counters back to source lines and costs nothing at runtime, unlike `-G`, which
disables optimisation entirely and makes any measurement meaningless.

### The four sections that matter

**SpeedOfLight** — is this memory-bound or compute-bound? Everything in this
project except the matmul should be memory-bound. If a kernel reports low
utilisation on *both*, it is latency-bound: not enough parallelism to hide
memory latency, so the fix is more resident warps, not fewer instructions.

**MemoryWorkloadAnalysis** — the coalescing check. The number to watch is
sectors per request:

| Sectors/request | Meaning |
| --- | --- |
| ~4 (FP32, 32 threads) | fully coalesced; the ideal |
| 8–16 | partially coalesced; strided or misaligned access |
| 32 | every thread hits a different sector — an 8x bandwidth penalty |

This is where a `float4` load shows up: the vectorised RMSNorm should issue a
quarter of the requests the scalar version does for the same bytes.

**Occupancy** — achieved vs theoretical. A large gap means something is limiting
residency: registers per thread, shared memory per block, or block size.
Remember that occupancy is a means, not an end — it matters only because
resident warps hide latency. If memory throughput is already near peak, raising
occupancy buys nothing.

**WarpStateStats** — where warps stall, and how much they diverge. `Stall Long
Scoreboard` dominating means waiting on global memory, which is expected for
these kernels. High branch divergence in a reduction usually means the loop
condition is the strided-modulo form rather than `index < half`.

### Reading a result

| Observation | Likely cause | Fix |
| --- | --- | --- |
| Low memory throughput, low compute | latency-bound | more warps: smaller blocks, fewer registers, or more blocks per SM |
| High sectors/request | uncoalesced access | change the access pattern or vectorise |
| High shared-memory bank conflicts | power-of-two stride | pad the array, as `tile_b[kTile][kTile + 1]` does |
| Memory throughput near peak | done | stop; further work needs an algorithmic change |
| High register spills | too much per-thread state | reduce block size or unrolling |

## Profiling the CPU side

The concurrency runtime has its own failure modes, and no GPU profiler will show
them.

**Lock contention.** `perf` on Linux:

```bash
perf record -g ./build/benchmarks/bench_queue
perf report --sort=symbol
```

Time in `__lll_lock_wait` or `pthread_cond_wait` is threads blocked, not
working. Some is expected — that is what a bounded queue does under load — but
if it dominates while throughput is flat, the queue's single mutex is the
bottleneck and the answer is sharding.

**False sharing.** Two atomics on the same 64-byte cache line cause the line to
ping-pong between cores, and the symptom is throughput that *falls* as threads
are added. `perf c2c record` identifies it directly. The counters in `ThreadPool`
and `Metrics` are candidates if the queue benchmark ever shows negative scaling.

**ThreadSanitizer.** Not a profiler, but it belongs in the same workflow: it
reports races even when the interleaving that would expose them did not occur
during the run.

```bash
./scripts/build.sh --sanitizer thread
./build-thread/tests/cpp/cudaforge_tests
```

## A workflow that converges

1. **Benchmark.** Establish a baseline number and confirm it is reproducible.
2. **`nsys`.** Find where the time actually goes. Frequently it is not the
   kernel anyone suspected.
3. **`ncu` on the top kernel.** Determine whether it is memory-bound,
   compute-bound or latency-bound. That answer selects the fix; guessing does
   not.
4. **Change one thing.** Two simultaneous changes make the measurement
   uninterpretable.
5. **Re-benchmark.** If the improvement is smaller than run-to-run variance, it
   is not an improvement.
6. **Record it.** Baseline, bottleneck, change, expected effect, measured
   effect. An optimisation whose measured effect was not recorded will be
   re-litigated later.

## Recording optimisation work

Every optimisation in this project is documented in that shape. For example:

> **Reduction, naive to warp-shuffle.**
> *Baseline:* one `atomicAdd` per element.
> *Bottleneck:* every thread contends on one address; the memory subsystem
> serialises conflicting atomics.
> *Change:* grid-stride loads, warp-shuffle reduction, one atomic per block.
> *Expected:* global atomics fall from N to N/blockDim, roughly 256x fewer.
> *Measured:* **not measured — no NVIDIA GPU on the development host.**

The last line is the important one. It is what the whole project says wherever a
number would otherwise go.

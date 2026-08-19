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
rather than guessed at:

```cpp
#include <nvtx3/nvToolsExt.h>
nvtxRangePush("batch_execute");
// ...
nvtxRangePop();
```

Worth adding around batch formation, H2D, compute and D2H when investigating a
specific stall.

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

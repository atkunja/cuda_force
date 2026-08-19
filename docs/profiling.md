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

# GPU Execution Model

What the stream scheduler is arranging, and the conditions under which the
arrangement actually produces overlap.

## Streams are the ordering primitive

Work issued to one stream runs in issue order. Work in different streams may
overlap. That is the entire contract, and everything below follows from it.

A GPU can run a kernel, a host-to-device copy and a device-to-host copy
simultaneously, because the copy engines are separate hardware from the SMs.
With three streams and per-batch work split into copy-in, compute and copy-out:

```
stream 0:  H2D(b0)  ──  kernel(b0)  ──  D2H(b0)
stream 1:            H2D(b1)  ──  kernel(b1)  ──  D2H(b1)
stream 2:                      H2D(b2)  ──  kernel(b2)  ──  D2H(b2)
                     ↑ copy and compute proceed at the same time
```

In steady state the throughput gain approaches the ratio of total work to
compute-only work. If copies are 40% of the timeline, removing them from the
critical path is close to a 1.6x improvement — with no change to any kernel.

## Three conditions, all required

Overlap does not happen by default. All three of these must hold:

**1. Different streams.** Same-stream work is strictly ordered by definition. Two
kernels issued to the same stream never overlap no matter how small they are.

**2. Pinned host memory.** `cudaMemcpyAsync` from pageable memory is not
asynchronous: the driver stages it through an internal pinned buffer, which
serialises the copy against both the host thread and the stream. This is the
condition people miss, because the code looks identical and simply performs like
the synchronous version. See [memory-management.md](memory-management.md).

**3. No device-wide synchronisation.** A single `cudaDeviceSynchronize()` is a
barrier across every stream and collapses the pipeline. It appears nowhere in
this project, and `scripts/check_cuda_sources.py` fails the build if it is
introduced.

## Non-blocking streams

Streams are created with `cudaStreamNonBlocking`:

```cuda
cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
```

A stream created with plain `cudaStreamCreate` implicitly synchronises with the
legacy default stream. Any library call that touches the default stream would
then serialise every stream the scheduler owns — silently undoing the overlap it
exists to create. The flag is not a micro-optimisation; without it the scheduler
does not work.

## Events

Events serve two distinct purposes, and the right creation flags differ:

| Purpose | Flags | Why |
| --- | --- | --- |
| Ordering | `cudaEventDisableTiming` | skips a timestamp write on every record |
| Timing | default | records a device-side timestamp |

Using a timing-enabled event purely for ordering is a common and easy
inefficiency, so `CudaEvent`'s constructor makes the choice explicit rather than
defaulting silently.

### Cross-stream dependencies

```cuda
producer.record_completion();          // mark a point in the producer's stream
GpuScheduler::chain(consumer, producer);  // consumer waits, host does not
```

`cudaStreamWaitEvent` makes one stream wait for another **without blocking the
host**. The alternative — synchronising the host and then issuing the dependent
work — idles the GPU for a full host round trip on every dependency, which for
a pipeline with a dependency per batch is most of the timeline.

### Timing GPU work

CUDA events are the only correct way to time a kernel:

```cuda
start.record(stream);
launch_kernel<<<grid, block, 0, stream>>>(...);
stop.record(stream);
stop.synchronize();
float ms = CudaEvent::elapsed_ms(start, stop);
```

A host timer around a launch measures the *launch*, because the launch is
asynchronous and returns immediately. Adding a synchronise to fix that measures
the synchronisation as well. Events are recorded on the device timeline and
measure exactly the work between them.

## Stream assignment

Round-robin over a relaxed atomic counter.

The obvious alternative — query each stream and pick an idle one — costs a
driver call per acquisition, and `cudaStreamQuery` only reports whether *all*
work on a stream is done. In practice that piles short batches onto whichever
stream happened to finish first, producing exactly the uneven distribution the
overlap argument assumes away.

Round-robin is contention-free, predictable, and produces the even spread the
pipeline needs. The counter is relaxed because it only distributes work: nothing
is published through it and no reader establishes happens-before from it.

### How many streams

| Streams | Effect |
| --- | --- |
| 1 | no overlap at all; copies and kernels serialise |
| 2 | compute overlaps one direction of copy |
| 3–4 | copy-in, compute and copy-out all in flight — the useful range |
| 8+ | copy engines saturate; additional streams add scheduling overhead |

Most parts have two copy engines, so three to four streams is enough to keep
them and the SMs busy simultaneously. The default is 4.

Streams are created at the highest available priority band so inference work is
not preempted by lower-priority background streams a host application may be
running on the same device.

## Launch configuration

### Block size

256 threads by default: four warps per block, enough resident warps to hide
memory latency while leaving register headroom. At 1024 threads the per-thread
register budget drops far enough to spill on several of these kernels, and a
spill to local memory costs more than the occupancy gains.

Row-wise kernels size the block to the row instead, rounded up to a power of two
so the tree reduction's halving loop terminates cleanly at one active thread.

### Grid size

Two different strategies, for two different reasons:

* **Grid-stride kernels** (whole-array reduction) use a grid proportional to the
  SM count, capped by the input. Each block handles many elements, so one block
  per tile would create far more blocks than can be resident and pay scheduling
  overhead for nothing.
* **Row-per-block kernels** (softmax, RMSNorm) use one block per row. The row is
  then reduced entirely in registers and shared memory, with no global atomics
  at all.

### Occupancy is a means, not an end

Occupancy — resident warps per SM as a fraction of the maximum — matters only
because idle warps are what hide memory latency. Past the point where latency is
already hidden, more occupancy buys nothing, and a kernel that raises occupancy
by using fewer registers can easily be slower.

Nsight Compute reports achieved occupancy alongside memory throughput; the
second is the number to optimise. See [profiling.md](profiling.md).

## Warp execution

Threads execute in warps of 32 in lockstep. Two consequences shape every kernel
here:

**Divergence serialises.** When threads in a warp take different branches, the
hardware executes both paths with the inactive lanes masked off. The reduction
loop uses `index < half` rather than a strided-modulo condition specifically so
that entire warps retire together instead of every warp running at partial
occupancy throughout.

**`__syncthreads()` must be reached by every thread in the block.** A barrier
inside a conditional deadlocks when threads diverge. `check_cuda_sources.py`
flags the shape that is always wrong.

**Warp primitives need an explicit mask.** On Volta and later, lanes can be at
genuinely different instructions, so `__shfl_down_sync` needs a mask naming
every participating lane. Passing `0xffffffff` from a divergent branch is
undefined behaviour. The maskless legacy forms are rejected by the structural
checks.

## What has not been measured

Nothing in this document has been executed. The development host has no NVIDIA
GPU — see [environment.md](environment.md). The scheduler compiles under the
CUDA container and the NVIDIA-runner CI job, and `tests/cuda/test_scheduler.cu`
verifies round-robin assignment, event ordering, cross-stream chaining and
per-stream accounting on real hardware.

Whether the overlap described above materialises on a given workload is a
question for Nsight Systems, not for this document.

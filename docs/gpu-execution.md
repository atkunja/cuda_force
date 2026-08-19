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

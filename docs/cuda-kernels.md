# CUDA Kernels

Every kernel here ships in more than one form. The point is not that the
optimised version is faster — it is that the *reason* it is faster is the same
handful of reasons every time, and seeing them applied to five different
problems is more useful than seeing one heavily tuned kernel.

> **No performance numbers appear in this document.** The development host has
> no NVIDIA GPU, so nothing here has been measured. The benchmark harness is
> complete and runs unchanged on CUDA hardware; see
> [benchmarking.md](benchmarking.md).

## What actually costs time

Almost every kernel in this project is **memory-bound**. A modern GPU can issue
tens of TFLOPs but only move a few TB/s, so for any operation doing O(1)
arithmetic per element, the arithmetic is free and the only question is how
efficiently bytes move.

That reframes optimisation. The levers that matter, in rough order of impact:

| Lever | Mechanism |
| --- | --- |
| Coalescing | adjacent threads read adjacent addresses, so the hardware merges them into the fewest possible transactions |
| Reducing traffic | fuse passes so each byte crosses the memory bus once instead of three times |
| Shared memory | stage a tile on-chip and reuse it, instead of re-reading from global memory |
| Warp primitives | exchange through registers, removing both shared-memory traffic and barriers |
| Occupancy | enough resident warps that memory latency is hidden by other warps' work |
| Launch overhead | fewer, larger kernels rather than many small ones |

Arithmetic optimisation appears nowhere on that list, which is the point.

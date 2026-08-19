# Memory Management

## The problem

`cudaMalloc` and `cudaFree` are not ordinary allocator calls. Both synchronise
the device: the driver must ensure no outstanding work references the memory
being mapped or unmapped, so every call drains the pipeline.

In a serving loop that allocates activation buffers per batch, that turns each
allocation into a full pipeline stall. Worse, it serialises exactly the work the
stream scheduler exists to overlap — a `cudaMalloc` between a copy and a kernel
undoes the overlap both were arranged to achieve.

The fix is not a faster allocator. It is to stop calling the allocator at all in
steady state.

## The pool

[`memory_pool.hpp`](../cpp/include/cudaforge/memory_pool.hpp) holds freed blocks
and hands them back out. Allocation becomes a free-list pop; the backend is
touched only when a size class is empty.

```
allocate(bytes)
  ├─ round up to the next power of two (floored at min_block_bytes)
  ├─ free list for that class non-empty?  →  pop and return   [no driver call]
  └─ empty?                               →  backend.allocate [one driver call]

deallocate(pointer)
  └─ look the size up and push onto that class's free list    [no driver call]
```

The size is looked up from the pool's own records rather than taken from the
caller, so a caller that reports the wrong size cannot corrupt the free lists.

## Size classes

Requests round up to the next power of two. That caps internal fragmentation at
just under 2x while making free lists O(1) to index and making blocks freely
interchangeable within a class.

The alternative — best-fit over exact sizes — wastes less memory but needs a
search on every allocation and fragments over time as odd-sized holes
accumulate. For inference, where the same shapes repeat batch after batch, exact
reuse dominates after warmup and the rounding rarely costs anything.

A minimum block size floors the classes. Without it, a workload allocating many
8-byte scratch buffers would create size classes far below any useful
granularity and defeat reuse across slightly different shapes.

| Request | Class | Slack |
| --- | --- | --- |
| 1,000 B | 1,024 B | 2.4% |
| 1,100 B | 2,048 B | 86% |
| 2,000 B | 2,048 B | 2.4% |
| 4,096 B | 4,096 B | 0% |

The 1,100-byte row is the worst case and is the price of the scheme. It is
acceptable because real workloads cluster on a handful of shapes rather than
spreading uniformly across the range.

## What this is not

There is no splitting, no coalescing, no defragmentation, and no stream-ordered
semantics. Those are what make a production allocator hard, and they are already
solved elsewhere:

| Allocator | Splitting/coalescing | Stream-ordered | Notes |
| --- | --- | --- | --- |
| `cudaMalloc` | n/a | no | synchronises the device on every call |
| `cudaMallocAsync` | yes | yes | driver-managed pool; the right default on CUDA 11.2+ |
| PyTorch caching allocator | yes | yes | per-stream pools, splitting, and an expandable-segments mode |
| CudaForge `MemoryPool` | no | no | readable, testable, and enough for fixed-shape inference |

**Use `cudaMallocAsync` or PyTorch's allocator in production.** This pool exists
because a from-scratch implementation is the clearest way to show what those
allocators are doing and why it matters, and because its behaviour is fully
testable without a GPU.

The most important thing it does *not* implement is stream-ordered semantics.
`cudaMallocAsync` associates a free with a stream, so memory is not reused until
the work that referenced it completes. This pool returns a block to the free
list immediately, which means **a caller must not free a buffer while a kernel
using it is still in flight.** In this codebase that is enforced structurally:
buffers are freed after the owning stream has been synchronised.

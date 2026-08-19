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

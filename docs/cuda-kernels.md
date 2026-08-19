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

## Kernel A — Reduction

Sum an array. The simplest possible operation, and the clearest illustration of
why naive GPU code is slow.

### Naive: one atomic per element

```cuda
atomicAdd(output, input[index]);
```

Correct, and roughly as slow as a GPU reduction can be. Every thread in the grid
targets the same address, and the memory subsystem serialises conflicting
atomics — so a machine with tens of thousands of threads performs the additions
essentially one at a time. The bottleneck is not the addition; it is contention.

### Shared memory: tree reduction per block

Each block reduces its own tile to a single value through `log2(blockDim)`
halving steps, then performs one atomic. Global atomics drop from N to
N/blockDim — a factor of 256 at the default block size.

Two details in the halving loop are easy to get wrong:

```cuda
for (unsigned half = blockDim.x / 2; half > 0; half >>= 1) {
    if (index < half) { tile[index] += tile[index + half]; }
    __syncthreads();
}
```

* `index < half` keeps the active threads **contiguous**, so entire warps retire
  together. The strided-modulo formulation (`if (index % (2 * stride) == 0)`)
  computes the same answer but leaves every warp partially active throughout,
  wasting most of the machine.
* Consecutive threads touch consecutive shared-memory addresses at every step,
  so there are no bank conflicts.

### Warp shuffle: the last five steps in registers

`__shfl_down_sync` reads another lane's register directly. The final five
halving steps — those within a single warp — need no shared memory and no
`__syncthreads()`, because lanes in a warp advance together.

The shared array shrinks from one float per thread to one per warp, and the
barrier count drops from `log2(blockDim)` to two.

**The mask is not optional.** On Volta and later, lanes in a warp can genuinely
be at different instructions, so the mask must name every lane that will execute
the shuffle. Passing `0xffffffff` from a divergent branch is undefined
behaviour — which is why the block reduction is structured so the whole warp
reaches the primitive.

### Grid sizing

The grid is fixed at a multiple of the SM count rather than derived from the
input size. With a grid-stride loop each block handles many elements, so
launching one block per tile would create far more blocks than the device can
hold resident and pay scheduling overhead for no additional parallelism. The
size is also capped by the input, so a tiny array does not launch blocks with
nothing to do.

## Kernel B — Softmax

Row-wise `exp(x_i - m) / sum(exp(x_j - m))`.

### Numerical stability is not optional

The textbook definition overflows for inputs above roughly 88 in FP32, and
attention logits routinely exceed that. Subtracting the row maximum first is an
exact identity — the `exp(-m)` factor cancels — but it makes the largest
argument to `exp` equal to 0, so the largest term is exactly 1 and nothing
overflows. Underflow of very negative terms to zero is harmless: they contribute
nothing to the sum.

`tests/cuda/test_softmax.cu` feeds logits of ±1000 to every variant and requires
finite output. Without the shift, all of them produce NaN.

### Three variants, three tradeoffs

| Variant | Global traffic | Row length limit | Notes |
| --- | --- | --- | --- |
| Naive | 3 reads + 1 write | none | separate max, sum and normalise passes |
| Shared memory | 1 read + 1 write | fits in shared memory | stages the row on-chip |
| Online | 2 reads + 1 write | none | maintains max and sum together |

The naive version is not a strawman — it is what a correct first implementation
looks like, and it is memory-bound at three times the minimum possible traffic.
That factor of three *is* the optimisation opportunity.

### Online softmax

Each thread maintains a running `(max, sum)` pair. When a larger element
appears, the accumulated sum is rescaled by `exp(old_max - new_max)` so it stays
expressed relative to the current maximum:

```cuda
if (value > running_max) {
    running_sum *= __expf(running_max - value);
    running_max  = value;
}
running_sum += __expf(value - running_max);
```

Combining two such pairs is the same operation, which is what makes the
reduction across threads valid. This removes the separate max pass, and because
nothing is cached the row length is unbounded — unlike the shared-memory
variant, whose capacity limit the launcher detects and falls back from rather
than failing the launch.

This is the same recurrence FlashAttention uses to avoid materialising the
attention matrix, applied to the simpler standalone case.

### FP16

Storage is FP16; accumulation is FP32. An FP16 accumulator has a 10-bit
mantissa, so summing more than about 2048 terms of similar magnitude stops
making progress — each addition rounds away — and the resulting distribution is
visibly wrong. Widening the accumulator costs nothing on a bandwidth-bound
kernel.

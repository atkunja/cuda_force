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

### The barrier that is easy to miss

A block reduction that ends `return shared[0];` is a race waiting for a caller
that reduces twice over the same array — which softmax does, max then sum. One
warp can overwrite `shared[0]` while another has not yet read it, and the
symptom is a silently wrong row rather than a crash.

The primitives therefore read into a register and bar every thread from leaving
until all have read:

```cuda
__syncthreads();
const T result = shared[0];
__syncthreads();          // nobody leaves until everyone has read
return result;
```

This was found by re-reading the sources rather than by a test — the code cannot
be compiled on the development host — and is now enforced by a structural rule,
checked at the primitive's definition rather than at its call sites.

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

## Kernel C — RMSNorm

```
y_i = x_i / sqrt(mean(x^2) + eps) * w_i
```

The difference from LayerNorm is that the mean is not subtracted: RMSNorm
rescales without re-centring. That removes one full pass over the row and one
reduction. At transformer widths the operation is memory-bound, so halving the
reductions is close to halving the cost — which is why LLaMA-family models moved
to it.

`eps` goes **inside** the square root, matching the reference PyTorch
formulation. Outside, the result differs for small-magnitude rows and parity
with the reference breaks.

### Vectorisation

The scalar kernel issues one 32-bit load per element. The vectorised kernel
treats the row as `cols / 4` `float4` values and issues one 128-bit transaction
instead of four 32-bit ones — a quarter of the memory instructions on a kernel
that is entirely memory-bound.

The constraint is alignment. A `float4` load faults unless the address is
16-byte aligned, and row starts must be aligned too, which requires `cols` to be
a multiple of four — otherwise row 1 begins mid-vector even when row 0 is
aligned. The launcher checks both the pointers and the row length, and falls
back to the scalar kernel rather than faulting:

```cuda
return cols % 4 == 0 && aligned(input) && aligned(weight) && aligned(output);
```

`tests/cuda/test_rmsnorm.cu` deliberately includes non-multiple-of-four widths
(17, 1023, 2049) so the fallback path is exercised rather than assumed.

### `rsqrtf`

`rsqrtf` maps to a single hardware instruction. Its relative error is well
inside the FP32 tolerance the tests use, and it removes both a division and a
square root from the inner loop. This is one of the few places where an
arithmetic choice is worth making at all — and it is worth it because it removes
instructions, not because it removes flops.

### FP16 overflow

FP16's maximum is 65504. A single activation of magnitude 256 squares to 65536
and overflows to infinity, which then propagates through the entire row. The
sum of squares is therefore accumulated in FP32 unconditionally.
`tests/cuda/test_rmsnorm.cu` uses inputs of 300 specifically to hit this, and
`tests/python/test_ops.py` asserts the same property for the reference path.

## Kernel D — LoRA Linear

```
Y = X W + scale * (X A) B
```

with `W: [in, out]` frozen, `A: [in, r]` and `B: [r, out]` trainable, and `r`
typically 8–64 against `in` in the thousands.

### The tiled matmul underneath

The naive matmul reads a full row of A and column of B from global memory for
every output element, so each element of A is re-read `n` times. Staging tiles
in shared memory means each element crosses the memory bus once per tile and is
then reused `kTile` times.

Tile width is 16: 256 threads per block, a 1 KB tile per operand, and 16 reuses
per loaded element. Wider tiles improve the reuse ratio, but shared memory grows
quadratically and occupancy falls — 32 would need 4 KB per operand and halve the
blocks resident per SM on most parts.

One line in the tile declaration is doing real work:

```cuda
__shared__ float tile_b[kTile][kTile + 1];
```

The `+ 1` breaks the power-of-two stride so that a column access hits `kTile`
distinct banks instead of all mapping to one. Without it, the column reads in
the inner loop are a 16-way bank conflict and the inner loop runs 16x slower.

Both `__syncthreads()` calls are required. The second is the one people omit:
without it, a fast warp can begin loading tile `t+1` while a slow warp is still
reading tile `t`.

**This matmul is not competitive with cuBLAS and is not meant to be.** cuBLAS
uses tensor cores, deeper register blocking, and per-architecture tuned tile
shapes. This one exists to make the shared-memory argument concrete and to give
the fused kernel a baseline.

### Why fusion pays here

`X A` is tall and extremely thin — small enough to live in shared memory, large
enough that a round trip through global memory costs more than the multiply
itself. That imbalance is exactly what kernel fusion removes.

The fused kernel computes a block's slice of `X A` into shared memory, then
walks the output columns adding the frozen and adapter contributions in one
pass. Relative to the unfused path it saves:

* two kernel launches,
* a `batch x rank` write to global memory and the matching read,
* a re-read of `X` for the adapter path, since those rows are already resident.

The unfused path remains as the correctness reference and the fallback when
`rank * kTile` floats exceed the shared-memory budget.

## Kernel E — Quantise / Dequantise

Block-wise symmetric INT8:

```
scale_b = max(|x|) over block b / 127
q_i     = round(x_i / scale_b), clamped to [-127, 127]
```

### Design choices

**Block-wise, not per-tensor.** Per-tensor scaling is cheap, but a single
outlier stretches the range for every other value — and transformer weights have
outliers. Per-element scales would be exact and would also defeat the purpose,
since the scales would cost more than the saved payload. At 64 elements per
scale the overhead is 4 bytes per 64 bytes of INT8, about 6%, and any outlier's
influence is confined to its own block.

`tests/cuda/test_quantization.cu` asserts this directly: an element sharing a
block with a 1000x outlier loses precision, while the same value three blocks
away round-trips within 1%.

**Symmetric, no zero point.** Dequantisation stays a single multiply, and exact
zeros map to exact zeros — which matters for padding and masks.

**127, not 128.** Negation stays representable and the grid is symmetric about
zero.

**`rintf`, not truncation.** Round-half-to-even matches the reference
implementation. Truncation would bias every value toward zero and shift the mean
of the dequantised tensor.

**An all-zero block gets a scale of 1.** Its absmax is zero; a zero scale would
divide by zero, while a scale of 1 maps the block to zero and back exactly.

### What this is not

This is **not** a reimplementation of bitsandbytes' NF4. NF4 uses a non-uniform
4-bit grid derived from the normal distribution's quantiles, plus double
quantisation of the scales themselves. Nothing here reproduces that, and the
QLoRA path in `training/` calls bitsandbytes directly rather than pretending
this is equivalent.

The error bound this scheme does guarantee is half a quantisation step —
`scale_b / 2` — and both the CUDA and Python test suites assert exactly that.

## Kernel F — Activations

SiLU, GELU and SwiGLU. Elementwise, entirely bandwidth-bound, and the clearest
possible case for fusion.

### SwiGLU and why it is a kernel at all

The LLaMA-family feed-forward block computes two projections of the same input
and multiplies the gated one elementwise:

```
FFN(x) = (silu(x W_gate) * (x W_up)) W_down
```

Written with framework primitives, `silu(gate) * up` is three passes over
memory: compute the activation, write it, read it back, multiply, write again.
The arithmetic — one sigmoid and two multiplies per element — is free by
comparison.

Fused, it is one read of each input and one write. That is the theoretical
minimum for this operation, which means the fused kernel is within a small
factor of the hardware limit no matter how much more effort goes into it. The
`float4` variant then quarters the number of memory instructions for the same
bytes.

| Form | Global traffic |
| --- | --- |
| Framework primitives | 2 reads + 1 write + 1 read + 1 write |
| Fused scalar | 2 reads + 1 write |
| Fused `float4` | same bytes, a quarter of the instructions |

### Why FP32 for the sigmoid

FP16 has enough *range* for these values, but `exp` of a moderately negative
input underflows a 10-bit mantissa long before it underflows FP32. The result
is that the activation's negative tail flattens to exactly zero — not a crash,
just a quietly different function. The sigmoid is therefore evaluated in FP32
and the product stored back as FP16.

### GELU: tanh, not erf

The tanh approximation, because that is what GPT-2 and BERT were trained
against. Substituting the exact erf form changes outputs by more than the
numerical difference suggests, since the weights were fitted against this
particular curve. `tests/python/test_ops.py` asserts that the two forms differ
measurably — so the choice cannot be silently reverted.

## Kernel G — Fused residual add + RMSNorm

The highest-frequency fusion opportunity in transformer inference, because it
occurs twice per layer, at every layer, for every token.

Every block is

```
x = x + attention(norm(x))
x = x + mlp(norm(x))
```

so a residual add is *always* immediately followed by the next sublayer's
normalisation. Unfused, that sequence writes the sum, reads it back to compute
the sum of squares, and reads it a third time to scale.

| Form | Global traffic per element |
| --- | --- |
| Separate add, then RMSNorm | 2 reads + 1 write, then 2 reads + 1 write |
| Fused | 2 reads + 2 writes |

The kernel computes the sum once, keeps it in registers for the reduction, and
writes it while the row is still resident.

### Two outputs, both required

`out` feeds the next sublayer; `residual_out` is the value the *following*
residual connection adds to. Returning only the normalised result would force
the caller to recompute the sum — and, worse, is an easy mistake to make that
silently changes the model rather than failing.
`tests/cuda/test_fused_norm.cu` asserts that `residual_out` is the plain sum.

### Why the second pass re-reads rather than recomputes

The normalisation loop reads back from `residual_out` instead of redoing
`input[col] + residual[col]`. By that point the row is in L2, so one read beats
two reads plus an add.

## Indexing and bounds

Two classes of defect that a GPU-less host cannot catch by running anything, and
that both surface far from their cause:

**32-bit index overflow.** `blockIdx.x * blockDim.x + threadIdx.x` is computed
in 32-bit arithmetic. For an `int` element count near `INT_MAX` the product
exceeds `INT_MAX`, and casting it to `int` is undefined — the kernel reads the
wrong element rather than faulting. Every index in this project is computed in
`std::size_t`:

```cuda
const std::size_t index =
    blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
if (index < static_cast<std::size_t>(count)) { ... }
```

**Unchecked copy lengths.** A `cudaMemcpyAsync` past the end of a device
allocation does not fault at the copy. It corrupts whatever allocation follows,
and the damage appears later as wrong numbers or an illegal access in an
unrelated kernel. `DeviceBuffer` therefore checks the count against its own
size and throws.

Both were found by reading rather than by running, which is the only tool
available for CUDA on this host — and is the reason the structural checks exist.

## FP16 and BF16

Both 16-bit formats are supported, and the difference between them is not
cosmetic:

| | Exponent | Mantissa | Max | Smallest normal |
| --- | --- | --- | --- | --- |
| FP16 | 5 bits | 10 bits | 65,504 | 6.1e-5 |
| BF16 | 8 bits | 7 bits | 3.4e38 | 1.2e-38 |
| FP32 | 8 bits | 23 bits | 3.4e38 | 1.2e-38 |

BF16 has **float32's exponent range** with three fewer mantissa bits. That trade
is why it displaced FP16 on Ampere and later: activations span many orders of
magnitude, and running out of *range* produces infinities that poison everything
downstream, whereas running out of *precision* merely adds noise.

Concretely: an activation of magnitude 300 squares to 90,000 — infinite in FP16,
unremarkable in BF16. `EngineConfig.resolve_dtype` therefore prefers bfloat16
wherever the hardware supports it, and the kernels have to match, or the
configured dtype has no kernel behind it.

### Arithmetic is FP32 for both

BF16's 7-bit mantissa is *worse* than FP16's for accumulation: a sum stops
making progress after a few hundred terms rather than a few thousand. Both
formats are therefore storage-only here, with every reduction in FP32. That
costs nothing on a bandwidth-bound kernel, where the bytes moved — not the width
of the adder — set the time.

### Hardware requirement

BF16 needs **compute capability 8.0 or later**. Below that it is emulated, which
would make the BF16 path slower than the FP16 one it is meant to replace — a
silent regression, not a build failure. The default `CMAKE_CUDA_ARCHITECTURES`
therefore starts at 80, and CMake warns if it is lowered. On Turing and earlier,
use the FP16 entry points.

### One kernel, two instantiations

`reduced_precision.cuh` provides conversion traits, and each kernel is templated
over the storage type:

```cuda
template <typename T>
__global__ void rmsnorm_reduced(const T* input, ...) {
    using Convert = ReducedPrecision<T>;
    // ... conversions through Convert; accumulator is float either way
}
```

Two copies of the same kernel body would drift — the FP32-accumulator rule is
exactly the kind of detail that gets fixed in one and forgotten in the other.

## The CUDA targets are C++20, not C++17

nvcc has supported C++20 since CUDA 12.0, and this project requires it — not for
anything in the kernels, but because the CUDA targets include the portable
headers. `MemoryPool` is constrained by a `concept` and rounds with
`std::bit_ceil`, so a C++17 CUDA build fails on the *shared* code while the
`.cu` files themselves would have been fine.

That is worth stating because the failure is misleading: the errors point at
`cpp/include/cudaforge/memory_pool.hpp`, which compiles perfectly well in the
portable build, and nothing in them mentions the standard.

## Autograd

The kernels are **inference-only**. No backward kernels exist for them.

That is a deliberate scope decision — the training path uses PEFT and
transformers, not these operators — but it needed handling, because PyTorch's
default for a custom operator with no registered autograd kernel is to warn
once and then produce **silently incorrect gradients**. Training would converge
to something, just not to the right thing, with nothing pointing at the cause.

Two mechanisms address it:

1. `TORCH_LIBRARY_IMPL(cudaforge, Autograd, ...)` registers
   `autogradNotImplementedFallback` for every operator, so differentiating
   through one raises with the operator's name.
2. `cudaforge.ops` checks whether a backward pass is expected — grad mode on and
   any operand requiring grad — and routes to the reference implementation,
   which is an ordinary ATen composition and differentiates correctly.

The result is identical either way; only the implementation and the presence of
a gradient differ. `tests/python/test_autograd.py` covers both, including a
`gradcheck` against the numerical derivative.

## Error handling

Every CUDA runtime call goes through `CUDAFORGE_CHECK`. Silently ignoring a
status is the single most common source of CUDA bugs that surface thousands of
lines later as an unrelated illegal access, because the failure is asynchronous
and the context stays poisoned.

Kernel launches need two checks, and they catch different things:

| Check | Catches | Cost |
| --- | --- | --- |
| `cudaGetLastError()` | launch configuration errors — bad grid or block dimensions, too much shared memory | free; reported synchronously |
| `cudaStreamSynchronize()` | faults inside the kernel | destroys overlap |

The synchronising half is compiled in only under `CUDAFORGE_DEBUG_SYNC`. Release
builds keep the cheap check and rely on the next stream synchronisation to
surface execution faults.

`CudaError` carries the status code, not just a message, so callers can branch:
`cudaErrorMemoryAllocation` is recoverable by trimming a cache and retrying,
while `cudaErrorIllegalAddress` has already poisoned the context and is not.
`is_sticky()` makes that distinction explicit.

## Verifying kernels without a GPU

The development host cannot compile or run any of this. Three things still hold
the code accountable:

1. **Structural checks.** `scripts/check_cuda_sources.py` runs without nvcc and
   enforces seven rules:

   | Rule | Catches |
   | --- | --- |
   | `unchecked-launch` | a launch whose errors are never checked |
   | `unchecked-status` | a discarded CUDA status |
   | `device-sync` | `cudaDeviceSynchronize`, which serialises every stream |
   | `maskless-shuffle` | a warp shuffle without an explicit participation mask |
   | `divergent-barrier` | `__syncthreads()` reached conditionally |
   | `unbarriered-reduction-return` | a block reduction returning straight from shared memory |
   | `narrow-index-arithmetic` | `blockIdx.x * blockDim.x` computed in 32-bit |
   | `undeclared-convert-alias` | `Convert::` used without the `using` that defines it |

   It parses logical statements rather than physical lines, so a call the
   formatter split across lines is still matched. Each rule is tested in both
   directions — it fires on the bad shape and stays quiet on the good one —
   because a linter that silently stops detecting anything looks exactly like a
   clean codebase.

2. **Reference implementations.** Every kernel has a PyTorch or host equivalent
   that defines its semantics, and those references are tested exhaustively on
   CPU. They also *are* the fallback path, so they are exercised continuously
   rather than rotting.

3. **Compilation on real hardware.** The CUDA container and the NVIDIA-runner CI
   job build the kernels and run `tests/cuda`. See [environment.md](environment.md)
   for exactly what has and has not been executed.

## Tolerances

Floating-point addition is not associative, so a GPU tree reduction and a host
sequential sum genuinely differ. Tests compare against a **double-precision**
host reference — the mathematical answer, not another float32 accumulation with
its own error — and scale the tolerance with the term count:

```cpp
tolerance = magnitude * 1e-5 * sqrt(count);
```

Worst-case error grows as O(N·eps); for random signs it behaves closer to
O(sqrt(N)·eps), which is what this reflects. Where an input is exactly
representable (a sum of ones below 2^24), the test demands exact equality
instead — any deviation there is a bug, not accumulated error.

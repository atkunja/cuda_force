#pragma once

#include <cuda_runtime.h>

namespace cudaforge {

enum class SoftmaxKernel {
    /// Three separate passes over the row (max, sum, normalise), each reading
    /// from global memory. Straightforward and memory-bound at 3x the minimum
    /// possible traffic.
    Naive,

    /// One pass to load the row into shared memory, then max, sum and normalise
    /// from there. Global traffic drops to one read and one write, but shared
    /// memory caps the row length the kernel can handle.
    SharedMemory,

    /// Online (single-pass) softmax: max and sum are maintained together and
    /// the running sum is rescaled whenever a new maximum appears. Reads the
    /// row twice — once to compute the statistics, once to normalise — with no
    /// shared-memory capacity limit, so it handles arbitrarily long rows.
    Online,
};

/// Row-wise softmax over a `rows x cols` row-major matrix.
///
/// ## Numerical stability
///
/// The naive definition `exp(x_i) / sum(exp(x_j))` overflows for inputs above
/// roughly 88 in FP32, and attention logits routinely exceed that. Every
/// variant here subtracts the row maximum first:
///
///     softmax(x)_i = exp(x_i - m) / sum_j exp(x_j - m),  m = max(x)
///
/// The identity is exact — the `exp(-m)` factor cancels — but the largest
/// argument to `exp` becomes 0, so the largest term is exactly 1 and nothing
/// overflows. Underflow to zero for very negative terms is harmless: those
/// terms contribute nothing to the sum.
///
/// `input` and `output` may alias.
void launch_softmax(const float* input, float* output, int rows, int cols,
                    SoftmaxKernel variant, cudaStream_t stream);

/// Half-precision row-wise softmax.
///
/// Accumulation is in FP32 regardless of the storage type. An FP16 accumulator
/// has a 10-bit mantissa, so summing more than about 2048 terms of similar
/// magnitude stops making progress — each addition rounds away — and the
/// resulting distribution is visibly wrong. Storage stays FP16; only the
/// arithmetic is widened.
void launch_softmax_half(const __half* input, __half* output, int rows, int cols,
                         cudaStream_t stream);

}  // namespace cudaforge

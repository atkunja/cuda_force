#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace cudaforge {

enum class RMSNormKernel {
    /// One block per row, scalar loads, block reduction for the sum of squares.
    Naive,

    /// Same structure with `float4` loads and stores. A row is read as 128-bit
    /// transactions instead of 32-bit ones, which quarters the number of memory
    /// instructions on a kernel that is entirely memory-bound. Requires the row
    /// length and both pointers to be 16-byte aligned; the launcher checks and
    /// falls back when they are not.
    Vectorised,
};

/// Root-mean-square layer normalisation, as used by LLaMA-family transformers.
///
///     y_i = x_i / sqrt(mean(x^2) + eps) * w_i
///
/// The difference from LayerNorm is that the mean is not subtracted: RMSNorm
/// rescales without re-centring. That removes one full pass over the row and
/// one reduction, which is why transformer implementations moved to it — at
/// transformer widths the normalisation is memory-bound, so halving the
/// reductions is close to halving the cost.
///
/// `eps` is added inside the square root, matching the reference PyTorch
/// formulation. Adding it outside would change the result for small-magnitude
/// rows and break parity with the reference implementation the tests compare
/// against.
void launch_rmsnorm(const float* input, const float* weight, float* output, int rows, int cols,
                    float eps, RMSNormKernel variant, cudaStream_t stream);

/// FP16 storage with FP32 accumulation of the sum of squares.
///
/// FP16 has a maximum of 65504. A single activation of magnitude 256 squares to
/// 65536 and overflows to infinity, which then propagates through the whole
/// row. Accumulating in FP32 removes that failure mode entirely, at no
/// meaningful cost since the kernel is bandwidth-bound rather than
/// arithmetic-bound.
void launch_rmsnorm_half(const __half* input, const __half* weight, __half* output, int rows,
                         int cols, float eps, cudaStream_t stream);

}  // namespace cudaforge

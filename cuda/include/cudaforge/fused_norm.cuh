#pragma once

#include <cuda_runtime.h>

namespace cudaforge {

/// Fused residual add followed by RMSNorm.
///
///     residual_out = x + residual
///     out          = rmsnorm(residual_out) * weight
///
/// ## Why these two and not any other pair
///
/// Every transformer block is
///
///     x = x + attention(norm(x))
///     x = x + mlp(norm(x))
///
/// so a residual add is *always* immediately followed by the next block's
/// normalisation. Unfused, that is: write the sum, read it back for the sum of
/// squares, read it a third time to scale. Fused, the sum is computed once,
/// kept in registers for the reduction, and written once — while the row is
/// still resident.
///
/// | Form | Global traffic per element |
/// | --- | --- |
/// | Separate add then RMSNorm | 2 reads + 1 write, then 2 reads + 1 write |
/// | Fused | 2 reads + 2 writes |
///
/// Both outputs are needed: `out` feeds the next sublayer, and `residual_out`
/// is the value the *following* residual connection adds to. Producing only the
/// normalised result would force the caller to recompute the sum.
///
/// This is one of the higher-value fusions in inference precisely because it
/// happens twice per layer, at every layer, for every token.
///
/// `weight` has `cols` entries. `residual_out` may alias `residual`.
void launch_fused_residual_rmsnorm(const float* input, const float* residual,
                                   const float* weight, float* output, float* residual_out,
                                   int rows, int cols, float eps, cudaStream_t stream);

}  // namespace cudaforge

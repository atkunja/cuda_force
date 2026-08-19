#include "cudaforge/fused_norm.cuh"

#include "cudaforge/cuda_error.cuh"
#include "cudaforge/cuda_utils.cuh"

namespace cudaforge {
namespace {

/// One block per row.
///
/// The sum is computed once, accumulated into the sum of squares in the same
/// pass, and written to `residual_out`. The normalisation pass then reads back
/// from `residual_out` rather than recomputing the addition — the row is in L2
/// by then, and re-reading two operands to redo the add would cost more than
/// reading one.
///
/// The sum of squares accumulates in float even though the data is float,
/// because the reduction is over the *sum*, whose magnitude exceeds either
/// operand. Nothing is gained from a wider accumulator at FP32 input precision,
/// but the ordering matters: adding then squaring, not squaring then adding.
__global__ void fused_residual_rmsnorm(const float* __restrict__ input,
                                       const float* __restrict__ residual,
                                       const float* __restrict__ weight,
                                       float* __restrict__ output,
                                       float* __restrict__ residual_out, int rows, int cols,
                                       float eps) {
    __shared__ float scratch[kWarpSize];

    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) {
        return;
    }

    const auto offset = static_cast<std::size_t>(row) * cols;
    const float* row_in = input + offset;
    const float* row_residual = residual + offset;
    float* row_out = output + offset;
    float* row_residual_out = residual_out + offset;

    const int tid = static_cast<int>(threadIdx.x);
    const int step = static_cast<int>(blockDim.x);

    float sum_squares = 0.0F;
    for (int col = tid; col < cols; col += step) {
        const float summed = row_in[col] + row_residual[col];
        // Written here so the next residual connection has it, and so the
        // normalisation pass below can re-read one array instead of two.
        row_residual_out[col] = summed;
        sum_squares += summed * summed;
    }

    // Required before the reduction: another thread's contribution to the same
    // row must be visible, and block_reduce_sum's shared-memory exchange
    // assumes every thread has finished its accumulation.
    sum_squares = block_reduce_sum(sum_squares, scratch);

    const float scale = rsqrtf(sum_squares / static_cast<float>(cols) + eps);

    for (int col = tid; col < cols; col += step) {
        row_out[col] = row_residual_out[col] * scale * weight[col];
    }
}

}  // namespace

void launch_fused_residual_rmsnorm(const float* input, const float* residual,
                                   const float* weight, float* output, float* residual_out,
                                   int rows, int cols, float eps, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) {
        return;
    }
    const int block = block_size_for_row(cols);
    fused_residual_rmsnorm<<<rows, block, 0, stream>>>(input, residual, weight, output,
                                                       residual_out, rows, cols, eps);
    CUDAFORGE_CHECK_LAUNCH(stream);
}

}  // namespace cudaforge

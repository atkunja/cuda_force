#include "cudaforge/softmax.cuh"

#include <cuda_fp16.h>
#include <cfloat>

#include "cudaforge/cuda_error.cuh"
#include "cudaforge/cuda_utils.cuh"

namespace cudaforge {
namespace {

/// Three global-memory passes. Kept as the baseline the other two are measured
/// against: it is the shape a correct-but-unoptimised implementation takes.
__global__ void softmax_naive(const float* __restrict__ input, float* __restrict__ output, int rows,
                              int cols) {
    __shared__ float scratch[kWarpSize];

    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) {
        return;
    }

    const float* row_in = input + static_cast<std::size_t>(row) * cols;
    float* row_out = output + static_cast<std::size_t>(row) * cols;
    const int tid = static_cast<int>(threadIdx.x);
    const int step = static_cast<int>(blockDim.x);

    float local_max = -FLT_MAX;
    for (int col = tid; col < cols; col += step) {
        local_max = fmaxf(local_max, row_in[col]);
    }
    const float row_max = block_reduce_max(local_max, scratch, -FLT_MAX);

    float local_sum = 0.0F;
    for (int col = tid; col < cols; col += step) {
        local_sum += __expf(row_in[col] - row_max);
    }
    const float row_sum = block_reduce_sum(local_sum, scratch);

    const float inv_sum = 1.0F / row_sum;
    for (int col = tid; col < cols; col += step) {
        row_out[col] = __expf(row_in[col] - row_max) * inv_sum;
    }
}

/// Stages the row in shared memory so max, sum and normalise all read from
/// on-chip storage. Global traffic falls to the theoretical minimum of one read
/// plus one write; the cost is that `cols` must fit in the dynamic shared
/// memory the launcher requested.
__global__ void softmax_shared(const float* __restrict__ input, float* __restrict__ output,
                               int rows, int cols) {
    extern __shared__ float row_cache[];
    __shared__ float scratch[kWarpSize];

    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) {
        return;
    }

    const float* row_in = input + static_cast<std::size_t>(row) * cols;
    float* row_out = output + static_cast<std::size_t>(row) * cols;
    const int tid = static_cast<int>(threadIdx.x);
    const int step = static_cast<int>(blockDim.x);

    float local_max = -FLT_MAX;
    for (int col = tid; col < cols; col += step) {
        const float value = row_in[col];
        row_cache[col] = value;
        local_max = fmaxf(local_max, value);
    }
    __syncthreads();

    const float row_max = block_reduce_max(local_max, scratch, -FLT_MAX);

    float local_sum = 0.0F;
    for (int col = tid; col < cols; col += step) {
        const float value = __expf(row_cache[col] - row_max);
        row_cache[col] = value;  // reuse the tile to hold exponentials
        local_sum += value;
    }
    __syncthreads();

    const float row_sum = block_reduce_sum(local_sum, scratch);
    const float inv_sum = 1.0F / row_sum;

    for (int col = tid; col < cols; col += step) {
        row_out[col] = row_cache[col] * inv_sum;
    }
}

/// Online softmax.
///
/// Each thread maintains a running (max, sum) pair over the elements it visits.
/// When a larger element appears, the accumulated sum is rescaled by
/// `exp(old_max - new_max)` so it stays expressed relative to the current
/// maximum. Combining two such pairs is the same operation, which is what makes
/// the reduction across threads valid.
///
/// This removes the separate max pass, and because nothing is cached the row
/// length is unbounded — unlike the shared-memory variant.
__global__ void softmax_online(const float* __restrict__ input, float* __restrict__ output,
                               int rows, int cols) {
    __shared__ float shared_max[kWarpSize];
    __shared__ float shared_sum[kWarpSize];

    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) {
        return;
    }

    const float* row_in = input + static_cast<std::size_t>(row) * cols;
    float* row_out = output + static_cast<std::size_t>(row) * cols;
    const int tid = static_cast<int>(threadIdx.x);
    const int step = static_cast<int>(blockDim.x);

    float running_max = -FLT_MAX;
    float running_sum = 0.0F;
    for (int col = tid; col < cols; col += step) {
        const float value = row_in[col];
        if (value > running_max) {
            running_sum *= __expf(running_max - value);
            running_max = value;
        }
        running_sum += __expf(value - running_max);
    }

    // Reduce the (max, sum) pairs. The max reduces normally; the sum must be
    // rescaled into the block-wide maximum before it can be added.
    const float row_max = block_reduce_max(running_max, shared_max, -FLT_MAX);
    const float rescaled = running_sum * __expf(running_max - row_max);
    const float row_sum = block_reduce_sum(rescaled, shared_sum);

    const float inv_sum = 1.0F / row_sum;
    for (int col = tid; col < cols; col += step) {
        row_out[col] = __expf(row_in[col] - row_max) * inv_sum;
    }
}

/// FP16 storage, FP32 arithmetic. See the note in softmax.cuh for why the
/// accumulator cannot be FP16.
__global__ void softmax_half(const __half* __restrict__ input, __half* __restrict__ output,
                             int rows, int cols) {
    __shared__ float scratch[kWarpSize];

    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) {
        return;
    }

    const __half* row_in = input + static_cast<std::size_t>(row) * cols;
    __half* row_out = output + static_cast<std::size_t>(row) * cols;
    const int tid = static_cast<int>(threadIdx.x);
    const int step = static_cast<int>(blockDim.x);

    float local_max = -FLT_MAX;
    for (int col = tid; col < cols; col += step) {
        local_max = fmaxf(local_max, __half2float(row_in[col]));
    }
    const float row_max = block_reduce_max(local_max, scratch, -FLT_MAX);

    float local_sum = 0.0F;
    for (int col = tid; col < cols; col += step) {
        local_sum += __expf(__half2float(row_in[col]) - row_max);
    }
    const float row_sum = block_reduce_sum(local_sum, scratch);

    const float inv_sum = 1.0F / row_sum;
    for (int col = tid; col < cols; col += step) {
        row_out[col] = __float2half(__expf(__half2float(row_in[col]) - row_max) * inv_sum);
    }
}

/// Shared memory available per block, cached because the query is a driver
/// call and the launcher runs on the per-batch path.
int max_shared_memory_per_block() {
    static const int cached = [] {
        int device = 0;
        CUDAFORGE_CHECK(cudaGetDevice(&device));
        int bytes = 0;
        CUDAFORGE_CHECK(cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlock, device));
        return bytes;
    }();
    return cached;
}

}  // namespace

void launch_softmax(const float* input, float* output, int rows, int cols, SoftmaxKernel variant,
                    cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) {
        return;
    }

    const int block = block_size_for_row(cols);

    SoftmaxKernel selected = variant;
    if (selected == SoftmaxKernel::SharedMemory) {
        // The kernel also declares a static `scratch[kWarpSize]` for the block
        // reduction, which comes out of the same budget. Ignoring it would let
        // a launch through that the hardware then rejects, at the exact row
        // width where the dynamic request alone just fits.
        const auto required = static_cast<std::size_t>(cols) * sizeof(float) +
                              static_cast<std::size_t>(kWarpSize) * sizeof(float);
        if (required > static_cast<std::size_t>(max_shared_memory_per_block())) {
            // Falling back keeps the call correct for long rows rather than
            // failing the launch. The online variant is the right fallback
            // because it has no capacity limit at all.
            selected = SoftmaxKernel::Online;
        }
    }

    switch (selected) {
        case SoftmaxKernel::Naive:
            softmax_naive<<<rows, block, 0, stream>>>(input, output, rows, cols);
            break;
        case SoftmaxKernel::SharedMemory: {
            const std::size_t shared_bytes = static_cast<std::size_t>(cols) * sizeof(float);
            softmax_shared<<<rows, block, shared_bytes, stream>>>(input, output, rows, cols);
            break;
        }
        case SoftmaxKernel::Online:
            softmax_online<<<rows, block, 0, stream>>>(input, output, rows, cols);
            break;
    }

    CUDAFORGE_CHECK_LAUNCH(stream);
}

void launch_softmax_half(const __half* input, __half* output, int rows, int cols,
                         cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) {
        return;
    }
    const int block = block_size_for_row(cols);
    softmax_half<<<rows, block, 0, stream>>>(input, output, rows, cols);
    CUDAFORGE_CHECK_LAUNCH(stream);
}

}  // namespace cudaforge

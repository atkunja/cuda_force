#include "cudaforge/reduction.cuh"

#include <algorithm>

#include "cudaforge/cuda_error.cuh"
#include "cudaforge/cuda_utils.cuh"

namespace cudaforge {
namespace {

/// One atomic per element. The bottleneck is not the addition but the fact that
/// every thread targets the same address: the memory subsystem serialises
/// conflicting atomics, so the whole grid degenerates to sequential updates.
__global__ void reduce_sum_naive(const float* __restrict__ input, float* __restrict__ output,
                                 std::size_t count) {
    const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (index < count) {
        atomicAdd(output, input[index]);
    }
}

/// Shared-memory tree reduction with a grid-stride load phase.
///
/// The halving loop is the standard formulation with two details that matter:
///   - `index < stride` (rather than the strided-modulo form) keeps the active
///     threads contiguous, so entire warps retire together instead of every
///     warp running at partial occupancy.
///   - Shared-memory accesses are bank-conflict free, because consecutive
///     threads read consecutive addresses at every step.
__global__ void reduce_sum_shared(const float* __restrict__ input, float* __restrict__ output,
                                  std::size_t count) {
    extern __shared__ float tile[];

    const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    float local = 0.0F;
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < count;
         i += stride) {
        local += input[i];
    }

    const unsigned index = threadIdx.x;
    tile[index] = local;
    __syncthreads();

    for (unsigned half = blockDim.x / 2; half > 0; half >>= 1) {
        if (index < half) {
            tile[index] += tile[index + half];
        }
        __syncthreads();
    }

    if (index == 0) {
        atomicAdd(output, tile[0]);
    }
}

/// Warp-shuffle reduction: the shared-memory array holds one float per warp
/// instead of one per thread, and the final five halving steps happen entirely
/// in registers.
__global__ void reduce_sum_warp(const float* __restrict__ input, float* __restrict__ output,
                                std::size_t count) {
    __shared__ float warp_sums[kWarpSize];

    const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    float local = 0.0F;
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < count;
         i += stride) {
        local += input[i];
    }

    local = warp_reduce_sum(local);

    const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    if (lane == 0) {
        warp_sums[warp] = local;
    }
    __syncthreads();

    // Only the first warp finishes the job; the rest have nothing left to do.
    if (warp == 0) {
        const int warp_count = ceil_div(static_cast<int>(blockDim.x), kWarpSize);
        float value = lane < warp_count ? warp_sums[lane] : 0.0F;
        value = warp_reduce_sum(value);
        if (lane == 0) {
            atomicAdd(output, value);
        }
    }
}

/// One block per row. Rows are contiguous in a row-major matrix, so a
/// grid-stride loop within the row gives fully coalesced loads: adjacent
/// threads read adjacent floats, and the hardware merges them into the minimum
/// number of memory transactions.
__global__ void row_sum(const float* __restrict__ input, float* __restrict__ output, int rows,
                        int cols) {
    __shared__ float warp_sums[kWarpSize];

    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) {
        return;
    }

    const float* row_data = input + static_cast<std::size_t>(row) * cols;
    float local = 0.0F;
    for (int col = static_cast<int>(threadIdx.x); col < cols; col += static_cast<int>(blockDim.x)) {
        local += row_data[col];
    }

    local = block_reduce_sum(local, warp_sums);
    if (threadIdx.x == 0) {
        output[row] = local;
    }
}

}  // namespace

int reduction_grid_size(std::size_t count, int block_size) {
    int device = 0;
    CUDAFORGE_CHECK(cudaGetDevice(&device));

    int sm_count = 0;
    CUDAFORGE_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));

    // Eight blocks per SM gives the scheduler enough independent work to hide
    // memory latency without creating so many blocks that launch and tail
    // effects dominate. The cap matters for small inputs, where a fixed grid
    // would launch blocks with nothing to do.
    const int by_occupancy = std::max(1, sm_count * 8);
    const auto by_size = static_cast<int>(ceil_div(count, static_cast<std::size_t>(block_size)));
    return std::max(1, std::min(by_occupancy, std::max(1, by_size)));
}

void launch_reduce_sum(const float* input, float* output, std::size_t count,
                       ReductionKernel variant, cudaStream_t stream) {
    if (count == 0) {
        return;
    }

    constexpr int kBlock = kDefaultBlockSize;

    switch (variant) {
        case ReductionKernel::Naive: {
            // No grid-stride loop here: the naive variant is deliberately the
            // textbook one-thread-per-element formulation it is being compared
            // against.
            const auto grid = static_cast<int>(ceil_div(count, static_cast<std::size_t>(kBlock)));
            reduce_sum_naive<<<grid, kBlock, 0, stream>>>(input, output, count);
            break;
        }
        case ReductionKernel::SharedMemory: {
            const int grid = reduction_grid_size(count, kBlock);
            const std::size_t shared_bytes = kBlock * sizeof(float);
            reduce_sum_shared<<<grid, kBlock, shared_bytes, stream>>>(input, output, count);
            break;
        }
        case ReductionKernel::WarpOptimised: {
            const int grid = reduction_grid_size(count, kBlock);
            reduce_sum_warp<<<grid, kBlock, 0, stream>>>(input, output, count);
            break;
        }
    }

    CUDAFORGE_CHECK_LAUNCH(stream);
}

void launch_row_sum(const float* input, float* output, int rows, int cols, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) {
        return;
    }
    const int block = block_size_for_row(cols);
    row_sum<<<rows, block, 0, stream>>>(input, output, rows, cols);
    CUDAFORGE_CHECK_LAUNCH(stream);
}

}  // namespace cudaforge

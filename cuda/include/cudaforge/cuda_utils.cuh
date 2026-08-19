#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace cudaforge {

/// Threads per warp. Hard-coded rather than read from device properties
/// because the warp-level primitives below assume 32 at compile time; a device
/// with a different warp size would need different code, not a different
/// constant.
inline constexpr int kWarpSize = 32;

/// Chosen for the elementwise and row-wise kernels here. 256 gives four warps
/// per block, which keeps enough warps resident to hide memory latency while
/// leaving register headroom — at 1024 threads the register budget per thread
/// drops far enough to spill on several of these kernels.
inline constexpr int kDefaultBlockSize = 256;

[[nodiscard]] inline constexpr int ceil_div(int numerator, int denominator) {
    return (numerator + denominator - 1) / denominator;
}

[[nodiscard]] inline constexpr std::size_t ceil_div(std::size_t numerator,
                                                    std::size_t denominator) {
    return (numerator + denominator - 1) / denominator;
}

/// Rounds up to the next power of two, capped at the maximum block size.
/// Row-wise reductions want a block size that is a power of two so the
/// tree-reduction halving loop terminates cleanly at one active thread.
[[nodiscard]] inline int block_size_for_row(int row_length) {
    int size = kWarpSize;
    while (size < row_length && size < 1024) {
        size *= 2;
    }
    return size;
}

/// Sums a value across the 32 lanes of a warp.
///
/// `__shfl_down_sync` reads another lane's register directly. That is why this
/// beats a shared-memory reduction for the last five steps: no shared memory
/// traffic, and no `__syncthreads()` — lanes in a warp advance together, so the
/// only synchronisation needed is the mask telling the primitive which lanes
/// participate.
///
/// The mask must name every lane that will execute the shuffle. Passing
/// 0xffffffff from a divergent branch is undefined behaviour on Volta and
/// later, where lanes can genuinely be at different instructions, so callers
/// must ensure the whole warp reaches this function.
template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T value, unsigned mask = 0xffffffffU) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(mask, value, offset);
    }
    return value;
}

template<typename T>
__device__ __forceinline__ T warp_reduce_max(T value, unsigned mask = 0xffffffffU) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        const T other = __shfl_down_sync(mask, value, offset);
        value = other > value ? other : value;
    }
    return value;
}

/// Block-wide sum, leaving the result in every thread.
///
/// The two-stage shape (warp reduce, then one value per warp through shared
/// memory, then a final warp reduce) means shared memory holds at most 32
/// floats and there are only two barriers, instead of log2(blockDim) barriers
/// and a full block-sized shared array in the classic formulation.
///
/// The result is broadcast back to all threads because the callers below need
/// every thread to divide by the same sum.
template<typename T>
__device__ __forceinline__ T block_reduce_sum(T value, T* shared) {
    const int lane = threadIdx.x % kWarpSize;
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const int warp_count = ceil_div(static_cast<int>(blockDim.x), kWarpSize);

    value = warp_reduce_sum(value);
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();

    // Only the first warp participates in the final reduction; the rest would
    // read uninitialised slots.
    value = (static_cast<int>(threadIdx.x) < warp_count) ? shared[lane] : T(0);
    if (warp == 0) {
        value = warp_reduce_sum(value);
        if (lane == 0) {
            shared[0] = value;
        }
    }
    __syncthreads();
    return shared[0];
}

template<typename T>
__device__ __forceinline__ T block_reduce_max(T value, T* shared, T identity) {
    const int lane = threadIdx.x % kWarpSize;
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const int warp_count = ceil_div(static_cast<int>(blockDim.x), kWarpSize);

    value = warp_reduce_max(value);
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();

    value = (static_cast<int>(threadIdx.x) < warp_count) ? shared[lane] : identity;
    if (warp == 0) {
        value = warp_reduce_max(value);
        if (lane == 0) {
            shared[0] = value;
        }
    }
    __syncthreads();
    return shared[0];
}

}  // namespace cudaforge

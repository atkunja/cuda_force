#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

// Launch geometry lives in a CUDA-free header so it can be compiled and tested
// on a machine with no toolkit. kWarpSize, kDefaultBlockSize, ceil_div,
// block_size_for_row and grid_size_for_stride_loop all come from here.
#include "cudaforge/launch_config.hpp"

namespace cudaforge {

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

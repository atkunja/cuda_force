#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace cudaforge {

/// Which reduction implementation to launch.
///
/// All three compute the same sum. They exist together because the progression
/// between them is the clearest available illustration of what actually costs
/// time on a GPU: not arithmetic, but memory traffic and serialisation.
enum class ReductionKernel {
    /// One `atomicAdd` to global memory per input element. Every thread in the
    /// grid contends on a single address, so the hardware serialises them.
    /// Correct, and roughly as slow as a GPU reduction can be.
    Naive,

    /// Classic shared-memory tree. Each block reduces its own tile to one value
    /// through log2(blockDim) halving steps, then performs a single atomic.
    /// Global atomics drop from N to N/blockDim.
    SharedMemory,

    /// Grid-stride loads, `__shfl_down_sync` within each warp, one shared-memory
    /// exchange between warps, one atomic per block. Removes both the
    /// shared-memory traffic and the `__syncthreads()` of the final five tree
    /// levels, and the grid-stride loop keeps accesses fully coalesced
    /// regardless of input size.
    WarpOptimised,
};

/// Sums `count` floats from `input`, adding the result into `*output`.
///
/// `*output` must be zeroed before the call — every variant accumulates into it
/// rather than overwriting, so that a caller reducing several chunks can do so
/// without an extra pass. `DeviceBuffer::fill_zero` is the intended way.
///
/// Asynchronous with respect to the host: the result is only valid after the
/// stream has been synchronised or a dependent event has completed.
void launch_reduce_sum(const float* input, float* output, std::size_t count,
                       ReductionKernel variant, cudaStream_t stream);

/// Per-row sums of a `rows x cols` row-major matrix into `output[rows]`.
/// One block per row, so a row is reduced entirely in registers and shared
/// memory with no global atomics at all.
void launch_row_sum(const float* input, float* output, int rows, int cols, cudaStream_t stream);

/// Grid size used for the whole-array variants.
///
/// Fixed at a multiple of the SM count rather than derived from the input size:
/// a grid-stride loop means each block handles many elements, and launching one
/// block per tile would create far more blocks than the device can resident,
/// paying scheduling overhead for no additional parallelism.
[[nodiscard]] int reduction_grid_size(std::size_t count, int block_size);

}  // namespace cudaforge

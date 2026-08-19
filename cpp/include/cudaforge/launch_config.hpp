#pragma once

#include <cstddef>

namespace cudaforge {

/// Launch-geometry arithmetic, deliberately free of any CUDA include.
///
/// These are pure host functions that happen to compute CUDA launch
/// parameters. Keeping them here rather than in `cuda_utils.cuh` means they are
/// compiled and tested on a machine with no toolkit — and getting a grid size
/// wrong is a real bug class (a kernel that silently processes part of its
/// input) that would otherwise only be caught on hardware.
///
/// `cuda_utils.cuh` includes this header, so there is one definition.

/// Threads per warp.
///
/// Hard-coded rather than read from device properties, because the warp-level
/// primitives assume 32 at compile time: a device with a different warp size
/// would need different code, not a different constant.
inline constexpr int kWarpSize = 32;

/// Default threads per block for the elementwise and row-wise kernels.
///
/// 256 gives four warps per block — enough resident warps to hide memory
/// latency while leaving register headroom. At 1024 the per-thread register
/// budget drops far enough to spill on several of these kernels, and a spill to
/// local memory costs more than the extra occupancy returns.
inline constexpr int kDefaultBlockSize = 256;

/// Largest block CUDA permits.
inline constexpr int kMaxBlockSize = 1024;

[[nodiscard]] inline constexpr int ceil_div(int numerator, int denominator) {
    return (numerator + denominator - 1) / denominator;
}

[[nodiscard]] inline constexpr std::size_t ceil_div(std::size_t numerator,
                                                    std::size_t denominator) {
    return (numerator + denominator - 1) / denominator;
}

/// Block size for a kernel that assigns one block per row.
///
/// Rounded up to a power of two so a tree reduction's halving loop terminates
/// cleanly at one active thread, floored at one warp so the warp-level
/// primitives always have a full warp, and capped at the hardware maximum.
[[nodiscard]] inline constexpr int block_size_for_row(int row_length) {
    int size = kWarpSize;
    while (size < row_length && size < kMaxBlockSize) {
        size *= 2;
    }
    return size;
}

/// Warps in a block of `block_size` threads.
[[nodiscard]] inline constexpr int warps_per_block(int block_size) {
    return ceil_div(block_size, kWarpSize);
}

/// Grid size for a grid-stride kernel.
///
/// Proportional to the SM count rather than to the input, because each block
/// handles many elements: one block per tile would create far more blocks than
/// can be resident and pay scheduling overhead for no extra parallelism. Capped
/// by the input so a small array does not launch blocks with nothing to do.
///
/// Takes `multiprocessor_count` as a parameter rather than querying the device,
/// which is what keeps it testable off hardware.
[[nodiscard]] inline constexpr int grid_size_for_stride_loop(std::size_t element_count,
                                                             int block_size,
                                                             int multiprocessor_count,
                                                             int blocks_per_sm = 8) {
    if (element_count == 0 || block_size <= 0) {
        return 1;
    }
    const int by_occupancy = multiprocessor_count > 0 ? multiprocessor_count * blocks_per_sm : 1;
    const auto by_size =
        static_cast<int>(ceil_div(element_count, static_cast<std::size_t>(block_size)));
    const int capped = by_occupancy < by_size ? by_occupancy : by_size;
    return capped < 1 ? 1 : capped;
}

}  // namespace cudaforge

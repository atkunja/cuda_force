#include "cudaforge/tensor_core_matmul.cuh"

#include <cstddef>

#include "cudaforge/cuda_error.cuh"
#include "cudaforge/launch_config.hpp"

// mma.h is the WMMA API. It compiles for any architecture; the instructions it
// emits only exist from sm_70, and the TF32 fragment type from sm_80.
#include <mma.h>

namespace cudaforge {
namespace {

// TF32 fragments are 16x16x8, unlike the 16x16x16 of the half-precision ones.
// Getting this wrong compiles and produces silent nonsense, so it is named once
// and used everywhere.
constexpr int kFragM = 16;
constexpr int kFragN = 16;
constexpr int kFragK = 8;

/// Warps per block, in each dimension. 4x4 warps = 512 threads, each warp
/// owning one 16x16 output tile, so a block covers 64x64 of the output.
constexpr int kWarpsX = 4;
constexpr int kWarpsY = 4;

// A guard whose failure mode is an empty kernel body is worse than no guard.
// The first version of this compiled to a no-op below sm_80, launched cleanly,
// and wrote nothing — so the output was exactly zero and indistinguishable from
// a kernel that had correctly computed zero. It cost a GPU round trip to find.
//
// Every architecture this project targets is 80 or later, so anything else is a
// build the author did not intend, and saying so at compile time is the only
// honest option.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#error \
    "tensor_core_matmul.cu needs compute capability 8.0+ for TF32 WMMA. \
Build with CMAKE_CUDA_ARCHITECTURES=80 or later, or drop this file from the \
target; launch_matmul_tensor_core() already refuses older devices at runtime."
#endif

/// One warp per 16x16 tile of C.
///
/// No shared-memory staging: the fragment loads go straight to global memory
/// and the L2 does the reuse. That is a deliberate simplification — a
/// production kernel double-buffers tiles through shared memory to hide the
/// load latency — and it is why this is expected to beat the tiled FP32 kernel
/// without approaching cuBLAS.
__global__ void matmul_tf32_wmma(const float* __restrict__ a, const float* __restrict__ b,
                                 float* __restrict__ c, int m, int n, int k) {
    using namespace nvcuda;

    // Lane layout: threadIdx.x spans warps in the n direction, threadIdx.y in
    // the m direction.
    // Widened before multiplying, as everywhere else in this project. These
    // particular values cannot overflow — they are bounded by the grid — but a
    // rule that holds only where someone checked is not a rule.
    const auto column_thread =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    const auto row_warp =
        static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(blockDim.y) + threadIdx.y;

    const int warp_n = static_cast<int>(column_thread / static_cast<std::size_t>(warpSize));
    const int warp_m = static_cast<int>(row_warp);

    const int tile_row = warp_m * kFragM;
    const int tile_col = warp_n * kFragN;
    if (tile_row >= m || tile_col >= n) {
        return;
    }

    wmma::fragment<wmma::matrix_a, kFragM, kFragN, kFragK, wmma::precision::tf32, wmma::row_major>
        fragment_a;
    wmma::fragment<wmma::matrix_b, kFragM, kFragN, kFragK, wmma::precision::tf32, wmma::row_major>
        fragment_b;
    wmma::fragment<wmma::accumulator, kFragM, kFragN, kFragK, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0F);

    for (int step = 0; step < k; step += kFragK) {
        const std::size_t a_offset =
            static_cast<std::size_t>(tile_row) * static_cast<std::size_t>(k) +
            static_cast<std::size_t>(step);
        const std::size_t b_offset = static_cast<std::size_t>(step) * static_cast<std::size_t>(n) +
                                     static_cast<std::size_t>(tile_col);

        wmma::load_matrix_sync(fragment_a, a + a_offset, k);
        wmma::load_matrix_sync(fragment_b, b + b_offset, n);

        // The loads bring in FP32; the multiply needs TF32. This rounds each
        // element in place — 23 mantissa bits to 10 — and is the entire
        // precision cost of the kernel. Omitting it does not fail to compile,
        // it just produces wrong answers.
        for (int i = 0; i < fragment_a.num_elements; ++i) {
            fragment_a.x[i] = wmma::__float_to_tf32(fragment_a.x[i]);
        }
        for (int i = 0; i < fragment_b.num_elements; ++i) {
            fragment_b.x[i] = wmma::__float_to_tf32(fragment_b.x[i]);
        }

        wmma::mma_sync(accumulator, fragment_a, fragment_b, accumulator);
    }

    const std::size_t c_offset = static_cast<std::size_t>(tile_row) * static_cast<std::size_t>(n) +
                                 static_cast<std::size_t>(tile_col);
    wmma::store_matrix_sync(c + c_offset, accumulator, n, wmma::mem_row_major);
}

}  // namespace

bool tensor_cores_available() {
    static const bool cached = [] {
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) {
            return false;
        }
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) !=
            cudaSuccess) {
            return false;
        }
        return major >= 8;
    }();
    return cached;
}

bool tensor_core_shape_supported(int m, int n, int k) {
    return m > 0 && n > 0 && k > 0 && m % kFragM == 0 && n % kFragN == 0 && k % kFragK == 0;
}

bool launch_matmul_tensor_core(const float* a, const float* b, float* c, int m, int n, int k,
                               cudaStream_t stream) {
    if (!tensor_core_shape_supported(m, n, k) || !tensor_cores_available()) {
        return false;
    }

    const int tiles_m = m / kFragM;
    const int tiles_n = n / kFragN;

    // blockDim.x counts *threads* covering kWarpsX warps; blockDim.y counts
    // warps directly. Mixing those two units is the classic way to launch a
    // kernel that runs and computes a fraction of the output.
    const dim3 block(static_cast<unsigned>(kWarpsX * kWarpSize), static_cast<unsigned>(kWarpsY));
    const dim3 grid(static_cast<unsigned>(ceil_div(tiles_n, kWarpsX)),
                    static_cast<unsigned>(ceil_div(tiles_m, kWarpsY)));

    matmul_tf32_wmma<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
    CUDAFORGE_CHECK_LAUNCH(stream);
    return true;
}

}  // namespace cudaforge

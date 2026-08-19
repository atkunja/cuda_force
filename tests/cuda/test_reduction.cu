#include <catch2/catch_test_macros.hpp>

#include "cuda_test_utils.cuh"
#include "cudaforge/reduction.cuh"

using namespace cudaforge;
using namespace cudaforge::test;

namespace {

/// Tolerance for a sum of N float32 values.
///
/// Floating-point addition is not associative, so the GPU's tree reduction and
/// the host's sequential sum genuinely differ. The error grows with the number
/// of terms — worst case O(N * eps), in practice closer to O(sqrt(N) * eps)
/// for random signs — so the tolerance is scaled by the magnitude of the input
/// rather than being a fixed constant.
float sum_tolerance(std::size_t count, double magnitude) {
    return static_cast<float>(magnitude * 1e-5 * std::sqrt(static_cast<double>(count)));
}

float run_reduction(const std::vector<float>& input, ReductionKernel variant) {
    const auto output =
        run_on_device(input, 1, [&](const float* in, float* out, cudaStream_t stream) {
            launch_reduce_sum(in, out, input.size(), variant, stream);
        });
    return output[0];
}

}  // namespace

TEST_CASE("every reduction variant matches the host sum", "[cuda][reduction]") {
    for (std::size_t count : {1U, 31U, 32U, 33U, 255U, 256U, 1000U, 1U << 16, 1U << 20}) {
        const auto input = random_vector(count, /*seed=*/count);
        const double expected = reference_sum(input);
        const double magnitude = std::max(std::fabs(expected), 1.0);

        for (ReductionKernel variant : {ReductionKernel::Naive, ReductionKernel::SharedMemory,
                                        ReductionKernel::WarpOptimised}) {
            INFO("count " << count << " variant " << static_cast<int>(variant));
            const float actual = run_reduction(input, variant);
            REQUIRE(std::fabs(actual - expected) <= sum_tolerance(count, magnitude));
        }
    }
}

TEST_CASE("the variants agree with each other", "[cuda][reduction]") {
    // A tighter check than agreeing with the host: all three do a tree
    // reduction in float32, so they should be much closer to one another than
    // any of them is to an exact sum.
    const auto input = random_vector(1 << 18, /*seed=*/7);

    const float naive = run_reduction(input, ReductionKernel::Naive);
    const float shared = run_reduction(input, ReductionKernel::SharedMemory);
    const float warp = run_reduction(input, ReductionKernel::WarpOptimised);

    const float scale = std::max(std::fabs(naive), 1.0F);
    REQUIRE(std::fabs(shared - warp) <= 1e-4F * scale);
    REQUIRE(std::fabs(naive - warp) <= 1e-3F * scale);
}

TEST_CASE("a sum of ones is exact", "[cuda][reduction]") {
    // Integers below 2^24 are exactly representable in float32, so there is no
    // rounding for this input and any deviation is a genuine bug rather than
    // accumulated error.
    const std::vector<float> input(100'000, 1.0F);
    for (ReductionKernel variant :
         {ReductionKernel::SharedMemory, ReductionKernel::WarpOptimised}) {
        REQUIRE(run_reduction(input, variant) == 100'000.0F);
    }
}

TEST_CASE("an empty reduction leaves the output untouched", "[cuda][reduction]") {
    CudaStream stream;
    DeviceBuffer<float> output(1);
    output.fill_zero(stream);
    launch_reduce_sum(nullptr, output.data(), 0, ReductionKernel::WarpOptimised, stream);

    float result = -1.0F;
    output.copy_to_host(&result, 1, stream);
    stream.synchronize();
    REQUIRE(result == 0.0F);
}

TEST_CASE("row sums match the host", "[cuda][reduction]") {
    for (auto [rows, cols] :
         {std::pair{1, 1}, std::pair{4, 17}, std::pair{33, 1023}, std::pair{128, 512}}) {
        const auto input = random_vector(static_cast<std::size_t>(rows) * cols,
                                         static_cast<unsigned>(rows * cols));

        const auto actual = run_on_device(input, static_cast<std::size_t>(rows),
                                          [&](const float* in, float* out, cudaStream_t stream) {
                                              launch_row_sum(in, out, rows, cols, stream);
                                          });

        std::vector<float> expected(static_cast<std::size_t>(rows));
        for (int row = 0; row < rows; ++row) {
            double total = 0.0;
            for (int col = 0; col < cols; ++col) {
                total += input[static_cast<std::size_t>(row) * cols + col];
            }
            expected[static_cast<std::size_t>(row)] = static_cast<float>(total);
        }

        INFO("rows " << rows << " cols " << cols);
        require_all_close(actual, expected, /*relative=*/1e-4F, /*absolute=*/1e-3F);
    }
}

TEST_CASE("the grid size is bounded and non-zero", "[cuda][reduction]") {
    REQUIRE(reduction_grid_size(1, 256) >= 1);
    REQUIRE(reduction_grid_size(1 << 24, 256) >= 1);
    // A tiny input must not launch blocks with nothing to do.
    REQUIRE(reduction_grid_size(100, 256) == 1);
}

#include <catch2/catch_test_macros.hpp>

#include "cuda_test_utils.cuh"
#include "cudaforge/fused_norm.cuh"
#include "cudaforge/rmsnorm.cuh"

using namespace cudaforge;
using namespace cudaforge::test;

namespace {

struct FusedResult {
    std::vector<float> normalised;
    std::vector<float> residual_out;
};

FusedResult run_fused(const std::vector<float>& input, const std::vector<float>& residual,
                      const std::vector<float>& weight, int rows, int cols, float eps) {
    CudaStream stream;
    DeviceBuffer<float> device_in(input.size());
    DeviceBuffer<float> device_residual(residual.size());
    DeviceBuffer<float> device_weight(weight.size());
    DeviceBuffer<float> device_out(input.size());
    DeviceBuffer<float> device_residual_out(input.size());

    device_in.copy_from_host(input.data(), input.size(), stream);
    device_residual.copy_from_host(residual.data(), residual.size(), stream);
    device_weight.copy_from_host(weight.data(), weight.size(), stream);

    launch_fused_residual_rmsnorm(device_in.data(), device_residual.data(), device_weight.data(),
                                  device_out.data(), device_residual_out.data(), rows, cols, eps,
                                  stream);

    FusedResult result;
    result.normalised.resize(input.size());
    result.residual_out.resize(input.size());
    device_out.copy_to_host(result.normalised.data(), result.normalised.size(), stream);
    device_residual_out.copy_to_host(result.residual_out.data(), result.residual_out.size(),
                                     stream);
    stream.synchronize();
    return result;
}

}  // namespace

TEST_CASE("the fused kernel matches add-then-normalise", "[cuda][fused]") {
    constexpr float kEps = 1e-6F;

    for (auto [rows, cols] : {std::pair{1, 1}, std::pair{4, 17}, std::pair{8, 128},
                              std::pair{33, 1023}, std::pair{16, 4096}}) {
        const auto count = static_cast<std::size_t>(rows) * cols;
        const auto input = random_vector(count, static_cast<unsigned>(rows * 13 + cols));
        const auto residual = random_vector(count, static_cast<unsigned>(rows * 29 + cols));
        const auto weight =
            random_vector(static_cast<std::size_t>(cols), static_cast<unsigned>(cols) + 7);

        std::vector<float> summed(count);
        for (std::size_t i = 0; i < count; ++i) {
            summed[i] = input[i] + residual[i];
        }
        const auto expected = reference_rmsnorm(summed, weight, rows, cols, kEps);

        const FusedResult result = run_fused(input, residual, weight, rows, cols, kEps);

        INFO("rows " << rows << " cols " << cols);
        // The residual output must be the plain sum: it is what the *following*
        // residual connection adds to, so a normalised value there would
        // silently change the model.
        require_all_close(result.residual_out, summed, 1e-6F, 1e-7F);
        require_all_close(result.normalised, expected, 1e-4F, 1e-5F);
    }
}

TEST_CASE("the fused kernel agrees with the separate rmsnorm kernel", "[cuda][fused]") {
    // A tighter check than agreeing with a host reference: both do the same
    // float32 tree reduction, so they should agree far more closely.
    constexpr int kRows = 8;
    constexpr int kCols = 1024;
    constexpr float kEps = 1e-6F;

    const auto input = random_vector(kRows * kCols, /*seed=*/101);
    const auto residual = random_vector(kRows * kCols, /*seed=*/103);
    const auto weight = random_vector(kCols, /*seed=*/107);

    std::vector<float> summed(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        summed[i] = input[i] + residual[i];
    }

    CudaStream stream;
    DeviceBuffer<float> device_sum(summed.size());
    DeviceBuffer<float> device_weight(weight.size());
    DeviceBuffer<float> device_out(summed.size());
    device_sum.copy_from_host(summed.data(), summed.size(), stream);
    device_weight.copy_from_host(weight.data(), weight.size(), stream);
    launch_rmsnorm(device_sum.data(), device_weight.data(), device_out.data(), kRows, kCols, kEps,
                   RMSNormKernel::Naive, stream);

    std::vector<float> separate(summed.size());
    device_out.copy_to_host(separate.data(), separate.size(), stream);
    stream.synchronize();

    const FusedResult fused = run_fused(input, residual, weight, kRows, kCols, kEps);
    require_all_close(fused.normalised, separate, 1e-5F, 1e-6F);
}

TEST_CASE("a zero residual reduces to plain rmsnorm", "[cuda][fused]") {
    constexpr int kRows = 4;
    constexpr int kCols = 256;
    constexpr float kEps = 1e-6F;

    const auto input = random_vector(kRows * kCols, /*seed=*/109);
    const std::vector<float> zeros(input.size(), 0.0F);
    const auto weight = random_vector(kCols, /*seed=*/113);

    const FusedResult result = run_fused(input, zeros, weight, kRows, kCols, kEps);
    require_all_close(result.residual_out, input, 1e-7F, 1e-7F);
    require_all_close(result.normalised, reference_rmsnorm(input, weight, kRows, kCols, kEps),
                      1e-4F, 1e-5F);
}

TEST_CASE("the operands are symmetric", "[cuda][fused]") {
    // Addition commutes, so swapping them must not change anything.
    constexpr int kRows = 4;
    constexpr int kCols = 128;

    const auto a = random_vector(kRows * kCols, /*seed=*/127);
    const auto b = random_vector(kRows * kCols, /*seed=*/131);
    const auto weight = random_vector(kCols, /*seed=*/137);

    const FusedResult first = run_fused(a, b, weight, kRows, kCols, 1e-6F);
    const FusedResult second = run_fused(b, a, weight, kRows, kCols, 1e-6F);
    require_all_close(first.normalised, second.normalised, 1e-6F, 1e-7F);
}

TEST_CASE("an empty fused launch is a no-op", "[cuda][fused]") {
    CudaStream stream;
    REQUIRE_NOTHROW(launch_fused_residual_rmsnorm(nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0,
                                                  1e-6F, stream));
}

#include <catch2/catch_test_macros.hpp>

#include <cuda_fp16.h>

#include "cuda_test_utils.cuh"
#include "cudaforge/rmsnorm.cuh"

using namespace cudaforge;
using namespace cudaforge::test;

namespace {

std::vector<float> run_rmsnorm(const std::vector<float>& input,
                               const std::vector<float>& weight, int rows, int cols,
                               float eps, RMSNormKernel variant) {
    CudaStream stream;
    DeviceBuffer<float> device_in(input.size());
    DeviceBuffer<float> device_weight(weight.size());
    DeviceBuffer<float> device_out(input.size());

    device_in.copy_from_host(input.data(), input.size(), stream);
    device_weight.copy_from_host(weight.data(), weight.size(), stream);

    launch_rmsnorm(device_in.data(), device_weight.data(), device_out.data(), rows, cols, eps,
                   variant, stream);

    std::vector<float> output(input.size());
    device_out.copy_to_host(output.data(), output.size(), stream);
    stream.synchronize();
    return output;
}

}  // namespace

TEST_CASE("rmsnorm matches the host reference", "[cuda][rmsnorm]") {
    constexpr float kEps = 1e-6F;

    // Sizes deliberately mix multiples of four (vectorisable) with sizes that
    // are not, so the launcher's fallback path is exercised too.
    for (auto [rows, cols] : {std::pair{1, 4}, std::pair{2, 17}, std::pair{4, 128},
                              std::pair{8, 1023}, std::pair{16, 4096}, std::pair{3, 2049}}) {
        const auto input = random_vector(static_cast<std::size_t>(rows) * cols,
                                         static_cast<unsigned>(rows * 17 + cols));
        const auto weight = random_vector(static_cast<std::size_t>(cols),
                                          static_cast<unsigned>(cols));
        const auto expected = reference_rmsnorm(input, weight, rows, cols, kEps);

        for (RMSNormKernel variant : {RMSNormKernel::Naive, RMSNormKernel::Vectorised}) {
            INFO("rows " << rows << " cols " << cols << " variant "
                         << static_cast<int>(variant));
            require_all_close(run_rmsnorm(input, weight, rows, cols, kEps, variant), expected,
                              /*relative=*/1e-4F, /*absolute=*/1e-5F);
        }
    }
}

TEST_CASE("the vectorised and scalar kernels agree", "[cuda][rmsnorm]") {
    // Both are float32 tree reductions over the same data, so they should agree
    // far more closely than either agrees with a double-precision reference.
    constexpr int kRows = 8;
    constexpr int kCols = 1024;
    const auto input = random_vector(kRows * kCols, /*seed=*/23);
    const auto weight = random_vector(kCols, /*seed=*/29);

    require_all_close(run_rmsnorm(input, weight, kRows, kCols, 1e-6F, RMSNormKernel::Naive),
                      run_rmsnorm(input, weight, kRows, kCols, 1e-6F,
                                  RMSNormKernel::Vectorised),
                      /*relative=*/1e-5F, /*absolute=*/1e-6F);
}

TEST_CASE("unit input with unit weight is the identity", "[cuda][rmsnorm]") {
    // Every element is 1, so the RMS is 1 and the output is the weight.
    constexpr int kRows = 4;
    constexpr int kCols = 256;
    const std::vector<float> input(kRows * kCols, 1.0F);
    const std::vector<float> weight(kCols, 1.0F);

    const auto output = run_rmsnorm(input, weight, kRows, kCols, 0.0F,
                                    RMSNormKernel::Vectorised);
    for (float value : output) {
        REQUIRE(close(value, 1.0F, 1e-5F, 1e-6F));
    }
}

TEST_CASE("rmsnorm is scale invariant", "[cuda][rmsnorm]") {
    // Dividing by the RMS is what makes this a normalisation: scaling the input
    // must leave the output unchanged as eps goes to zero.
    constexpr int kRows = 4;
    constexpr int kCols = 512;
    const auto input = random_vector(kRows * kCols, /*seed=*/31);
    const std::vector<float> weight(kCols, 1.0F);

    std::vector<float> scaled(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        scaled[i] = input[i] * 10.0F;
    }

    require_all_close(
        run_rmsnorm(scaled, weight, kRows, kCols, 1e-12F, RMSNormKernel::Vectorised),
        run_rmsnorm(input, weight, kRows, kCols, 1e-12F, RMSNormKernel::Vectorised),
        /*relative=*/1e-4F, /*absolute=*/1e-5F);
}

TEST_CASE("half-precision rmsnorm survives magnitudes that overflow fp16 squares",
          "[cuda][rmsnorm][fp16]") {
    // 300^2 = 90000 exceeds fp16's maximum of 65504. Accumulating the sum of
    // squares in fp32 is what prevents the whole row becoming infinity.
    constexpr int kRows = 2;
    constexpr int kCols = 128;

    std::vector<__half> host_in(static_cast<std::size_t>(kRows) * kCols);
    std::vector<__half> host_weight(kCols);
    for (auto& value : host_in) {
        value = __float2half(300.0F);
    }
    for (auto& value : host_weight) {
        value = __float2half(1.0F);
    }

    CudaStream stream;
    DeviceBuffer<__half> device_in(host_in.size());
    DeviceBuffer<__half> device_weight(host_weight.size());
    DeviceBuffer<__half> device_out(host_in.size());

    device_in.copy_from_host(host_in.data(), host_in.size(), stream);
    device_weight.copy_from_host(host_weight.data(), host_weight.size(), stream);
    launch_rmsnorm_half(device_in.data(), device_weight.data(), device_out.data(), kRows,
                        kCols, 1e-6F, stream);

    std::vector<__half> host_out(host_in.size());
    device_out.copy_to_host(host_out.data(), host_out.size(), stream);
    stream.synchronize();

    for (const __half& value : host_out) {
        const float restored = __half2float(value);
        REQUIRE(std::isfinite(restored));
        REQUIRE(close(restored, 1.0F, 2e-2F, 1e-2F));
    }
}

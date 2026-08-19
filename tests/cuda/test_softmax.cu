#include <catch2/catch_test_macros.hpp>

#include <cuda_fp16.h>

#include "cuda_test_utils.cuh"
#include "cudaforge/softmax.cuh"

using namespace cudaforge;
using namespace cudaforge::test;

namespace {

std::vector<float> run_softmax(const std::vector<float>& input, int rows, int cols,
                               SoftmaxKernel variant) {
    return run_on_device(input, input.size(),
                         [&](const float* in, float* out, cudaStream_t stream) {
                             launch_softmax(in, out, rows, cols, variant, stream);
                         });
}

constexpr SoftmaxKernel kVariants[] = {
    SoftmaxKernel::Naive,
    SoftmaxKernel::SharedMemory,
    SoftmaxKernel::Online,
};

}  // namespace

TEST_CASE("every softmax variant matches the host", "[cuda][softmax]") {
    for (auto [rows, cols] : {std::pair{1, 1}, std::pair{1, 32}, std::pair{4, 17},
                              std::pair{8, 256}, std::pair{3, 1025}, std::pair{64, 4096}}) {
        const auto input = random_vector(static_cast<std::size_t>(rows) * cols,
                                         static_cast<unsigned>(rows * 31 + cols));
        const auto expected = reference_softmax(input, rows, cols);

        for (SoftmaxKernel variant : kVariants) {
            INFO("rows " << rows << " cols " << cols << " variant "
                         << static_cast<int>(variant));
            require_all_close(run_softmax(input, rows, cols, variant), expected,
                              /*relative=*/1e-4F, /*absolute=*/1e-6F);
        }
    }
}

TEST_CASE("softmax rows sum to one", "[cuda][softmax]") {
    constexpr int kRows = 16;
    constexpr int kCols = 512;
    const auto input = random_vector(kRows * kCols, /*seed=*/11, /*scale=*/5.0F);

    for (SoftmaxKernel variant : kVariants) {
        const auto output = run_softmax(input, kRows, kCols, variant);
        for (int row = 0; row < kRows; ++row) {
            double total = 0.0;
            for (int col = 0; col < kCols; ++col) {
                total += output[static_cast<std::size_t>(row) * kCols + col];
            }
            INFO("variant " << static_cast<int>(variant) << " row " << row);
            REQUIRE(std::fabs(total - 1.0) < 1e-4);
        }
    }
}

TEST_CASE("softmax is stable for logits that would overflow exp", "[cuda][softmax]") {
    // exp(1000) is infinity in float32. Subtracting the row maximum is what
    // keeps this finite, and attention logits genuinely reach these values.
    constexpr int kRows = 2;
    constexpr int kCols = 4;
    const std::vector<float> input = {
        1000.0F, 1000.0F, 1000.0F, 1000.0F,
        -1000.0F, 0.0F, 500.0F, 1000.0F,
    };

    for (SoftmaxKernel variant : kVariants) {
        const auto output = run_softmax(input, kRows, kCols, variant);
        INFO("variant " << static_cast<int>(variant));
        for (float value : output) {
            REQUIRE(std::isfinite(value));
        }
        // A uniform row must come out uniform.
        REQUIRE(close(output[0], 0.25F, 1e-5F, 1e-6F));
        // The dominant logit takes essentially all the mass.
        REQUIRE(output[7] > 0.99F);
    }
}

TEST_CASE("softmax is invariant to a constant shift", "[cuda][softmax]") {
    constexpr int kRows = 4;
    constexpr int kCols = 128;
    const auto input = random_vector(kRows * kCols, /*seed=*/3);

    std::vector<float> shifted(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        shifted[i] = input[i] + 12.5F;
    }

    for (SoftmaxKernel variant : kVariants) {
        INFO("variant " << static_cast<int>(variant));
        require_all_close(run_softmax(shifted, kRows, kCols, variant),
                          run_softmax(input, kRows, kCols, variant), 1e-5F, 1e-7F);
    }
}

TEST_CASE("the shared-memory variant falls back for very long rows", "[cuda][softmax]") {
    // Beyond the shared-memory limit the launcher switches to the online
    // variant rather than failing the launch. The result must still be correct.
    constexpr int kRows = 2;
    constexpr int kCols = 32768;
    const auto input = random_vector(static_cast<std::size_t>(kRows) * kCols, /*seed=*/5);

    require_all_close(run_softmax(input, kRows, kCols, SoftmaxKernel::SharedMemory),
                      reference_softmax(input, kRows, kCols), 1e-4F, 1e-6F);
}

TEST_CASE("half-precision softmax matches within fp16 tolerance", "[cuda][softmax][fp16]") {
    constexpr int kRows = 4;
    constexpr int kCols = 256;
    const auto input = random_vector(kRows * kCols, /*seed=*/13);
    const auto expected = reference_softmax(input, kRows, kCols);

    std::vector<__half> host_in(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        host_in[i] = __float2half(input[i]);
    }

    CudaStream stream;
    DeviceBuffer<__half> device_in(host_in.size());
    DeviceBuffer<__half> device_out(host_in.size());
    device_in.copy_from_host(host_in.data(), host_in.size(), stream);
    launch_softmax_half(device_in.data(), device_out.data(), kRows, kCols, stream);

    std::vector<__half> host_out(host_in.size());
    device_out.copy_to_host(host_out.data(), host_out.size(), stream);
    stream.synchronize();

    // fp16 has a 10-bit mantissa, so roughly 1e-3 relative precision. The
    // absolute floor covers values near zero, where relative error is
    // meaningless.
    for (std::size_t i = 0; i < host_out.size(); ++i) {
        INFO("element " << i);
        REQUIRE(close(__half2float(host_out[i]), expected[i], 2e-2F, 1e-3F));
    }
}

#include <catch2/catch_test_macros.hpp>

#include <cuda_fp16.h>

#include <cmath>

#include "cuda_test_utils.cuh"
#include "cudaforge/activations.cuh"

using namespace cudaforge;
using namespace cudaforge::test;

namespace {

float reference_silu(float x) { return x / (1.0F + std::exp(-x)); }

float reference_gelu(float x) {
    constexpr float kSqrt2OverPi = 0.7978845608028654F;
    constexpr float kCoefficient = 0.044715F;
    return 0.5F * x * (1.0F + std::tanh(kSqrt2OverPi * (x + kCoefficient * x * x * x)));
}

std::vector<float> run_swiglu(const std::vector<float>& gate, const std::vector<float>& up,
                              SwiGLUKernel variant) {
    CudaStream stream;
    DeviceBuffer<float> device_gate(gate.size());
    DeviceBuffer<float> device_up(up.size());
    DeviceBuffer<float> device_out(gate.size());

    device_gate.copy_from_host(gate.data(), gate.size(), stream);
    device_up.copy_from_host(up.data(), up.size(), stream);
    launch_swiglu(device_gate.data(), device_up.data(), device_out.data(),
                  static_cast<int>(gate.size()), variant, stream);

    std::vector<float> output(gate.size());
    device_out.copy_to_host(output.data(), output.size(), stream);
    stream.synchronize();
    return output;
}

}  // namespace

TEST_CASE("silu matches the host reference", "[cuda][activation]") {
    for (int count : {1, 63, 64, 65, 4096, 1 << 20}) {
        const auto input = random_vector(static_cast<std::size_t>(count),
                                         static_cast<unsigned>(count), 4.0F);

        const auto actual = run_on_device(
            input, input.size(), [&](const float* in, float* out, cudaStream_t stream) {
                launch_silu(in, out, count, stream);
            });

        std::vector<float> expected(input.size());
        for (std::size_t i = 0; i < input.size(); ++i) {
            expected[i] = reference_silu(input[i]);
        }

        INFO("count " << count);
        // __expf is the fast intrinsic: a few ULP less accurate than expf, well
        // inside this tolerance and a single instruction rather than a
        // polynomial.
        require_all_close(actual, expected, /*relative=*/1e-4F, /*absolute=*/1e-5F);
    }
}

TEST_CASE("silu is zero at zero and monotone for large inputs", "[cuda][activation]") {
    const std::vector<float> input = {0.0F, -20.0F, 20.0F};
    const auto output = run_on_device(
        input, input.size(), [&](const float* in, float* out, cudaStream_t stream) {
            launch_silu(in, out, 3, stream);
        });

    REQUIRE(output[0] == 0.0F);
    // Large negative saturates toward zero; large positive approaches identity.
    REQUIRE(std::fabs(output[1]) < 1e-6F);
    REQUIRE(close(output[2], 20.0F, 1e-5F, 1e-5F));
}

TEST_CASE("gelu matches the tanh approximation", "[cuda][activation]") {
    const auto input = random_vector(4096, /*seed=*/71, 3.0F);
    const auto actual = run_on_device(
        input, input.size(), [&](const float* in, float* out, cudaStream_t stream) {
            launch_gelu(in, out, 4096, stream);
        });

    std::vector<float> expected(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        expected[i] = reference_gelu(input[i]);
    }
    require_all_close(actual, expected, 1e-4F, 1e-5F);
}

TEST_CASE("both swiglu variants match the reference", "[cuda][activation]") {
    // Counts deliberately mix multiples of four with sizes that are not, so the
    // vectorised path and its scalar fallback are both exercised.
    for (int count : {1, 3, 64, 65, 1023, 1024, 1 << 20}) {
        const auto gate = random_vector(static_cast<std::size_t>(count),
                                        static_cast<unsigned>(count), 3.0F);
        const auto up = random_vector(static_cast<std::size_t>(count),
                                      static_cast<unsigned>(count) + 1, 3.0F);

        std::vector<float> expected(gate.size());
        for (std::size_t i = 0; i < gate.size(); ++i) {
            expected[i] = reference_silu(gate[i]) * up[i];
        }

        for (SwiGLUKernel variant : {SwiGLUKernel::Scalar, SwiGLUKernel::Vectorised}) {
            INFO("count " << count << " variant " << static_cast<int>(variant));
            require_all_close(run_swiglu(gate, up, variant), expected, 1e-4F, 1e-5F);
        }
    }
}

TEST_CASE("swiglu with a unit up-projection is silu", "[cuda][activation]") {
    const auto gate = random_vector(1024, /*seed=*/73, 4.0F);
    const std::vector<float> ones(gate.size(), 1.0F);

    const auto swiglu = run_swiglu(gate, ones, SwiGLUKernel::Vectorised);
    const auto silu = run_on_device(
        gate, gate.size(), [&](const float* in, float* out, cudaStream_t stream) {
            launch_silu(in, out, static_cast<int>(gate.size()), stream);
        });

    require_all_close(swiglu, silu, 1e-6F, 1e-7F);
}

TEST_CASE("swiglu is linear in the up-projection", "[cuda][activation]") {
    const auto gate = random_vector(512, /*seed=*/77, 3.0F);
    const auto up = random_vector(512, /*seed=*/79, 3.0F);

    std::vector<float> doubled(up.size());
    for (std::size_t i = 0; i < up.size(); ++i) {
        doubled[i] = up[i] * 2.0F;
    }

    const auto once = run_swiglu(gate, up, SwiGLUKernel::Vectorised);
    const auto twice = run_swiglu(gate, doubled, SwiGLUKernel::Vectorised);

    std::vector<float> expected(once.size());
    for (std::size_t i = 0; i < once.size(); ++i) {
        expected[i] = once[i] * 2.0F;
    }
    require_all_close(twice, expected, 1e-5F, 1e-6F);
}

TEST_CASE("half-precision swiglu keeps the negative tail", "[cuda][activation][fp16]") {
    // The sigmoid is evaluated in FP32: exp() of a moderately negative input
    // underflows a 10-bit mantissa long before it underflows FP32, which would
    // flatten the activation's negative side to exactly zero.
    constexpr int kCount = 256;
    std::vector<__half> gate(kCount);
    std::vector<__half> up(kCount);
    std::vector<float> expected(kCount);

    for (int i = 0; i < kCount; ++i) {
        const float value = -12.0F + 24.0F * static_cast<float>(i) / (kCount - 1);
        gate[static_cast<std::size_t>(i)] = __float2half(value);
        up[static_cast<std::size_t>(i)] = __float2half(1.0F);
        expected[static_cast<std::size_t>(i)] = reference_silu(value);
    }

    CudaStream stream;
    DeviceBuffer<__half> device_gate(gate.size());
    DeviceBuffer<__half> device_up(up.size());
    DeviceBuffer<__half> device_out(gate.size());
    device_gate.copy_from_host(gate.data(), gate.size(), stream);
    device_up.copy_from_host(up.data(), up.size(), stream);
    launch_swiglu_half(device_gate.data(), device_up.data(), device_out.data(), kCount,
                       stream);

    std::vector<__half> output(gate.size());
    device_out.copy_to_host(output.data(), output.size(), stream);
    stream.synchronize();

    for (std::size_t i = 0; i < output.size(); ++i) {
        INFO("element " << i);
        REQUIRE(close(__half2float(output[i]), expected[i], 2e-2F, 1e-3F));
    }
}

TEST_CASE("an empty activation launch is a no-op", "[cuda][activation]") {
    CudaStream stream;
    REQUIRE_NOTHROW(launch_silu(nullptr, nullptr, 0, stream));
    REQUIRE_NOTHROW(launch_gelu(nullptr, nullptr, 0, stream));
    REQUIRE_NOTHROW(
        launch_swiglu(nullptr, nullptr, nullptr, 0, SwiGLUKernel::Vectorised, stream));
}

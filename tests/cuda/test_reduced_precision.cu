#include <catch2/catch_test_macros.hpp>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>

#include "cuda_test_utils.cuh"
#include "cudaforge/activations.cuh"
#include "cudaforge/rmsnorm.cuh"
#include "cudaforge/softmax.cuh"

using namespace cudaforge;
using namespace cudaforge::test;

// FP16 and BF16 share one kernel body per operation, differing only in the
// conversions. These tests cover the BF16 instantiations and, more importantly,
// the property that motivates BF16 at all: float32's exponent range, which
// makes magnitudes that overflow FP16 unremarkable.

namespace {

/// BF16 keeps 8 exponent bits and 7 mantissa bits, so roughly 2^-8 relative
/// precision. Tolerances are correspondingly looser than FP16's.
constexpr float kBf16Relative = 8e-2F;
constexpr float kBf16Absolute = 1e-2F;

std::vector<__nv_bfloat16> to_bf16(const std::vector<float>& values) {
    std::vector<__nv_bfloat16> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        out[i] = __float2bfloat16(values[i]);
    }
    return out;
}

}  // namespace

TEST_CASE("bf16 rmsnorm matches the host reference", "[cuda][bf16]") {
    constexpr int kRows = 4;
    constexpr int kCols = 256;
    constexpr float kEps = 1e-6F;

    const auto input = random_vector(kRows * kCols, /*seed=*/211);
    const auto weight = random_vector(kCols, /*seed=*/223);
    const auto expected = reference_rmsnorm(input, weight, kRows, kCols, kEps);

    const auto host_in = to_bf16(input);
    const auto host_weight = to_bf16(weight);

    CudaStream stream;
    DeviceBuffer<__nv_bfloat16> device_in(host_in.size());
    DeviceBuffer<__nv_bfloat16> device_weight(host_weight.size());
    DeviceBuffer<__nv_bfloat16> device_out(host_in.size());
    device_in.copy_from_host(host_in.data(), host_in.size(), stream);
    device_weight.copy_from_host(host_weight.data(), host_weight.size(), stream);
    launch_rmsnorm_bf16(device_in.data(), device_weight.data(), device_out.data(), kRows, kCols,
                        kEps, stream);

    std::vector<__nv_bfloat16> host_out(host_in.size());
    device_out.copy_to_host(host_out.data(), host_out.size(), stream);
    stream.synchronize();

    for (std::size_t i = 0; i < host_out.size(); ++i) {
        INFO("element " << i);
        REQUIRE(close(__bfloat162float(host_out[i]), expected[i], kBf16Relative, kBf16Absolute));
    }
}

TEST_CASE("bf16 rmsnorm handles magnitudes that overflow fp16", "[cuda][bf16]") {
    // 300^2 = 90,000 exceeds FP16's maximum of 65,504 — the failure the FP32
    // accumulator exists to prevent there. In BF16 the value is unremarkable,
    // which is the whole reason the format exists.
    constexpr int kRows = 2;
    constexpr int kCols = 128;

    std::vector<__nv_bfloat16> host_in(static_cast<std::size_t>(kRows) * kCols);
    std::vector<__nv_bfloat16> host_weight(kCols);
    for (auto& value : host_in) {
        value = __float2bfloat16(300.0F);
    }
    for (auto& value : host_weight) {
        value = __float2bfloat16(1.0F);
    }

    CudaStream stream;
    DeviceBuffer<__nv_bfloat16> device_in(host_in.size());
    DeviceBuffer<__nv_bfloat16> device_weight(host_weight.size());
    DeviceBuffer<__nv_bfloat16> device_out(host_in.size());
    device_in.copy_from_host(host_in.data(), host_in.size(), stream);
    device_weight.copy_from_host(host_weight.data(), host_weight.size(), stream);
    launch_rmsnorm_bf16(device_in.data(), device_weight.data(), device_out.data(), kRows, kCols,
                        1e-6F, stream);

    std::vector<__nv_bfloat16> host_out(host_in.size());
    device_out.copy_to_host(host_out.data(), host_out.size(), stream);
    stream.synchronize();

    for (const __nv_bfloat16& value : host_out) {
        const float restored = __bfloat162float(value);
        REQUIRE(std::isfinite(restored));
        REQUIRE(close(restored, 1.0F, kBf16Relative, kBf16Absolute));
    }
}

TEST_CASE("bf16 softmax rows sum to one", "[cuda][bf16]") {
    constexpr int kRows = 4;
    constexpr int kCols = 128;

    const auto input = random_vector(kRows * kCols, /*seed=*/227);
    const auto host_in = to_bf16(input);

    CudaStream stream;
    DeviceBuffer<__nv_bfloat16> device_in(host_in.size());
    DeviceBuffer<__nv_bfloat16> device_out(host_in.size());
    device_in.copy_from_host(host_in.data(), host_in.size(), stream);
    launch_softmax_bf16(device_in.data(), device_out.data(), kRows, kCols, stream);

    std::vector<__nv_bfloat16> host_out(host_in.size());
    device_out.copy_to_host(host_out.data(), host_out.size(), stream);
    stream.synchronize();

    for (int row = 0; row < kRows; ++row) {
        double total = 0.0;
        for (int col = 0; col < kCols; ++col) {
            total += __bfloat162float(host_out[static_cast<std::size_t>(row) * kCols + col]);
        }
        INFO("row " << row);
        // Looser than FP32: 7 mantissa bits accumulated over 128 terms.
        REQUIRE(std::fabs(total - 1.0) < 5e-2);
    }
}

TEST_CASE("bf16 softmax survives logits fp16 cannot represent", "[cuda][bf16]") {
    // FP16's maximum is 65,504; a logit of 100,000 is not representable at all.
    // BF16 has float32's exponent range, so it is.
    constexpr int kRows = 1;
    constexpr int kCols = 4;

    std::vector<__nv_bfloat16> host_in(kCols);
    const float logits[kCols] = {100000.0F, 99999.0F, -100000.0F, 0.0F};
    for (int i = 0; i < kCols; ++i) {
        host_in[static_cast<std::size_t>(i)] = __float2bfloat16(logits[i]);
        REQUIRE(std::isfinite(__bfloat162float(host_in[static_cast<std::size_t>(i)])));
    }

    CudaStream stream;
    DeviceBuffer<__nv_bfloat16> device_in(host_in.size());
    DeviceBuffer<__nv_bfloat16> device_out(host_in.size());
    device_in.copy_from_host(host_in.data(), host_in.size(), stream);
    launch_softmax_bf16(device_in.data(), device_out.data(), kRows, kCols, stream);

    std::vector<__nv_bfloat16> host_out(host_in.size());
    device_out.copy_to_host(host_out.data(), host_out.size(), stream);
    stream.synchronize();

    for (const __nv_bfloat16& value : host_out) {
        REQUIRE(std::isfinite(__bfloat162float(value)));
    }
    // The dominant logit takes essentially all the mass.
    REQUIRE(__bfloat162float(host_out[0]) > 0.5F);
}

TEST_CASE("bf16 swiglu matches the host reference", "[cuda][bf16]") {
    constexpr int kCount = 1024;

    const auto gate = random_vector(kCount, /*seed=*/229, 3.0F);
    const auto up = random_vector(kCount, /*seed=*/233, 3.0F);
    const auto host_gate = to_bf16(gate);
    const auto host_up = to_bf16(up);

    CudaStream stream;
    DeviceBuffer<__nv_bfloat16> device_gate(host_gate.size());
    DeviceBuffer<__nv_bfloat16> device_up(host_up.size());
    DeviceBuffer<__nv_bfloat16> device_out(host_gate.size());
    device_gate.copy_from_host(host_gate.data(), host_gate.size(), stream);
    device_up.copy_from_host(host_up.data(), host_up.size(), stream);
    launch_swiglu_bf16(device_gate.data(), device_up.data(), device_out.data(), kCount, stream);

    std::vector<__nv_bfloat16> host_out(host_gate.size());
    device_out.copy_to_host(host_out.data(), host_out.size(), stream);
    stream.synchronize();

    for (std::size_t i = 0; i < host_out.size(); ++i) {
        // Compared against the reference applied to the *rounded* inputs, since
        // rounding to BF16 is part of what the kernel receives.
        const float rounded_gate = __bfloat162float(host_gate[i]);
        const float rounded_up = __bfloat162float(host_up[i]);
        const float expected = rounded_gate / (1.0F + std::exp(-rounded_gate)) * rounded_up;

        INFO("element " << i);
        REQUIRE(close(__bfloat162float(host_out[i]), expected, kBf16Relative, kBf16Absolute));
    }
}

TEST_CASE("bf16 preserves the negative tail of the activation", "[cuda][bf16]") {
    // The property the FP32 sigmoid protects: exp() of a moderately negative
    // input must not underflow to exactly zero, or the activation flattens.
    constexpr int kCount = 64;

    std::vector<__nv_bfloat16> gate(kCount);
    std::vector<__nv_bfloat16> up(kCount);
    for (int i = 0; i < kCount; ++i) {
        gate[static_cast<std::size_t>(i)] = __float2bfloat16(-8.0F + 0.05F * i);
        up[static_cast<std::size_t>(i)] = __float2bfloat16(1.0F);
    }

    CudaStream stream;
    DeviceBuffer<__nv_bfloat16> device_gate(gate.size());
    DeviceBuffer<__nv_bfloat16> device_up(up.size());
    DeviceBuffer<__nv_bfloat16> device_out(gate.size());
    device_gate.copy_from_host(gate.data(), gate.size(), stream);
    device_up.copy_from_host(up.data(), up.size(), stream);
    launch_swiglu_bf16(device_gate.data(), device_up.data(), device_out.data(), kCount, stream);

    std::vector<__nv_bfloat16> host_out(gate.size());
    device_out.copy_to_host(host_out.data(), host_out.size(), stream);
    stream.synchronize();

    int non_zero = 0;
    for (const __nv_bfloat16& value : host_out) {
        if (__bfloat162float(value) != 0.0F) {
            ++non_zero;
        }
    }
    // A flattened tail would be all zeros; the curve must remain a curve.
    REQUIRE(non_zero > kCount / 2);
}

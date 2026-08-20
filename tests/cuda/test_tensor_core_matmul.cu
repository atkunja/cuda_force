#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "cuda_test_utils.cuh"
#include "cudaforge/lora_linear.cuh"
#include "cudaforge/tensor_core_matmul.cuh"

using namespace cudaforge;
using namespace cudaforge::test;

namespace {

std::vector<float> reference_matmul(const std::vector<float>& a, const std::vector<float>& b, int m,
                                    int n, int k) {
    std::vector<float> out(static_cast<std::size_t>(m) * n, 0.0F);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            double sum = 0.0;
            for (int step = 0; step < k; ++step) {
                sum += static_cast<double>(a[static_cast<std::size_t>(row) * k + step]) *
                       static_cast<double>(b[static_cast<std::size_t>(step) * n + col]);
            }
            out[static_cast<std::size_t>(row) * n + col] = static_cast<float>(sum);
        }
    }
    return out;
}

/// Runs the tensor-core kernel, returning empty when it declined the shape.
std::vector<float> run_tensor_core(const std::vector<float>& a, const std::vector<float>& b, int m,
                                   int n, int k) {
    CudaStream stream;
    DeviceBuffer<float> device_a(a.size());
    DeviceBuffer<float> device_b(b.size());
    DeviceBuffer<float> device_c(static_cast<std::size_t>(m) * n);

    device_a.copy_from_host(a.data(), a.size(), stream);
    device_b.copy_from_host(b.data(), b.size(), stream);
    device_c.fill_zero(stream);

    if (!launch_matmul_tensor_core(device_a.data(), device_b.data(), device_c.data(), m, n, k,
                                   stream)) {
        return {};
    }

    std::vector<float> out(static_cast<std::size_t>(m) * n);
    device_c.copy_to_host(out.data(), out.size(), stream);
    stream.synchronize();
    return out;
}

}  // namespace

TEST_CASE("the tensor-core matmul matches a host reference", "[cuda][matmul][tensorcore]") {
    if (!tensor_cores_available()) {
        SKIP("device is older than sm_80, so it has no TF32 tensor cores");
    }

    // All dimensions are multiples of the 16x16x8 fragment.
    for (auto [m, n, k] : {std::tuple{16, 16, 8}, std::tuple{64, 32, 16}, std::tuple{128, 128, 64},
                           std::tuple{256, 64, 128}}) {
        const auto a = random_vector(static_cast<std::size_t>(m) * k, static_cast<unsigned>(m + k));
        const auto b = random_vector(static_cast<std::size_t>(k) * n, static_cast<unsigned>(n + 7));
        const auto expected = reference_matmul(a, b, m, n, k);
        const auto actual = run_tensor_core(a, b, m, n, k);

        REQUIRE(actual.size() == expected.size());
        for (std::size_t i = 0; i < expected.size(); ++i) {
            INFO("shape " << m << "x" << n << "x" << k << " index " << i);
            // TF32 keeps 10 mantissa bits against FP32's 23, so the tolerance
            // is far looser than the tiled kernel's. It is a relative bound
            // because the products grow with k.
            REQUIRE(actual[i] == Catch::Approx(expected[i]).epsilon(5e-3).margin(1e-3));
        }
    }
}

TEST_CASE("the tensor-core matmul refuses shapes it cannot tile", "[cuda][matmul][tensorcore]") {
    // Refusing is the contract. Padding silently would hide a performance
    // cliff; masking would be a different kernel.
    REQUIRE_FALSE(tensor_core_shape_supported(15, 16, 8));
    REQUIRE_FALSE(tensor_core_shape_supported(16, 17, 8));
    REQUIRE_FALSE(tensor_core_shape_supported(16, 16, 12));
    REQUIRE_FALSE(tensor_core_shape_supported(0, 16, 8));
    REQUIRE(tensor_core_shape_supported(16, 16, 8));
    REQUIRE(tensor_core_shape_supported(256, 512, 128));

    if (!tensor_cores_available()) {
        SKIP("device is older than sm_80");
    }
    const auto a = random_vector(15 * 8, 3);
    const auto b = random_vector(8 * 16, 4);
    REQUIRE(run_tensor_core(a, b, 15, 16, 8).empty());
}

TEST_CASE("tensor cores and the tiled kernel agree within tf32's precision",
          "[cuda][matmul][tensorcore]") {
    // The two paths are interchangeable for inference and are not bit-identical.
    // Pinning that relationship is the point: if they ever agree exactly, the
    // TF32 conversion has been dropped and the kernel is quietly running on the
    // CUDA cores.
    if (!tensor_cores_available()) {
        SKIP("device is older than sm_80");
    }
    constexpr int kM = 128;
    constexpr int kN = 128;
    constexpr int kK = 64;

    const auto a = random_vector(kM * kK, 11);
    const auto b = random_vector(kK * kN, 13);

    const auto from_tensor_cores = run_tensor_core(a, b, kM, kN, kK);
    REQUIRE_FALSE(from_tensor_cores.empty());

    CudaStream stream;
    DeviceBuffer<float> device_a(a.size());
    DeviceBuffer<float> device_b(b.size());
    DeviceBuffer<float> device_c(static_cast<std::size_t>(kM) * kN);
    device_a.copy_from_host(a.data(), a.size(), stream);
    device_b.copy_from_host(b.data(), b.size(), stream);
    launch_matmul(device_a.data(), device_b.data(), device_c.data(), kM, kN, kK, stream);

    std::vector<float> from_tiled(static_cast<std::size_t>(kM) * kN);
    device_c.copy_to_host(from_tiled.data(), from_tiled.size(), stream);
    stream.synchronize();

    bool any_difference = false;
    for (std::size_t i = 0; i < from_tiled.size(); ++i) {
        INFO("index " << i);
        REQUIRE(from_tensor_cores[i] == Catch::Approx(from_tiled[i]).epsilon(5e-3).margin(1e-3));
        any_difference = any_difference || std::fabs(from_tensor_cores[i] - from_tiled[i]) > 1e-7F;
    }
    REQUIRE(any_difference);
}

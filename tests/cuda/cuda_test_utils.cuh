#pragma once

#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "cudaforge/cuda_raii.cuh"

namespace cudaforge::test {

/// Deterministic input generation.
///
/// Every test seeds explicitly. An unseeded generator makes a marginal
/// tolerance failure appear intermittently, which is far harder to diagnose
/// than a consistent one — and floating-point kernels sit close enough to
/// their tolerances that this matters.
inline std::vector<float> random_vector(std::size_t count, unsigned seed, float scale = 1.0F) {
    std::mt19937 engine(seed);
    std::normal_distribution<float> distribution(0.0F, scale);

    std::vector<float> values(count);
    for (float& value : values) {
        value = distribution(engine);
    }
    return values;
}

/// Host-side reference sum in double precision.
///
/// The point of comparison is the *mathematical* sum, not another float32
/// accumulation. Summing in float32 on the host would accumulate its own error
/// and the test would be comparing two wrong answers.
inline double reference_sum(const std::vector<float>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0,
                           [](double acc, float value) { return acc + value; });
}

inline std::vector<float> reference_softmax(const std::vector<float>& input, int rows,
                                            int cols) {
    std::vector<float> output(input.size());
    for (int row = 0; row < rows; ++row) {
        const float* in = input.data() + static_cast<std::size_t>(row) * cols;
        float* out = output.data() + static_cast<std::size_t>(row) * cols;

        const float maximum = *std::max_element(in, in + cols);
        double total = 0.0;
        for (int col = 0; col < cols; ++col) {
            out[col] = std::exp(in[col] - maximum);
            total += out[col];
        }
        for (int col = 0; col < cols; ++col) {
            out[col] = static_cast<float>(out[col] / total);
        }
    }
    return output;
}

inline std::vector<float> reference_rmsnorm(const std::vector<float>& input,
                                            const std::vector<float>& weight, int rows,
                                            int cols, float eps) {
    std::vector<float> output(input.size());
    for (int row = 0; row < rows; ++row) {
        const float* in = input.data() + static_cast<std::size_t>(row) * cols;
        float* out = output.data() + static_cast<std::size_t>(row) * cols;

        double sum_squares = 0.0;
        for (int col = 0; col < cols; ++col) {
            sum_squares += static_cast<double>(in[col]) * in[col];
        }
        const auto scale =
            static_cast<float>(1.0 / std::sqrt(sum_squares / cols + eps));
        for (int col = 0; col < cols; ++col) {
            out[col] = in[col] * scale * weight[col];
        }
    }
    return output;
}

/// Relative comparison with an absolute floor.
///
/// A pure relative check divides by values near zero and reports enormous
/// errors for results that are numerically fine; a pure absolute check is too
/// loose for large values. The floor is what makes the near-zero case sane.
inline bool close(float actual, float expected, float relative, float absolute) {
    const float difference = std::fabs(actual - expected);
    return difference <= absolute + relative * std::fabs(expected);
}

inline void require_all_close(const std::vector<float>& actual,
                              const std::vector<float>& expected, float relative,
                              float absolute) {
    REQUIRE(actual.size() == expected.size());

    std::size_t worst_index = 0;
    float worst_difference = 0.0F;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float difference = std::fabs(actual[i] - expected[i]);
        if (difference > worst_difference) {
            worst_difference = difference;
            worst_index = i;
        }
    }

    // Reporting the worst element rather than the first failure makes a
    // tolerance miss immediately actionable.
    INFO("worst element " << worst_index << ": actual " << actual[worst_index]
                          << " expected " << expected[worst_index] << " (difference "
                          << worst_difference << ")");
    REQUIRE(close(actual[worst_index], expected[worst_index], relative, absolute));
}

/// Round-trip a host vector through the device with the given kernel launch.
template <typename Launch>
std::vector<float> run_on_device(const std::vector<float>& input, std::size_t output_count,
                                 Launch&& launch) {
    CudaStream stream;
    DeviceBuffer<float> device_in(input.size());
    DeviceBuffer<float> device_out(output_count);

    device_in.copy_from_host(input.data(), input.size(), stream);
    device_out.fill_zero(stream);

    launch(device_in.data(), device_out.data(), stream.get());

    std::vector<float> output(output_count);
    device_out.copy_to_host(output.data(), output_count, stream);
    stream.synchronize();
    return output;
}

}  // namespace cudaforge::test

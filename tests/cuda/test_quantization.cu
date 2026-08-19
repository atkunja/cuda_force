#include <catch2/catch_test_macros.hpp>

#include "cuda_test_utils.cuh"
#include "cudaforge/quantization.cuh"

using namespace cudaforge;
using namespace cudaforge::test;

namespace {

struct QuantResult {
    std::vector<std::int8_t> values;
    std::vector<float> scales;
};

QuantResult quantize(const std::vector<float>& input) {
    const int count = static_cast<int>(input.size());
    const int blocks = quant_scale_count(count);

    CudaStream stream;
    DeviceBuffer<float> device_in(input.size());
    DeviceBuffer<std::int8_t> device_q(input.size());
    DeviceBuffer<float> device_scales(static_cast<std::size_t>(blocks));

    device_in.copy_from_host(input.data(), input.size(), stream);
    launch_quantize_int8(device_in.data(), device_q.data(), device_scales.data(), count,
                         stream);

    QuantResult result;
    result.values.resize(input.size());
    result.scales.resize(static_cast<std::size_t>(blocks));
    device_q.copy_to_host(result.values.data(), result.values.size(), stream);
    device_scales.copy_to_host(result.scales.data(), result.scales.size(), stream);
    stream.synchronize();
    return result;
}

std::vector<float> dequantize(const QuantResult& quantised, int count) {
    CudaStream stream;
    DeviceBuffer<std::int8_t> device_q(quantised.values.size());
    DeviceBuffer<float> device_scales(quantised.scales.size());
    DeviceBuffer<float> device_out(static_cast<std::size_t>(count));

    device_q.copy_from_host(quantised.values.data(), quantised.values.size(), stream);
    device_scales.copy_from_host(quantised.scales.data(), quantised.scales.size(), stream);
    launch_dequantize_int8(device_q.data(), device_scales.data(), device_out.data(), count,
                           stream);

    std::vector<float> output(static_cast<std::size_t>(count));
    device_out.copy_to_host(output.data(), output.size(), stream);
    stream.synchronize();
    return output;
}

}  // namespace

TEST_CASE("the scale count matches the block size", "[cuda][quant]") {
    REQUIRE(quant_scale_count(0) == 0);
    REQUIRE(quant_scale_count(1) == 1);
    REQUIRE(quant_scale_count(kQuantBlockSize) == 1);
    REQUIRE(quant_scale_count(kQuantBlockSize + 1) == 2);
}

TEST_CASE("round-trip error stays within half a quantisation step", "[cuda][quant]") {
    // Symmetric absmax rounding cannot err by more than half a step, and the
    // step is the block's scale. Exceeding this is a bug, not merely loss.
    for (int count : {1, 63, 64, 65, 1000, 4096}) {
        const auto input = random_vector(static_cast<std::size_t>(count),
                                         static_cast<unsigned>(count), 5.0F);
        const auto quantised = quantize(input);
        const auto restored = dequantize(quantised, count);

        const float largest_scale =
            *std::max_element(quantised.scales.begin(), quantised.scales.end());

        for (int i = 0; i < count; ++i) {
            INFO("count " << count << " element " << i);
            REQUIRE(std::fabs(input[static_cast<std::size_t>(i)] -
                              restored[static_cast<std::size_t>(i)]) <=
                    largest_scale / 2.0F + 1e-5F);
        }
    }
}

TEST_CASE("quantised values stay in the symmetric int8 range", "[cuda][quant]") {
    // The range stops at 127 rather than 128 so negation is representable and
    // the grid is symmetric about zero.
    const auto input = random_vector(4096, /*seed=*/17, /*scale=*/100.0F);
    const auto quantised = quantize(input);

    for (std::int8_t value : quantised.values) {
        REQUIRE(value >= -127);
        REQUIRE(value <= 127);
    }
}

TEST_CASE("an all-zero block round-trips exactly", "[cuda][quant]") {
    // The scale would be zero; substituting 1 avoids the division and maps the
    // block to zero and back exactly.
    const std::vector<float> input(256, 0.0F);
    const auto quantised = quantize(input);
    const auto restored = dequantize(quantised, static_cast<int>(input.size()));

    for (float scale : quantised.scales) {
        REQUIRE(scale > 0.0F);
    }
    for (float value : restored) {
        REQUIRE(value == 0.0F);
    }
}

TEST_CASE("the block maximum survives the round trip", "[cuda][quant]") {
    // The absmax element defines its block's scale, so it maps to exactly 127
    // and back to itself.
    std::vector<float> input(kQuantBlockSize, 0.1F);
    input[5] = 4.0F;

    const auto quantised = quantize(input);
    const auto restored = dequantize(quantised, static_cast<int>(input.size()));

    REQUIRE(quantised.values[5] == 127);
    REQUIRE(close(restored[5], 4.0F, 1e-5F, 1e-6F));
}

TEST_CASE("an outlier degrades only its own block", "[cuda][quant]") {
    // Block-wise scaling exists so one large value does not stretch the range
    // of unrelated elements.
    std::vector<float> input(4 * kQuantBlockSize, 0.5F);
    input[0] = 1000.0F;

    const auto quantised = quantize(input);
    const auto restored = dequantize(quantised, static_cast<int>(input.size()));

    const float shared_block_error = std::fabs(input[1] - restored[1]);
    const float clean_block_error =
        std::fabs(input[3 * kQuantBlockSize] - restored[3 * kQuantBlockSize]);

    REQUIRE(shared_block_error > clean_block_error);
    REQUIRE(clean_block_error < 0.01F);
}

TEST_CASE("fake quantisation matches an explicit round trip", "[cuda][quant]") {
    const int count = 1024;
    const auto input = random_vector(static_cast<std::size_t>(count), /*seed=*/19, 3.0F);

    const auto quantised = quantize(input);
    const auto expected = dequantize(quantised, count);

    CudaStream stream;
    DeviceBuffer<float> device_in(input.size());
    DeviceBuffer<float> device_out(input.size());
    device_in.copy_from_host(input.data(), input.size(), stream);
    launch_quantize_dequantize(device_in.data(), device_out.data(), count, stream);

    std::vector<float> actual(input.size());
    device_out.copy_to_host(actual.data(), actual.size(), stream);
    stream.synchronize();

    require_all_close(actual, expected, /*relative=*/1e-6F, /*absolute=*/1e-6F);
}

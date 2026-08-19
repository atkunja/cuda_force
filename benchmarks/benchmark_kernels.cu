// CUDA kernel benchmarks.
//
// Requires an NVIDIA GPU. Nothing here has been executed on the development
// host, which has none; the numbers this produces are whatever the machine
// running it measures, and no results are committed to the repository.
//
// Method:
//
//   * Timing uses CUDA events, not a host clock. A kernel launch is
//     asynchronous, so a host timer around it measures the launch; adding a
//     synchronise to fix that measures the synchronisation too.
//   * Every kernel is warmed up before timing. The first launch pays for
//     context creation, module loading and JIT, none of which is the kernel.
//   * Each measurement is the median of N runs. The mean is dragged around by
//     occasional scheduling interference; the median is not.
//   * Effective bandwidth is reported for the memory-bound kernels, because
//     that is the number to compare against the device's peak. A kernel at 80%
//     of peak bandwidth is essentially done, regardless of how its time
//     compares to another implementation.

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "bench_common.hpp"
#include "cudaforge/cuda_raii.cuh"
#include "cudaforge/lora_linear.cuh"
#include "cudaforge/quantization.cuh"
#include "cudaforge/reduction.cuh"
#include "cudaforge/rmsnorm.cuh"
#include "cudaforge/softmax.cuh"

using namespace cudaforge;
using namespace cudaforge::bench;

namespace {

constexpr int kWarmupRuns = 20;
constexpr int kTimedRuns = 100;

struct Timing {
    double median_ms;
    double min_ms;
    double p95_ms;
};

/// Times a launch with CUDA events, discarding warmup runs.
template <typename Launch>
Timing time_kernel(cudaStream_t stream, Launch&& launch) {
    CudaEvent start(CudaEvent::Purpose::Timing);
    CudaEvent stop(CudaEvent::Purpose::Timing);

    for (int i = 0; i < kWarmupRuns; ++i) {
        launch();
    }
    CUDAFORGE_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> samples;
    samples.reserve(kTimedRuns);
    for (int i = 0; i < kTimedRuns; ++i) {
        start.record(stream);
        launch();
        stop.record(stream);
        stop.synchronize();
        samples.push_back(static_cast<double>(CudaEvent::elapsed_ms(start, stop)));
    }

    std::vector<double> sorted = samples;
    return Timing{
        percentile(sorted, 0.50),
        *std::min_element(samples.begin(), samples.end()),
        percentile(sorted, 0.95),
    };
}

/// GB/s implied by moving `bytes` in `milliseconds`.
///
/// Meaningful only for kernels whose cost is dominated by memory traffic, which
/// is all of these except the matmul. Compare it against the device's
/// theoretical peak: the ratio says how much headroom is left, whereas a raw
/// millisecond figure says nothing without a reference.
double effective_bandwidth(std::size_t bytes, double milliseconds) {
    if (milliseconds <= 0.0) {
        return 0.0;
    }
    return static_cast<double>(bytes) / (milliseconds * 1e-3) / 1e9;
}

void emit(JsonWriter& writer, const std::string& kernel, const std::string& variant,
          const std::string& shape, const Timing& timing, double bandwidth) {
    writer.array_element_begin();
    writer.field("kernel", kernel);
    writer.field("variant", variant);
    writer.field("shape", shape);
    writer.field("median_ms", timing.median_ms);
    writer.field("min_ms", timing.min_ms);
    writer.field("p95_ms", timing.p95_ms);
    if (bandwidth > 0.0) {
        writer.field("effective_bandwidth_gb_s", bandwidth);
    }
    writer.array_element_end();
}

void benchmark_reduction(JsonWriter& writer, cudaStream_t stream) {
    for (std::size_t count : {1UL << 16, 1UL << 20, 1UL << 24}) {
        DeviceBuffer<float> input(count);
        DeviceBuffer<float> output(1);
        input.fill_zero(stream);

        const std::pair<ReductionKernel, const char*> variants[] = {
            {ReductionKernel::Naive, "naive"},
            {ReductionKernel::SharedMemory, "shared_memory"},
            {ReductionKernel::WarpOptimised, "warp_shuffle"},
        };

        for (const auto& [variant, name] : variants) {
            const Timing timing = time_kernel(stream, [&] {
                output.fill_zero(stream);
                launch_reduce_sum(input.data(), output.data(), count, variant, stream);
            });
            // A reduction reads the input once and writes almost nothing.
            emit(writer, "reduce_sum", name, std::to_string(count), timing,
                 effective_bandwidth(count * sizeof(float), timing.median_ms));
        }
    }
}

void benchmark_softmax(JsonWriter& writer, cudaStream_t stream) {
    const std::pair<int, int> shapes[] = {{1024, 512}, {512, 2048}, {128, 8192}};

    for (const auto& [rows, cols] : shapes) {
        const auto elements = static_cast<std::size_t>(rows) * cols;
        DeviceBuffer<float> input(elements);
        DeviceBuffer<float> output(elements);
        input.fill_zero(stream);

        const std::pair<SoftmaxKernel, const char*> variants[] = {
            {SoftmaxKernel::Naive, "naive"},
            {SoftmaxKernel::SharedMemory, "shared_memory"},
            {SoftmaxKernel::Online, "online"},
        };

        for (const auto& [variant, name] : variants) {
            const Timing timing = time_kernel(stream, [&] {
                launch_softmax(input.data(), output.data(), rows, cols, variant, stream);
            });
            // One read and one write in the best case; the naive variant reads
            // three times, which is exactly what the comparison should expose.
            emit(writer, "softmax", name,
                 std::to_string(rows) + "x" + std::to_string(cols), timing,
                 effective_bandwidth(2 * elements * sizeof(float), timing.median_ms));
        }
    }
}

void benchmark_rmsnorm(JsonWriter& writer, cudaStream_t stream) {
    const std::pair<int, int> shapes[] = {{4096, 1024}, {2048, 4096}, {512, 8192}};

    for (const auto& [rows, cols] : shapes) {
        const auto elements = static_cast<std::size_t>(rows) * cols;
        DeviceBuffer<float> input(elements);
        DeviceBuffer<float> weight(static_cast<std::size_t>(cols));
        DeviceBuffer<float> output(elements);
        input.fill_zero(stream);
        weight.fill_zero(stream);

        const std::pair<RMSNormKernel, const char*> variants[] = {
            {RMSNormKernel::Naive, "scalar"},
            {RMSNormKernel::Vectorised, "float4"},
        };

        for (const auto& [variant, name] : variants) {
            const Timing timing = time_kernel(stream, [&] {
                launch_rmsnorm(input.data(), weight.data(), output.data(), rows, cols, 1e-6F,
                               variant, stream);
            });
            // Input is read twice (once for the reduction, once to normalise)
            // and output written once.
            emit(writer, "rmsnorm", name,
                 std::to_string(rows) + "x" + std::to_string(cols), timing,
                 effective_bandwidth(3 * elements * sizeof(float), timing.median_ms));
        }
    }
}

void benchmark_lora(JsonWriter& writer, cudaStream_t stream) {
    struct Shape {
        int batch;
        int in_features;
        int out_features;
        int rank;
    };

    const Shape shapes[] = {
        {32, 1024, 1024, 8},
        {64, 2048, 2048, 16},
        {128, 4096, 4096, 16},
    };

    for (const Shape& shape : shapes) {
        DeviceBuffer<float> x(static_cast<std::size_t>(shape.batch) * shape.in_features);
        DeviceBuffer<float> w(
            static_cast<std::size_t>(shape.in_features) * shape.out_features);
        DeviceBuffer<float> a(static_cast<std::size_t>(shape.in_features) * shape.rank);
        DeviceBuffer<float> b(static_cast<std::size_t>(shape.rank) * shape.out_features);
        DeviceBuffer<float> y(static_cast<std::size_t>(shape.batch) * shape.out_features);
        DeviceBuffer<float> workspace(
            static_cast<std::size_t>(shape.batch) * shape.rank);

        for (DeviceBuffer<float>* buffer : {&x, &w, &a, &b, &y, &workspace}) {
            buffer->fill_zero(stream);
        }

        const std::string label = std::to_string(shape.batch) + "x" +
                                  std::to_string(shape.in_features) + "x" +
                                  std::to_string(shape.out_features) + "r" +
                                  std::to_string(shape.rank);

        for (const auto& [variant, name] : {std::pair{LoRAKernel::Unfused, "unfused"},
                                            std::pair{LoRAKernel::Fused, "fused"}}) {
            const Timing timing = time_kernel(stream, [&] {
                launch_lora_linear(x.data(), w.data(), a.data(), b.data(), y.data(),
                                   workspace.data(), shape.batch, shape.in_features,
                                   shape.out_features, shape.rank, 2.0F, variant, stream);
            });
            // No bandwidth figure: this one is not purely memory-bound, so the
            // number would invite a comparison against peak that does not mean
            // what it appears to.
            emit(writer, "lora_linear", name, label, timing, 0.0);
        }
    }
}

void benchmark_quantization(JsonWriter& writer, cudaStream_t stream) {
    for (int count : {1 << 20, 1 << 24}) {
        DeviceBuffer<float> input(static_cast<std::size_t>(count));
        DeviceBuffer<std::int8_t> quantised(static_cast<std::size_t>(count));
        DeviceBuffer<float> scales(static_cast<std::size_t>(quant_scale_count(count)));
        DeviceBuffer<float> restored(static_cast<std::size_t>(count));
        input.fill_zero(stream);

        const Timing quantise = time_kernel(stream, [&] {
            launch_quantize_int8(input.data(), quantised.data(), scales.data(), count, stream);
        });
        emit(writer, "quantize_int8", "blockwise", std::to_string(count), quantise,
             effective_bandwidth(static_cast<std::size_t>(count) * (sizeof(float) + 1),
                                 quantise.median_ms));

        const Timing dequantise = time_kernel(stream, [&] {
            launch_dequantize_int8(quantised.data(), scales.data(), restored.data(), count,
                                   stream);
        });
        emit(writer, "dequantize_int8", "blockwise", std::to_string(count), dequantise,
             effective_bandwidth(static_cast<std::size_t>(count) * (sizeof(float) + 1),
                                 dequantise.median_ms));
    }
}

void write_device_info(JsonWriter& writer) {
    int device = 0;
    CUDAFORGE_CHECK(cudaGetDevice(&device));

    cudaDeviceProp properties{};
    CUDAFORGE_CHECK(cudaGetDeviceProperties(&properties, device));

    writer.field("device", std::string(properties.name));
    writer.field("compute_capability",
                 std::to_string(properties.major) + "." + std::to_string(properties.minor));
    writer.field("multiprocessors", static_cast<std::uint64_t>(properties.multiProcessorCount));
    // The figure every bandwidth number below should be read against.
    writer.field("theoretical_bandwidth_gb_s",
                 2.0 * properties.memoryClockRate * (properties.memoryBusWidth / 8) / 1.0e6);
}

}  // namespace

int main() {
    try {
        CudaStream stream;

        JsonWriter writer(std::cout);
        writer.begin_object();
        writer.field("benchmark", std::string("cuda_kernels"));
        write_device_info(writer);
        writer.field("warmup_runs", static_cast<std::uint64_t>(kWarmupRuns));
        writer.field("timed_runs", static_cast<std::uint64_t>(kTimedRuns));
        writer.begin_array("results");

        benchmark_reduction(writer, stream);
        benchmark_softmax(writer, stream);
        benchmark_rmsnorm(writer, stream);
        benchmark_lora(writer, stream);
        benchmark_quantization(writer, stream);

        writer.end_array();
        writer.end_object();
        writer.finish();
        return 0;
    } catch (const CudaError& error) {
        std::cerr << "CUDA failure: " << error.what() << "\n";
        return 1;
    }
}

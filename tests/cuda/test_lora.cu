#include <catch2/catch_test_macros.hpp>

#include "cuda_test_utils.cuh"
#include "cudaforge/lora_linear.cuh"

using namespace cudaforge;
using namespace cudaforge::test;

namespace {

std::vector<float> reference_lora(const std::vector<float>& x, const std::vector<float>& w,
                                  const std::vector<float>& a, const std::vector<float>& b,
                                  int batch, int in_features, int out_features, int rank,
                                  float scale) {
    // Accumulated in double so the reference is the mathematical answer, not
    // another float32 sum with its own error.
    std::vector<float> output(static_cast<std::size_t>(batch) * out_features);

    std::vector<double> xa(static_cast<std::size_t>(batch) * rank, 0.0);
    for (int row = 0; row < batch; ++row) {
        for (int r = 0; r < rank; ++r) {
            double total = 0.0;
            for (int i = 0; i < in_features; ++i) {
                total += static_cast<double>(x[static_cast<std::size_t>(row) * in_features + i]) *
                         a[static_cast<std::size_t>(i) * rank + r];
            }
            xa[static_cast<std::size_t>(row) * rank + r] = total;
        }
    }

    for (int row = 0; row < batch; ++row) {
        for (int col = 0; col < out_features; ++col) {
            double frozen = 0.0;
            for (int i = 0; i < in_features; ++i) {
                frozen += static_cast<double>(x[static_cast<std::size_t>(row) * in_features + i]) *
                          w[static_cast<std::size_t>(i) * out_features + col];
            }
            double adapter = 0.0;
            for (int r = 0; r < rank; ++r) {
                adapter += xa[static_cast<std::size_t>(row) * rank + r] *
                           b[static_cast<std::size_t>(r) * out_features + col];
            }
            output[static_cast<std::size_t>(row) * out_features + col] =
                static_cast<float>(frozen + scale * adapter);
        }
    }
    return output;
}

std::vector<float> run_lora(const std::vector<float>& x, const std::vector<float>& w,
                            const std::vector<float>& a, const std::vector<float>& b,
                            int batch, int in_features, int out_features, int rank,
                            float scale, LoRAKernel variant) {
    CudaStream stream;
    DeviceBuffer<float> dx(x.size());
    DeviceBuffer<float> dw(w.size());
    DeviceBuffer<float> da(a.size());
    DeviceBuffer<float> db(b.size());
    DeviceBuffer<float> dy(static_cast<std::size_t>(batch) * out_features);
    DeviceBuffer<float> workspace(static_cast<std::size_t>(batch) * rank);

    dx.copy_from_host(x.data(), x.size(), stream);
    dw.copy_from_host(w.data(), w.size(), stream);
    da.copy_from_host(a.data(), a.size(), stream);
    db.copy_from_host(b.data(), b.size(), stream);
    dy.fill_zero(stream);

    launch_lora_linear(dx.data(), dw.data(), da.data(), db.data(), dy.data(), workspace.data(),
                       batch, in_features, out_features, rank, scale, variant, stream);

    std::vector<float> output(dy.size());
    dy.copy_to_host(output.data(), output.size(), stream);
    stream.synchronize();
    return output;
}

}  // namespace

TEST_CASE("both lora variants match the reference", "[cuda][lora]") {
    struct Shape {
        int batch;
        int in_features;
        int out_features;
        int rank;
    };

    // Includes non-multiples of the 16-wide tile so the tiled matmul's bounds
    // checks are exercised rather than assumed.
    for (const Shape shape : {Shape{1, 8, 4, 2}, Shape{4, 64, 32, 8}, Shape{7, 129, 65, 3},
                              Shape{16, 256, 256, 16}, Shape{33, 100, 50, 4}}) {
        const auto x = random_vector(
            static_cast<std::size_t>(shape.batch) * shape.in_features, 1);
        const auto w = random_vector(
            static_cast<std::size_t>(shape.in_features) * shape.out_features, 2);
        const auto a = random_vector(
            static_cast<std::size_t>(shape.in_features) * shape.rank, 3);
        const auto b = random_vector(
            static_cast<std::size_t>(shape.rank) * shape.out_features, 4);
        constexpr float kScale = 0.25F;

        const auto expected = reference_lora(x, w, a, b, shape.batch, shape.in_features,
                                             shape.out_features, shape.rank, kScale);

        for (LoRAKernel variant : {LoRAKernel::Unfused, LoRAKernel::Fused}) {
            INFO("batch " << shape.batch << " in " << shape.in_features << " out "
                          << shape.out_features << " rank " << shape.rank << " variant "
                          << static_cast<int>(variant));
            require_all_close(run_lora(x, w, a, b, shape.batch, shape.in_features,
                                       shape.out_features, shape.rank, kScale, variant),
                              expected, /*relative=*/1e-3F, /*absolute=*/1e-3F);
        }
    }
}

TEST_CASE("a zero b matrix reduces lora to the frozen layer", "[cuda][lora]") {
    // This is how LoRA is initialised: with B at zero the adapted model is
    // numerically identical to the base model at step 0.
    constexpr int kBatch = 8;
    constexpr int kIn = 64;
    constexpr int kOut = 32;
    constexpr int kRank = 8;

    const auto x = random_vector(kBatch * kIn, 11);
    const auto w = random_vector(kIn * kOut, 12);
    const auto a = random_vector(kIn * kRank, 13);
    const std::vector<float> b(kRank * kOut, 0.0F);

    const auto expected = reference_lora(x, w, a, b, kBatch, kIn, kOut, kRank, 1.0F);
    for (LoRAKernel variant : {LoRAKernel::Unfused, LoRAKernel::Fused}) {
        INFO("variant " << static_cast<int>(variant));
        require_all_close(run_lora(x, w, a, b, kBatch, kIn, kOut, kRank, 1.0F, variant),
                          expected, 1e-3F, 1e-3F);
    }
}

TEST_CASE("the fused and unfused paths agree", "[cuda][lora]") {
    constexpr int kBatch = 32;
    constexpr int kIn = 128;
    constexpr int kOut = 128;
    constexpr int kRank = 16;

    const auto x = random_vector(kBatch * kIn, 21);
    const auto w = random_vector(kIn * kOut, 22);
    const auto a = random_vector(kIn * kRank, 23);
    const auto b = random_vector(kRank * kOut, 24);

    require_all_close(
        run_lora(x, w, a, b, kBatch, kIn, kOut, kRank, 0.5F, LoRAKernel::Fused),
        run_lora(x, w, a, b, kBatch, kIn, kOut, kRank, 0.5F, LoRAKernel::Unfused),
        /*relative=*/1e-4F, /*absolute=*/1e-4F);
}

TEST_CASE("the tiled matmul matches a host reference", "[cuda][lora][matmul]") {
    for (auto [m, n, k] : {std::tuple{1, 1, 1}, std::tuple{16, 16, 16},
                           std::tuple{17, 33, 65}, std::tuple{128, 64, 256}}) {
        const auto a = random_vector(static_cast<std::size_t>(m) * k, 41);
        const auto b = random_vector(static_cast<std::size_t>(k) * n, 42);

        CudaStream stream;
        DeviceBuffer<float> da(a.size());
        DeviceBuffer<float> db(b.size());
        DeviceBuffer<float> dc(static_cast<std::size_t>(m) * n);
        da.copy_from_host(a.data(), a.size(), stream);
        db.copy_from_host(b.data(), b.size(), stream);
        launch_matmul(da.data(), db.data(), dc.data(), m, n, k, stream);

        std::vector<float> actual(dc.size());
        dc.copy_to_host(actual.data(), actual.size(), stream);
        stream.synchronize();

        std::vector<float> expected(actual.size());
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
                double total = 0.0;
                for (int i = 0; i < k; ++i) {
                    total += static_cast<double>(a[static_cast<std::size_t>(row) * k + i]) *
                             b[static_cast<std::size_t>(i) * n + col];
                }
                expected[static_cast<std::size_t>(row) * n + col] = static_cast<float>(total);
            }
        }

        INFO("m " << m << " n " << n << " k " << k);
        require_all_close(actual, expected, 1e-3F, 1e-3F);
    }
}

TEST_CASE("the workspace size covers the adapter intermediate", "[cuda][lora]") {
    REQUIRE(lora_workspace_bytes(8, 16) == 8 * 16 * sizeof(float));
    REQUIRE(lora_workspace_bytes(0, 16) == 0);
}

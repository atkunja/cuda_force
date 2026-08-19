#include "cudaforge/rmsnorm.cuh"

#include <cstdint>

#include "cudaforge/cuda_error.cuh"
#include "cudaforge/cuda_utils.cuh"
#include "cudaforge/reduced_precision.cuh"

namespace cudaforge {
namespace {

__global__ void rmsnorm_naive(const float* __restrict__ input, const float* __restrict__ weight,
                              float* __restrict__ output, int rows, int cols, float eps) {
    __shared__ float scratch[kWarpSize];

    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) {
        return;
    }

    const float* row_in = input + static_cast<std::size_t>(row) * cols;
    float* row_out = output + static_cast<std::size_t>(row) * cols;
    const int tid = static_cast<int>(threadIdx.x);
    const int step = static_cast<int>(blockDim.x);

    float sum_squares = 0.0F;
    for (int col = tid; col < cols; col += step) {
        const float value = row_in[col];
        sum_squares += value * value;
    }
    sum_squares = block_reduce_sum(sum_squares, scratch);

    // rsqrtf maps to a single hardware instruction. Its relative error is well
    // under the FP32 tolerance the tests use, and it removes both a division
    // and a square root from the inner loop.
    const float scale = rsqrtf(sum_squares / static_cast<float>(cols) + eps);

    for (int col = tid; col < cols; col += step) {
        row_out[col] = row_in[col] * scale * weight[col];
    }
}

/// `float4` loads and stores. The row is treated as `cols / 4` vectors, so each
/// thread issues one 128-bit transaction where the scalar kernel issues four
/// 32-bit ones. The reduction and the scale are unchanged.
__global__ void rmsnorm_vectorised(const float4* __restrict__ input,
                                   const float4* __restrict__ weight, float4* __restrict__ output,
                                   int rows, int vec_cols, int cols, float eps) {
    __shared__ float scratch[kWarpSize];

    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) {
        return;
    }

    const float4* row_in = input + static_cast<std::size_t>(row) * vec_cols;
    float4* row_out = output + static_cast<std::size_t>(row) * vec_cols;
    const int tid = static_cast<int>(threadIdx.x);
    const int step = static_cast<int>(blockDim.x);

    float sum_squares = 0.0F;
    for (int index = tid; index < vec_cols; index += step) {
        const float4 value = row_in[index];
        sum_squares += value.x * value.x;
        sum_squares += value.y * value.y;
        sum_squares += value.z * value.z;
        sum_squares += value.w * value.w;
    }
    sum_squares = block_reduce_sum(sum_squares, scratch);

    const float scale = rsqrtf(sum_squares / static_cast<float>(cols) + eps);

    for (int index = tid; index < vec_cols; index += step) {
        const float4 value = row_in[index];
        const float4 gain = weight[index];
        float4 result;
        result.x = value.x * scale * gain.x;
        result.y = value.y * scale * gain.y;
        result.z = value.z * scale * gain.z;
        result.w = value.w * scale * gain.w;
        row_out[index] = result;
    }
}

/// Templated over the storage type so FP16 and BF16 share one implementation.
/// Only the conversions differ; the accumulator is FP32 either way — see
/// reduced_precision.cuh for why that is not negotiable for either format.
template <typename T>
__global__ void rmsnorm_reduced(const T* __restrict__ input, const T* __restrict__ weight,
                                T* __restrict__ output, int rows, int cols, float eps) {
    using Convert = ReducedPrecision<T>;
    __shared__ float scratch[kWarpSize];

    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) {
        return;
    }

    const T* row_in = input + static_cast<std::size_t>(row) * cols;
    T* row_out = output + static_cast<std::size_t>(row) * cols;
    const int tid = static_cast<int>(threadIdx.x);
    const int step = static_cast<int>(blockDim.x);

    float sum_squares = 0.0F;
    for (int col = tid; col < cols; col += step) {
        const float value = Convert::to_float(row_in[col]);
        sum_squares += value * value;
    }
    sum_squares = block_reduce_sum(sum_squares, scratch);

    const float scale = rsqrtf(sum_squares / static_cast<float>(cols) + eps);

    for (int col = tid; col < cols; col += step) {
        const float value =
            Convert::to_float(row_in[col]) * scale * Convert::to_float(weight[col]);
        row_out[col] = Convert::from_float(value);
    }
}

/// A `float4` load faults unless the address is 16-byte aligned. Row starts
/// must also be aligned, which requires `cols` to be a multiple of four —
/// otherwise row 1 begins mid-vector even when row 0 is aligned.
bool vectorisable(const void* input, const void* weight, const void* output, int cols) {
    constexpr std::uintptr_t kAlignment = alignof(float4);
    const auto aligned = [](const void* pointer) {
        return reinterpret_cast<std::uintptr_t>(pointer) % kAlignment == 0;
    };
    return cols % 4 == 0 && aligned(input) && aligned(weight) && aligned(output);
}

}  // namespace

void launch_rmsnorm(const float* input, const float* weight, float* output, int rows, int cols,
                    float eps, RMSNormKernel variant, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) {
        return;
    }

    if (variant == RMSNormKernel::Vectorised && vectorisable(input, weight, output, cols)) {
        const int vec_cols = cols / 4;
        const int block = block_size_for_row(vec_cols);
        rmsnorm_vectorised<<<rows, block, 0, stream>>>(
            reinterpret_cast<const float4*>(input), reinterpret_cast<const float4*>(weight),
            reinterpret_cast<float4*>(output), rows, vec_cols, cols, eps);
    } else {
        const int block = block_size_for_row(cols);
        rmsnorm_naive<<<rows, block, 0, stream>>>(input, weight, output, rows, cols, eps);
    }

    CUDAFORGE_CHECK_LAUNCH(stream);
}

void launch_rmsnorm_half(const __half* input, const __half* weight, __half* output, int rows,
                         int cols, float eps, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) {
        return;
    }
    const int block = block_size_for_row(cols);
    rmsnorm_reduced<<<rows, block, 0, stream>>>(input, weight, output, rows, cols, eps);
    CUDAFORGE_CHECK_LAUNCH(stream);
}

void launch_rmsnorm_bf16(const __nv_bfloat16* input, const __nv_bfloat16* weight,
                         __nv_bfloat16* output, int rows, int cols, float eps,
                         cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) {
        return;
    }
    const int block = block_size_for_row(cols);
    rmsnorm_reduced<<<rows, block, 0, stream>>>(input, weight, output, rows, cols, eps);
    CUDAFORGE_CHECK_LAUNCH(stream);
}

}  // namespace cudaforge

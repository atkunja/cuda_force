#include "cudaforge/quantization.cuh"

#include <cfloat>

#include "cudaforge/cuda_error.cuh"
#include "cudaforge/cuda_utils.cuh"

namespace cudaforge {
namespace {

/// One block of threads per quantisation block.
///
/// Launching exactly `kQuantBlockSize` threads means the absmax reduction is a
/// pure warp-shuffle reduction over two warps, with a single shared-memory
/// exchange between them. Every thread then quantises exactly one element using
/// the scale it helped compute, so the scale never leaves the SM.
__global__ void quantize_int8(const float* __restrict__ input, std::int8_t* __restrict__ output,
                              float* __restrict__ scales, int count) {
    __shared__ float shared[kWarpSize];

    const int block_index = static_cast<int>(blockIdx.x);
    const int base = block_index * kQuantBlockSize;
    const int index = base + static_cast<int>(threadIdx.x);

    const float value = index < count ? input[index] : 0.0F;
    const float absmax = block_reduce_max(fabsf(value), shared, 0.0F);

    // An all-zero block has absmax 0. Quantising with a zero scale would divide
    // by zero; a scale of 1 maps every element to 0 and dequantises back to 0,
    // which is exactly right.
    const float scale = absmax > 0.0F ? absmax / 127.0F : 1.0F;

    if (threadIdx.x == 0) {
        scales[block_index] = scale;
    }

    if (index < count) {
        // rintf rounds half to even, matching the reference implementation.
        // Truncation would bias every value toward zero and shift the mean of
        // the dequantised tensor.
        const float quantised = fminf(fmaxf(rintf(value / scale), -127.0F), 127.0F);
        output[index] = static_cast<std::int8_t>(quantised);
    }
}

__global__ void dequantize_int8(const std::int8_t* __restrict__ input,
                                const float* __restrict__ scales, float* __restrict__ output,
                                int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) {
        return;
    }
    output[index] = static_cast<float>(input[index]) * scales[index / kQuantBlockSize];
}

/// Round trip without materialising the INT8 tensor. Used to measure the
/// accuracy cost of the scheme; the quantised values stay in registers.
__global__ void quantize_dequantize(const float* __restrict__ input, float* __restrict__ output,
                                    int count) {
    __shared__ float shared[kWarpSize];

    const int base = static_cast<int>(blockIdx.x) * kQuantBlockSize;
    const int index = base + static_cast<int>(threadIdx.x);

    const float value = index < count ? input[index] : 0.0F;
    const float absmax = block_reduce_max(fabsf(value), shared, 0.0F);
    const float scale = absmax > 0.0F ? absmax / 127.0F : 1.0F;

    if (index < count) {
        const float quantised = fminf(fmaxf(rintf(value / scale), -127.0F), 127.0F);
        output[index] = quantised * scale;
    }
}

}  // namespace

int quant_scale_count(int count) {
    return count <= 0 ? 0 : ceil_div(count, kQuantBlockSize);
}

void launch_quantize_int8(const float* input, std::int8_t* output, float* scales, int count,
                          cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    const int blocks = quant_scale_count(count);
    quantize_int8<<<blocks, kQuantBlockSize, 0, stream>>>(input, output, scales, count);
    CUDAFORGE_CHECK_LAUNCH(stream);
}

void launch_dequantize_int8(const std::int8_t* input, const float* scales, float* output, int count,
                            cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    const int blocks = ceil_div(count, kDefaultBlockSize);
    dequantize_int8<<<blocks, kDefaultBlockSize, 0, stream>>>(input, scales, output, count);
    CUDAFORGE_CHECK_LAUNCH(stream);
}

void launch_quantize_dequantize(const float* input, float* output, int count, cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    const int blocks = quant_scale_count(count);
    quantize_dequantize<<<blocks, kQuantBlockSize, 0, stream>>>(input, output, count);
    CUDAFORGE_CHECK_LAUNCH(stream);
}

}  // namespace cudaforge

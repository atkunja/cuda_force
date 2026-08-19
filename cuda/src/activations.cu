#include "cudaforge/activations.cuh"

#include <cstddef>
#include <cstdint>

#include "cudaforge/cuda_error.cuh"
#include "cudaforge/cuda_utils.cuh"

namespace cudaforge {
namespace {

/// `x * sigmoid(x)`, evaluated as `x / (1 + exp(-x))`.
///
/// `__expf` is the fast-math intrinsic: a few ULP less accurate than `expf` and
/// a single instruction rather than a polynomial. On a bandwidth-bound kernel
/// the accuracy difference is far below the tolerance these tests use and the
/// instruction count is what matters.
__device__ __forceinline__ float silu(float x) {
    return x / (1.0F + __expf(-x));
}

/// GELU, tanh approximation:
///
///     0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
///
/// The constants are from the original formulation and are not tunable — they
/// are what the models using this were trained against.
__device__ __forceinline__ float gelu_tanh(float x) {
    constexpr float kSqrt2OverPi = 0.7978845608028654F;
    constexpr float kCoefficient = 0.044715F;
    const float inner = kSqrt2OverPi * (x + kCoefficient * x * x * x);
    return 0.5F * x * (1.0F + tanhf(inner));
}

__global__ void silu_kernel(const float* __restrict__ input, float* __restrict__ output,
                            int count) {
    // Computed in size_t: the product overflows a 32-bit signed index at counts
    // near INT_MAX, and the result would be a wrong element rather than a fault.
    const std::size_t index =
        blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (index < static_cast<std::size_t>(count)) {
        output[index] = silu(input[index]);
    }
}

__global__ void gelu_kernel(const float* __restrict__ input, float* __restrict__ output,
                            int count) {
    // Computed in size_t: the product overflows a 32-bit signed index at counts
    // near INT_MAX, and the result would be a wrong element rather than a fault.
    const std::size_t index =
        blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (index < static_cast<std::size_t>(count)) {
        output[index] = gelu_tanh(input[index]);
    }
}

/// Fused `silu(gate) * up`. Two reads and one write — the minimum possible
/// traffic for this operation.
__global__ void add_kernel(const float* __restrict__ a, const float* __restrict__ b,
                           float* __restrict__ output, int count) {
    // Computed in size_t: the product overflows a 32-bit signed index at counts
    // near INT_MAX, and the result would be a wrong element rather than a fault.
    const std::size_t index =
        blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (index < static_cast<std::size_t>(count)) {
        output[index] = a[index] + b[index];
    }
}

__global__ void swiglu_scalar(const float* __restrict__ gate, const float* __restrict__ up,
                              float* __restrict__ output, int count) {
    // Computed in size_t: the product overflows a 32-bit signed index at counts
    // near INT_MAX, and the result would be a wrong element rather than a fault.
    const std::size_t index =
        blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (index < static_cast<std::size_t>(count)) {
        output[index] = silu(gate[index]) * up[index];
    }
}

/// Same arithmetic over `float4`, so each thread issues one 128-bit transaction
/// per operand instead of four 32-bit ones.
__global__ void swiglu_vectorised(const float4* __restrict__ gate, const float4* __restrict__ up,
                                  float4* __restrict__ output, int vec_count) {
    const std::size_t index =
        blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (index >= static_cast<std::size_t>(vec_count)) {
        return;
    }

    const float4 g = gate[index];
    const float4 u = up[index];

    float4 result;
    result.x = silu(g.x) * u.x;
    result.y = silu(g.y) * u.y;
    result.z = silu(g.z) * u.z;
    result.w = silu(g.w) * u.w;
    output[index] = result;
}

/// FP16 storage, FP32 sigmoid. See the note in activations.cuh: `exp` of a
/// moderately negative input underflows a 10-bit mantissa long before it
/// underflows FP32, which would flatten the activation's negative tail.
__global__ void swiglu_half_kernel(const __half* __restrict__ gate, const __half* __restrict__ up,
                                   __half* __restrict__ output, int count) {
    // Computed in size_t: the product overflows a 32-bit signed index at counts
    // near INT_MAX, and the result would be a wrong element rather than a fault.
    const std::size_t index =
        blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (index < static_cast<std::size_t>(count)) {
        const float activated = silu(__half2float(gate[index]));
        output[index] = __float2half(activated * __half2float(up[index]));
    }
}

/// A `float4` access faults unless the address is 16-byte aligned, and the
/// element count must be divisible by four or the tail is left unwritten.
bool vectorisable(const void* gate, const void* up, const void* output, int count) {
    constexpr std::uintptr_t kAlignment = alignof(float4);
    const auto aligned = [](const void* pointer) {
        return reinterpret_cast<std::uintptr_t>(pointer) % kAlignment == 0;
    };
    return count % 4 == 0 && aligned(gate) && aligned(up) && aligned(output);
}

}  // namespace

void launch_silu(const float* input, float* output, int count, cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    const int grid = ceil_div(count, kDefaultBlockSize);
    silu_kernel<<<grid, kDefaultBlockSize, 0, stream>>>(input, output, count);
    CUDAFORGE_CHECK_LAUNCH(stream);
}

void launch_gelu(const float* input, float* output, int count, cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    const int grid = ceil_div(count, kDefaultBlockSize);
    gelu_kernel<<<grid, kDefaultBlockSize, 0, stream>>>(input, output, count);
    CUDAFORGE_CHECK_LAUNCH(stream);
}

void launch_add(const float* a, const float* b, float* output, int count, cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    const int grid = ceil_div(count, kDefaultBlockSize);
    add_kernel<<<grid, kDefaultBlockSize, 0, stream>>>(a, b, output, count);
    CUDAFORGE_CHECK_LAUNCH(stream);
}

void launch_swiglu(const float* gate, const float* up, float* output, int count,
                   SwiGLUKernel variant, cudaStream_t stream) {
    if (count <= 0) {
        return;
    }

    if (variant == SwiGLUKernel::Vectorised && vectorisable(gate, up, output, count)) {
        const int vec_count = count / 4;
        const int grid = ceil_div(vec_count, kDefaultBlockSize);
        swiglu_vectorised<<<grid, kDefaultBlockSize, 0, stream>>>(
            reinterpret_cast<const float4*>(gate), reinterpret_cast<const float4*>(up),
            reinterpret_cast<float4*>(output), vec_count);
    } else {
        const int grid = ceil_div(count, kDefaultBlockSize);
        swiglu_scalar<<<grid, kDefaultBlockSize, 0, stream>>>(gate, up, output, count);
    }

    CUDAFORGE_CHECK_LAUNCH(stream);
}

void launch_swiglu_half(const __half* gate, const __half* up, __half* output, int count,
                        cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    const int grid = ceil_div(count, kDefaultBlockSize);
    swiglu_half_kernel<<<grid, kDefaultBlockSize, 0, stream>>>(gate, up, output, count);
    CUDAFORGE_CHECK_LAUNCH(stream);
}

}  // namespace cudaforge

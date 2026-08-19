#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace cudaforge {

enum class SwiGLUKernel {
    /// One thread per element, scalar loads and stores.
    Scalar,

    /// `float4` loads and stores. The kernel is purely bandwidth-bound, so
    /// quartering the number of memory instructions is close to the only lever
    /// available. Requires 16-byte alignment and a length divisible by four;
    /// the launcher checks and falls back.
    Vectorised,
};

/// SiLU, also called swish: `x * sigmoid(x)`.
///
/// Smooth and non-monotonic, unlike ReLU, and its gradient does not vanish for
/// negative inputs — which is the property that made it displace ReLU in the
/// transformer feed-forward block.
void launch_silu(const float* input, float* output, int count, cudaStream_t stream);

/// SwiGLU: `silu(gate) * up`.
///
/// The activation used by LLaMA-family feed-forward blocks. The layer computes
/// two projections of the same input — a gate and an up-projection — and
/// multiplies the gated one elementwise:
///
///     FFN(x) = (silu(x W_gate) * (x W_up)) W_down
///
/// ## Why this deserves a kernel
///
/// Written with framework primitives it is three passes: compute `silu(gate)`,
/// write it, read it back, multiply by `up`, write again. Every one of those
/// crosses the memory bus, and the arithmetic — one sigmoid and two multiplies
/// per element — is free by comparison.
///
/// Fused, it is one read of each input and one write. That is the minimum
/// possible traffic, so the fused kernel is within a small factor of the
/// hardware limit no matter how much more effort is spent on it.
///
/// `gate` and `up` must have the same length. `output` may alias either.
void launch_swiglu(const float* gate, const float* up, float* output, int count,
                   SwiGLUKernel variant, cudaStream_t stream);

/// Half-precision SwiGLU.
///
/// The sigmoid is evaluated in FP32 and the result stored back as FP16. FP16
/// has enough range for the values themselves, but `exp` of a moderately
/// negative input underflows a 10-bit mantissa to zero far earlier than FP32
/// does, which flattens the activation's negative tail.
void launch_swiglu_half(const __half* gate, const __half* up, __half* output, int count,
                        cudaStream_t stream);

/// GELU, tanh approximation — the activation used by GPT-2 and BERT.
///
/// The tanh form rather than the exact erf form because that is what those
/// models were trained with; substituting the exact version changes outputs by
/// more than the numerical difference suggests, since the weights were fitted
/// against this curve.
void launch_gelu(const float* input, float* output, int count, cudaStream_t stream);

/// Elementwise `output = a + b`.
///
/// Present as the unfused baseline the fused residual+RMSNorm kernel is
/// measured against. Comparing the fused kernel against nothing would say
/// nothing; comparing it against the two launches it replaces is the whole
/// point.
void launch_add(const float* a, const float* b, float* output, int count,
                cudaStream_t stream);

}  // namespace cudaforge

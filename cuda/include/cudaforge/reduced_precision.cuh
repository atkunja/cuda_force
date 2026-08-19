#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace cudaforge {

/// Conversion traits for the two 16-bit formats, so a kernel can be written
/// once and instantiated for both.
///
/// ## Why both exist
///
/// They have the same width and very different shapes:
///
/// | | Exponent | Mantissa | Max | Smallest normal |
/// | --- | --- | --- | --- | --- |
/// | FP16 | 5 bits | 10 bits | 65,504 | 6.1e-5 |
/// | BF16 | 8 bits | 7 bits | 3.4e38 | 1.2e-38 |
/// | FP32 | 8 bits | 23 bits | 3.4e38 | 1.2e-38 |
///
/// BF16 has **float32's exponent range** with three fewer mantissa bits. That
/// trade is why it displaced FP16 for training and, increasingly, inference:
/// activations and gradients span many orders of magnitude, and running out of
/// range produces infinities that poison everything downstream, whereas running
/// out of precision merely adds noise.
///
/// Concretely, an activation of magnitude 300 squares to 90,000 and overflows
/// FP16 — the failure mode `rmsnorm_half` exists to avoid by accumulating in
/// FP32. In BF16 the same value is unremarkable.
///
/// ## Why arithmetic is still FP32
///
/// BF16's 7-bit mantissa is *worse* than FP16's for accumulation: summing more
/// than about 256 terms of similar magnitude stops making progress. Both
/// formats are therefore storage-only here, with reductions in FP32. That costs
/// nothing on a bandwidth-bound kernel, where the bytes moved — not the width
/// of the adder — set the time.
template <typename T>
struct ReducedPrecision;

template <>
struct ReducedPrecision<__half> {
    __device__ __forceinline__ static float to_float(__half value) {
        return __half2float(value);
    }
    __device__ __forceinline__ static __half from_float(float value) {
        return __float2half(value);
    }
    static constexpr const char* name() { return "fp16"; }
};

template <>
struct ReducedPrecision<__nv_bfloat16> {
    __device__ __forceinline__ static float to_float(__nv_bfloat16 value) {
        return __bfloat162float(value);
    }
    __device__ __forceinline__ static __nv_bfloat16 from_float(float value) {
        return __float2bfloat16(value);
    }
    static constexpr const char* name() { return "bf16"; }
};

}  // namespace cudaforge

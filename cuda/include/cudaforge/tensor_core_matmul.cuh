#pragma once

#include <cuda_runtime.h>

namespace cudaforge {

/// Matmul on the tensor cores: `C[m, n] = A[m, k] B[k, n]`, row-major FP32.
///
/// ## Why this exists
///
/// `launch_matmul` stages 16x16 tiles in shared memory and reuses each loaded
/// element sixteen times. That fixes the *memory* problem and leaves the
/// arithmetic one: every product is a scalar FMA on the CUDA cores, while the
/// tensor cores sitting beside them do a 16x16x8 matrix multiply per
/// instruction. On an RTX 3090 that is roughly an order of magnitude of
/// throughput left unused.
///
/// ## TF32, and what it costs
///
/// The interface stays FP32 because everything calling it is FP32. Ampere's
/// tensor cores cannot multiply FP32 directly, but they do **TF32**: the same
/// 8-bit exponent as FP32 with the mantissa truncated from 23 bits to 10. So
/// the range is unchanged — nothing overflows that would not have overflowed —
/// and the precision drops to roughly three decimal digits per product.
///
/// Accumulation stays in full FP32, which is what keeps the error from
/// compounding across `k`. For neural network inference this is the standard
/// trade and the reason TF32 exists; for anything needing reproducible FP32
/// arithmetic it is the wrong kernel, and `launch_matmul` is still there.
///
/// ## The alignment requirement
///
/// WMMA operates on whole 16x16x8 fragments. Rather than pad or mask, this
/// refuses shapes it cannot tile exactly and reports so, leaving the caller to
/// use `launch_matmul`. Silently falling back would hide a performance cliff
/// behind an interface that looks like it succeeded.
///
/// Requires compute capability 8.0 or later. Returns false without launching on
/// anything older, or on a misaligned shape.
[[nodiscard]] bool launch_matmul_tensor_core(const float* a, const float* b, float* c, int m, int n,
                                             int k, cudaStream_t stream);

/// True when this device has tensor cores this kernel can use (sm_80+).
[[nodiscard]] bool tensor_cores_available();

/// True when `launch_matmul_tensor_core` will accept these dimensions.
///
/// Exposed so a caller can choose a path without launching and checking, and so
/// the benchmark can report *why* a shape was skipped rather than silently
/// omitting it.
[[nodiscard]] bool tensor_core_shape_supported(int m, int n, int k);

}  // namespace cudaforge

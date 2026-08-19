#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace cudaforge {

/// Elements sharing one scale factor.
///
/// Per-tensor quantisation is cheap but a single outlier stretches the range
/// for every other value, and transformer weights have outliers. Per-element
/// scales would be exact and would also defeat the point, since the scales
/// would cost more than the saved weight bytes. Block-wise is the compromise
/// the field settled on: 64 elements per scale adds 4 bytes per 64 bytes of
/// INT8 payload — about 6% overhead — while confining any outlier's influence
/// to its own block.
inline constexpr int kQuantBlockSize = 64;

/// Symmetric absmax INT8 quantisation, block-wise.
///
///     scale_b = max(|x|) over block b / 127
///     q_i     = round(x_i / scale_b)   clamped to [-127, 127]
///
/// Symmetric (no zero point) because it keeps dequantisation a single multiply
/// and maps exact zeros to exact zeros — which matters for padding and masks.
/// The range stops at 127 rather than 128 so that negation is representable and
/// the quantisation grid is symmetric about zero.
///
/// ## What this is and is not
///
/// This is a self-contained, readable implementation of INT8 absmax
/// quantisation. It is **not** a reimplementation of bitsandbytes' NF4, which
/// uses a non-uniform 4-bit grid derived from the normal distribution's
/// quantiles, plus double quantisation of the scales themselves. Nothing here
/// reproduces that, and the QLoRA path in `training/` calls bitsandbytes
/// directly rather than pretending this is equivalent.
///
/// `scales` must have `ceil(count / kQuantBlockSize)` entries.
void launch_quantize_int8(const float* input, std::int8_t* output, float* scales,
                          int count, cudaStream_t stream);

/// Inverse of `launch_quantize_int8`. Lossy by construction: the round-trip
/// error is bounded by half a quantisation step, i.e. `scale_b / 2`.
void launch_dequantize_int8(const std::int8_t* input, const float* scales, float* output,
                            int count, cudaStream_t stream);

/// Quantises and immediately dequantises, producing FP32 output on the INT8
/// grid. This is the "fake quantisation" used to measure the accuracy cost of a
/// quantisation scheme without changing any downstream kernel's dtype.
void launch_quantize_dequantize(const float* input, float* output, int count,
                                cudaStream_t stream);

/// Number of block scales `count` elements require.
[[nodiscard]] int quant_scale_count(int count);

}  // namespace cudaforge

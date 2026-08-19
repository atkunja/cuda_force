#pragma once

#include <cuda_runtime.h>

namespace cudaforge {

enum class LoRAKernel {
    /// Computes `X A` into a scratch buffer, then `(X A) B` and the frozen
    /// `X W`, in three launches. Straightforward and easy to verify; the
    /// intermediate goes to global memory and back.
    Unfused,

    /// A single tiled kernel that computes the frozen and adapter paths
    /// together. `X A` is held in shared memory and consumed immediately, so
    /// the `batch x rank` intermediate never reaches global memory and two
    /// kernel launches disappear.
    Fused,
};

/// LoRA-adapted linear layer.
///
///     Y = X W + scale * (X A) B
///
/// Shapes (all row-major):
///   X : [batch, in_features]
///   W : [in_features, out_features]   frozen base weight
///   A : [in_features, rank]           adapter down-projection
///   B : [rank, out_features]          adapter up-projection
///   Y : [batch, out_features]
///
/// ## Why the low-rank path is worth a custom kernel
///
/// `rank` is typically 8 to 64 against `in_features` in the thousands, so
/// `X A` is a tall, extremely thin matrix. The intermediate is small enough to
/// live in shared memory but large enough that a round trip through global
/// memory costs more than the multiply itself. That imbalance — trivial
/// arithmetic, disproportionate memory traffic — is exactly what kernel fusion
/// removes, which is why the fused variant exists.
///
/// `scale` is the usual `alpha / rank`, applied by the caller so the kernel
/// stays agnostic to the convention.
///
/// `workspace` must hold at least `batch * rank` floats for the unfused
/// variant, and is ignored by the fused one. Passing it in rather than
/// allocating internally keeps the kernel off the allocator path, where a
/// `cudaMalloc` would synchronise the device once per layer per token.
void launch_lora_linear(const float* x, const float* w, const float* a, const float* b,
                        float* y, float* workspace, int batch, int in_features,
                        int out_features, int rank, float scale, LoRAKernel variant,
                        cudaStream_t stream);

/// Plain tiled matmul: `C[m, n] = A[m, k] B[k, n]`, row-major.
///
/// Present because the adapter path needs it and because it is the baseline the
/// fused kernel is compared against. It is not competitive with cuBLAS and is
/// not meant to be — see docs/cuda-kernels.md.
void launch_matmul(const float* a, const float* b, float* c, int m, int n, int k,
                   cudaStream_t stream);

/// Bytes of workspace `launch_lora_linear` needs for the unfused variant.
[[nodiscard]] std::size_t lora_workspace_bytes(int batch, int rank);

}  // namespace cudaforge

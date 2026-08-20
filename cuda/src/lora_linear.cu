#include "cudaforge/lora_linear.cuh"

#include <algorithm>
#include <stdexcept>

#include "cudaforge/cuda_error.cuh"
#include "cudaforge/cuda_utils.cuh"

namespace cudaforge {
namespace {

/// Tile edge for the shared-memory matmul.
///
/// 16 gives 256 threads per block, a 16x16 float tile of each operand at 1 KB
/// apiece, and 16 reuses of every loaded element. Larger tiles improve the
/// reuse ratio but the shared-memory footprint grows quadratically and
/// occupancy falls; 32 would need 4 KB per operand and halves the blocks
/// resident per SM on most parts.
constexpr int kTile = 16;

/// Shared-memory tiled matmul.
///
/// The naive formulation reads a full row of A and column of B from global
/// memory for every output element, so each element of A is re-read `n` times.
/// Staging tiles in shared memory means each element is read from global memory
/// once per tile and then reused `kTile` times, cutting global traffic by that
/// factor. This is the canonical case of trading on-chip storage for off-chip
/// bandwidth.
///
/// The bounds checks make the kernel correct for dimensions that are not
/// multiples of kTile: out-of-range loads write zero into the tile, which
/// contributes nothing to the dot product.
__global__ void matmul_tiled(const float* __restrict__ a, const float* __restrict__ b,
                             float* __restrict__ c, int m, int n, int k) {
    __shared__ float tile_a[kTile][kTile];
    // Padding by one float breaks the power-of-two stride so that a column
    // access hits kTile distinct banks instead of all mapping to one. Without
    // it the column reads in the inner loop are a 16-way bank conflict.
    __shared__ float tile_b[kTile][kTile + 1];

    const int row = static_cast<int>(blockIdx.y * kTile + threadIdx.y);
    const int col = static_cast<int>(blockIdx.x * kTile + threadIdx.x);

    float accumulator = 0.0F;

    const int tiles = ceil_div(k, kTile);
    for (int t = 0; t < tiles; ++t) {
        const int a_col = t * kTile + static_cast<int>(threadIdx.x);
        const int b_row = t * kTile + static_cast<int>(threadIdx.y);

        tile_a[threadIdx.y][threadIdx.x] =
            (row < m && a_col < k) ? a[static_cast<std::size_t>(row) * k + a_col] : 0.0F;
        tile_b[threadIdx.y][threadIdx.x] =
            (b_row < k && col < n) ? b[static_cast<std::size_t>(b_row) * n + col] : 0.0F;
        __syncthreads();

#pragma unroll
        for (int i = 0; i < kTile; ++i) {
            accumulator += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
        }
        // Required before the next iteration overwrites the tiles: without it,
        // a fast warp could begin loading tile t+1 while a slow warp is still
        // reading tile t.
        __syncthreads();
    }

    if (row < m && col < n) {
        c[static_cast<std::size_t>(row) * n + col] = accumulator;
    }
}

/// Adds `scale * (P B)` into an existing `Y`, where `P` is the `batch x rank`
/// adapter intermediate. Accumulating in place avoids materialising a second
/// `batch x out_features` buffer just to add it.
__global__ void accumulate_lora(const float* __restrict__ p, const float* __restrict__ b,
                                float* __restrict__ y, int batch, int out_features, int rank,
                                float scale) {
    const int row = static_cast<int>(blockIdx.y * kTile + threadIdx.y);
    const int col = static_cast<int>(blockIdx.x * kTile + threadIdx.x);
    if (row >= batch || col >= out_features) {
        return;
    }

    float accumulator = 0.0F;
    for (int r = 0; r < rank; ++r) {
        accumulator += p[static_cast<std::size_t>(row) * rank + r] *
                       b[static_cast<std::size_t>(r) * out_features + col];
    }
    y[static_cast<std::size_t>(row) * out_features + col] += scale * accumulator;
}

/// Fused LoRA.
///
/// Each block owns `kTile` rows of the batch. It first computes that block's
/// slice of `X A` into shared memory — `kTile x rank` floats, small because
/// rank is small — and then walks the output columns, adding the frozen `X W`
/// contribution and the adapter contribution in the same pass.
///
/// What this saves versus the unfused path:
///   - two kernel launches,
///   - a `batch x rank` write to global memory and the matching read,
///   - a re-read of `X` for the adapter path, since the rows are already
///     resident when `X A` is computed.
///
/// The constraint is that `rank * kTile` floats must fit in the dynamic shared
/// memory the launcher requests; the launcher checks and falls back when it
/// does not.
/// Fused LoRA: the frozen and adapter paths in one launch, with the
/// `batch x rank` intermediate held in shared memory instead of global.
///
/// ## What the first version got wrong
///
/// It kept `X A` in shared memory and then computed the frozen path with an
/// untiled loop: every thread walked the whole of `in_features`, reading `x`
/// and `w` from global memory with no staging and no reuse. Measured on an
/// RTX 3090 that made it **21.8-32.3x slower** than the unfused path, which
/// simply calls `matmul_tiled`.
///
/// The fusion was never the problem. Dropping the tiling to make room for the
/// fusion was. A few thousand floats of intermediate were saved against 16x
/// reuse given up on a `batch x in x out` matmul — an obviously bad trade once
/// the two are written next to each other, and one no amount of reading the
/// source had revealed until it was measured.
///
/// This version tiles the frozen path exactly as `matmul_tiled` does and keeps
/// the fusion. The grid is two-dimensional again, so each block owns one
/// `kTile x kTile` tile of the output and there is real parallelism to fill the
/// device with.
///
/// ## The cost that remains
///
/// Every block in a row-strip recomputes `X A` for its rows, because the strip
/// is now split across `out_features / kTile` blocks. That is `rank / kTile` of
/// the frozen path's arithmetic — 50% extra at rank 8, 100% at rank 16 —
/// against a 16x reuse win. Worth it, but it is the reason a genuinely fast
/// implementation would compute `X A` in its own launch and fuse only the
/// second half.
__global__ void lora_fused(const float* __restrict__ x, const float* __restrict__ w,
                           const float* __restrict__ a, const float* __restrict__ b,
                           float* __restrict__ y, int batch, int in_features, int out_features,
                           int rank, float scale) {
    __shared__ float tile_x[kTile][kTile];
    // Padded exactly as in `matmul_tiled`: the column-wise read below would
    // otherwise put every thread of a warp in the same bank.
    __shared__ float tile_w[kTile][kTile + 1];
    extern __shared__ float xa_tile[];  // [kTile][rank]

    const int row = static_cast<int>(blockIdx.y) * kTile + static_cast<int>(threadIdx.y);
    const int col = static_cast<int>(blockIdx.x) * kTile + static_cast<int>(threadIdx.x);
    const int local_row = static_cast<int>(threadIdx.y);

    // Phase 1: X A for this block's rows. No barrier inside the loop, so the
    // uneven trip count when `rank` is not a multiple of kTile is harmless.
    for (int r = static_cast<int>(threadIdx.x); r < rank; r += kTile) {
        float accumulator = 0.0F;
        if (row < batch) {
            for (int i = 0; i < in_features; ++i) {
                accumulator += x[static_cast<std::size_t>(row) * in_features + i] *
                               a[static_cast<std::size_t>(i) * rank + r];
            }
        }
        xa_tile[local_row * rank + r] = accumulator;
    }
    __syncthreads();

    // Phase 2: the frozen path, tiled. Every thread reaches every barrier —
    // the early return on out-of-range rows that the previous version used
    // would deadlock here.
    float frozen = 0.0F;
    const int tiles = ceil_div(in_features, kTile);
    for (int t = 0; t < tiles; ++t) {
        const int k_for_x = t * kTile + static_cast<int>(threadIdx.x);
        tile_x[threadIdx.y][threadIdx.x] =
            (row < batch && k_for_x < in_features)
                ? x[static_cast<std::size_t>(row) * in_features + k_for_x]
                : 0.0F;

        const int k_for_w = t * kTile + static_cast<int>(threadIdx.y);
        tile_w[threadIdx.y][threadIdx.x] =
            (k_for_w < in_features && col < out_features)
                ? w[static_cast<std::size_t>(k_for_w) * out_features + col]
                : 0.0F;
        __syncthreads();

        for (int i = 0; i < kTile; ++i) {
            frozen += tile_x[threadIdx.y][i] * tile_w[i][threadIdx.x];
        }
        // Before the next iteration overwrites the tiles.
        __syncthreads();
    }

    if (row >= batch || col >= out_features) {
        return;
    }

    float adapter = 0.0F;
    for (int r = 0; r < rank; ++r) {
        adapter +=
            xa_tile[local_row * rank + r] * b[static_cast<std::size_t>(r) * out_features + col];
    }

    y[static_cast<std::size_t>(row) * out_features + col] = frozen + scale * adapter;
}

int max_shared_memory_per_block() {
    static const int cached = [] {
        int device = 0;
        CUDAFORGE_CHECK(cudaGetDevice(&device));
        int bytes = 0;
        CUDAFORGE_CHECK(cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlock, device));
        return bytes;
    }();
    return cached;
}

}  // namespace

std::size_t lora_workspace_bytes(int batch, int rank) {
    if (batch <= 0 || rank <= 0) {
        return 0;
    }
    return static_cast<std::size_t>(batch) * static_cast<std::size_t>(rank) * sizeof(float);
}

void launch_matmul(const float* a, const float* b, float* c, int m, int n, int k,
                   cudaStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0) {
        return;
    }
    const dim3 block(kTile, kTile);
    const dim3 grid(static_cast<unsigned>(ceil_div(n, kTile)),
                    static_cast<unsigned>(ceil_div(m, kTile)));
    matmul_tiled<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
    CUDAFORGE_CHECK_LAUNCH(stream);
}

void launch_lora_linear(const float* x, const float* w, const float* a, const float* b, float* y,
                        float* workspace, int batch, int in_features, int out_features, int rank,
                        float scale, LoRAKernel variant, cudaStream_t stream) {
    if (batch <= 0 || in_features <= 0 || out_features <= 0 || rank <= 0) {
        return;
    }

    const std::size_t fused_shared =
        static_cast<std::size_t>(kTile) * static_cast<std::size_t>(rank) * sizeof(float);
    const bool fusable = variant == LoRAKernel::Fused &&
                         fused_shared <= static_cast<std::size_t>(max_shared_memory_per_block());

    if (fusable) {
        // Two-dimensional again: one kTile x kTile tile of the output per
        // block. The previous one-dimensional grid gave `batch / kTile` blocks
        // — two of them at batch 32 — which left an 82-SM device almost
        // entirely idle on top of doing untiled arithmetic.
        const dim3 block(kTile, kTile);
        const dim3 grid(static_cast<unsigned>(ceil_div(out_features, kTile)),
                        static_cast<unsigned>(ceil_div(batch, kTile)));
        lora_fused<<<grid, block, fused_shared, stream>>>(x, w, a, b, y, batch, in_features,
                                                          out_features, rank, scale);
        CUDAFORGE_CHECK_LAUNCH(stream);
        return;
    }

    if (workspace == nullptr) {
        throw std::invalid_argument("launch_lora_linear: unfused path requires a workspace");
    }

    // Frozen path: Y = X W.
    launch_matmul(x, w, y, batch, out_features, in_features, stream);
    // Adapter down-projection: P = X A, shape [batch, rank].
    launch_matmul(x, a, workspace, batch, rank, in_features, stream);
    // Adapter up-projection accumulated in place: Y += scale * (P B).
    const dim3 block(kTile, kTile);
    const dim3 grid(static_cast<unsigned>(ceil_div(out_features, kTile)),
                    static_cast<unsigned>(ceil_div(batch, kTile)));
    accumulate_lora<<<grid, block, 0, stream>>>(workspace, b, y, batch, out_features, rank, scale);
    CUDAFORGE_CHECK_LAUNCH(stream);
}

}  // namespace cudaforge

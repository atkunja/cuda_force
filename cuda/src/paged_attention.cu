#include "cudaforge/paged_attention.cuh"

#include <cfloat>
#include <cstddef>

#include "cudaforge/cuda_error.cuh"
#include "cudaforge/cuda_utils.cuh"

namespace cudaforge {
namespace {

/// One block per (sequence, head). Threads own one element of `head_dim` each,
/// so the dot product against a key is a block reduction and the weighted sum
/// over values needs no communication at all.
///
/// The loop over context is serial. That is the cost of the readable version:
/// a block reduction per token means two barriers per token, where a production
/// kernel would tile the context and keep several keys in flight. This one is
/// here to be correct and to show the block-table indirection plainly; it is
/// not competitive with a tiled implementation and is not claimed to be.
__global__ void paged_attention_kernel(
    const float* __restrict__ query, const float* __restrict__ k_cache,
    const float* __restrict__ v_cache, const DeviceBlockId* __restrict__ block_tables,
    const int* __restrict__ context_lens, float* __restrict__ out, int num_heads, int num_kv_heads,
    int head_dim, int block_size, int max_blocks_per_sequence, float scale) {
    extern __shared__ float reduction[];

    const int head = static_cast<int>(blockIdx.x);
    const int sequence = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int context = context_lens[sequence];

    // Computed in size_t: num_blocks * block_size * num_kv_heads * head_dim
    // overflows a 32-bit index on a cache of any real size.
    const std::size_t row =
        (static_cast<std::size_t>(sequence) * static_cast<std::size_t>(num_heads) +
         static_cast<std::size_t>(head)) *
        static_cast<std::size_t>(head_dim);

    if (context <= 0) {
        if (tid < head_dim) {
            out[row + static_cast<std::size_t>(tid)] = 0.0F;
        }
        return;
    }

    // Grouped-query attention: several query heads share one KV head. With
    // num_kv_heads == num_heads the group is 1 and this is ordinary attention.
    const int group = num_heads / num_kv_heads;
    const int kv_head = head / group;

    const float q = (tid < head_dim) ? query[row + static_cast<std::size_t>(tid)] : 0.0F;

    // Online softmax state. `running_max` starts at -inf so the first score
    // always becomes the maximum; the correction below is guarded for it,
    // because -inf minus -inf is NaN rather than the 0 the recurrence wants.
    float running_max = -FLT_MAX;
    float running_sum = 0.0F;
    float accumulator = 0.0F;

    for (int token = 0; token < context; ++token) {
        const DeviceBlockId block =
            block_tables[static_cast<std::size_t>(sequence) *
                             static_cast<std::size_t>(max_blocks_per_sequence) +
                         static_cast<std::size_t>(token / block_size)];
        const int offset = token % block_size;

        // The indirection the whole paged design exists for: token `token` is
        // at `offset` inside physical block `block`, not at index `token`.
        const std::size_t slot =
            ((static_cast<std::size_t>(block) * static_cast<std::size_t>(block_size) +
              static_cast<std::size_t>(offset)) *
                 static_cast<std::size_t>(num_kv_heads) +
             static_cast<std::size_t>(kv_head)) *
            static_cast<std::size_t>(head_dim);

        const float k = (tid < head_dim) ? k_cache[slot + static_cast<std::size_t>(tid)] : 0.0F;

        // Every thread must reach this; `context` is uniform across the block,
        // so the loop bound is not divergent.
        const float score = block_reduce_sum(q * k, reduction) * scale;

        const float updated_max = fmaxf(running_max, score);
        const float correction =
            (running_max == -FLT_MAX) ? 0.0F : __expf(running_max - updated_max);
        const float weight = __expf(score - updated_max);

        const float v = (tid < head_dim) ? v_cache[slot + static_cast<std::size_t>(tid)] : 0.0F;

        running_sum = running_sum * correction + weight;
        accumulator = accumulator * correction + weight * v;
        running_max = updated_max;
    }

    if (tid < head_dim) {
        out[row + static_cast<std::size_t>(tid)] = accumulator / running_sum;
    }
}

}  // namespace

void launch_paged_attention(const float* query, const float* k_cache, const float* v_cache,
                            const DeviceBlockId* block_tables, const int* context_lens, float* out,
                            int num_sequences, int num_heads, int num_kv_heads, int head_dim,
                            int block_size, int max_blocks_per_sequence, float scale,
                            cudaStream_t stream) {
    if (num_sequences <= 0 || num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
        block_size <= 0 || max_blocks_per_sequence <= 0) {
        return;
    }
    // Grouped-query attention divides the heads evenly; anything else is a
    // caller error rather than something to guess at.
    if (num_heads % num_kv_heads != 0) {
        return;
    }

    // One thread per element of head_dim, so a head wider than the maximum
    // block cannot be served. Refused rather than clamped: clamping would write
    // the first 1024 elements and leave the rest of every output row whatever
    // it happened to contain, which is silent corruption rather than a failure.
    if (head_dim > 1024) {
        return;
    }

    // Rounded up to a whole warp so the block reduction's warp arithmetic
    // holds. Threads past head_dim contribute zero and still reach every
    // barrier.
    const int threads = ceil_div(head_dim, kWarpSize) * kWarpSize;
    const int warps = ceil_div(threads, kWarpSize);
    const std::size_t shared = static_cast<std::size_t>(warps) * sizeof(float);

    const dim3 grid(static_cast<unsigned>(num_heads), static_cast<unsigned>(num_sequences));
    paged_attention_kernel<<<grid, threads, shared, stream>>>(
        query, k_cache, v_cache, block_tables, context_lens, out, num_heads, num_kv_heads, head_dim,
        block_size, max_blocks_per_sequence, scale);
    CUDAFORGE_CHECK_LAUNCH(stream);
}

}  // namespace cudaforge

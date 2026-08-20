#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace cudaforge {

/// Physical block identifier, matching `cudaforge::BlockId` on the host.
using DeviceBlockId = std::uint32_t;

/// Attention over a **paged** KV cache, for one decode step.
///
/// This is the kernel the block allocator exists for. `BlockAllocator` and
/// `SequenceBlockTable` do the bookkeeping — reference counting, copy-on-write,
/// eviction — but until something *reads* through a block table, a paged cache
/// buys nothing: the attention still has to be handed a contiguous tensor, so
/// the cache has to be contiguous.
///
/// ## Why the indirection is the whole point
///
/// A contiguous cache reserves `max_sequence_length` tokens per sequence
/// whether or not they are used. Real traffic is long-tailed, so most of that
/// reservation is waste — measured at 93% on chat-shaped traffic in
/// `bench_kv_cache`. A paged cache stores tokens in fixed-size blocks scattered
/// through one pool and gives each sequence a *block table*: a list of which
/// physical blocks hold its logical tokens, in order. Internal fragmentation
/// falls to at most `block_size - 1` tokens per sequence.
///
/// The cost is that token `t` is no longer at offset `t`. It lives at
///
///     block  = block_table[sequence][t / block_size]
///     offset = t % block_size
///
/// which is an extra dependent load per token — and that is what this kernel
/// pays in exchange for the memory.
///
/// ## Layouts
///
/// * `query`         `[num_sequences, num_heads, head_dim]` — one token per
///                   sequence, which is what a decode step has.
/// * `k_cache`,
///   `v_cache`       `[num_blocks, block_size, num_kv_heads, head_dim]`
/// * `block_tables`  `[num_sequences, max_blocks_per_sequence]`
/// * `context_lens`  `[num_sequences]` — tokens each sequence actually holds,
///                   so sequences of different lengths share a launch.
/// * `out`           `[num_sequences, num_heads, head_dim]`
///
/// `num_heads` may exceed `num_kv_heads` for grouped-query attention; each
/// query head reads the KV head `head / (num_heads / num_kv_heads)`.
///
/// ## Numerics
///
/// Scores are accumulated in FP32 and the softmax is online: a running maximum
/// and sum are rescaled whenever a larger score appears, so no scores array is
/// materialised and context length is bounded only by the cache, not by shared
/// memory. Same recurrence as the `Online` softmax variant, for the same reason.
/// `head_dim` must be at most 1024 (one thread serves one element) and
/// `num_heads` must be a multiple of `num_kv_heads`. Both are refused rather
/// than approximated: a clamped `head_dim` would leave part of every output row
/// untouched, which is worse than doing nothing.
void launch_paged_attention(const float* query, const float* k_cache, const float* v_cache,
                            const DeviceBlockId* block_tables, const int* context_lens, float* out,
                            int num_sequences, int num_heads, int num_kv_heads, int head_dim,
                            int block_size, int max_blocks_per_sequence, float scale,
                            cudaStream_t stream);

}  // namespace cudaforge

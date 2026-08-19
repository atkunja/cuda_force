#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace cudaforge {

/// Identifier for a physical block of KV cache memory.
using BlockId = std::uint32_t;

/// Opaque handle for a sequence being served.
using SequenceId = std::uint64_t;

inline constexpr BlockId kInvalidBlock = static_cast<BlockId>(-1);

/// Paged KV cache block allocator.
///
/// ## The problem it solves
///
/// A contiguous KV cache must be sized for the longest sequence the server will
/// accept, for every slot, up front. A server configured for 2048 tokens and
/// serving 100-token requests wastes 95% of its cache — and that waste is what
/// caps concurrency, because concurrency is cache-bound long before it is
/// compute-bound.
///
/// Paging removes the contiguity requirement. The cache is divided into
/// fixed-size blocks; a sequence holds a *block table* mapping its logical
/// block indices to physical blocks, and grows by one block at a time. Internal
/// fragmentation falls to at most one partly-filled block per sequence — a few
/// tokens instead of thousands.
///
/// This is the allocator underneath that scheme. It is deliberately host-side
/// and device-agnostic: the hard parts are bookkeeping, reference counting and
/// eviction policy, none of which need a GPU to be correct, and all of which
/// are difficult to debug once they are entangled with device memory.
///
/// ## Reference counting
///
/// Blocks are reference counted so that two sequences sharing a prefix — a
/// common system prompt, or a beam-search fork — can share physical blocks
/// rather than duplicating them. A block is returned to the free list only when
/// its last referent releases it.
///
/// A shared block cannot be appended to: writing would corrupt the other
/// referent. `is_writable` reports that, and the caller must copy-on-write
/// before extending.
class BlockAllocator {
public:
    /// `block_count` physical blocks, each holding `block_size` tokens' worth
    /// of keys and values.
    ///
    /// `block_size` is a real tradeoff: larger blocks mean fewer table entries
    /// and better locality, smaller blocks mean less internal fragmentation in
    /// the last block of each sequence. 16 is the common choice.
    BlockAllocator(std::size_t block_count, std::size_t block_size);

    BlockAllocator(const BlockAllocator&) = delete;
    BlockAllocator& operator=(const BlockAllocator&) = delete;

    /// Takes a free block. Returns nullopt when exhausted rather than throwing:
    /// running out of cache is an expected condition under load, and the
    /// scheduler's response is to preempt a sequence, not to unwind.
    [[nodiscard]] std::optional<BlockId> allocate();

    /// Adds a reference. Used when a sequence forks and shares a prefix.
    void add_reference(BlockId block);

    /// Drops a reference, freeing the block when the count reaches zero.
    void release(BlockId block);

    [[nodiscard]] std::uint32_t reference_count(BlockId block) const;

    /// False when the block has more than one referent, so appending to it
    /// would corrupt another sequence.
    [[nodiscard]] bool is_writable(BlockId block) const;

    [[nodiscard]] std::size_t free_blocks() const;
    [[nodiscard]] std::size_t used_blocks() const;
    [[nodiscard]] std::size_t total_blocks() const noexcept { return total_; }
    [[nodiscard]] std::size_t block_size() const noexcept { return block_size_; }

    /// Fraction of blocks currently in use. The signal a scheduler admits or
    /// preempts on.
    [[nodiscard]] double utilisation() const;

private:
    mutable std::mutex mutex_;
    std::size_t total_;
    std::size_t block_size_;
    std::vector<BlockId> free_list_;
    std::vector<std::uint32_t> reference_counts_;
};

}  // namespace cudaforge

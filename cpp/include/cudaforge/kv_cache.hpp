#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
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

/// Per-sequence mapping from logical block index to physical block.
///
/// A sequence's tokens are laid out logically as
/// `[block 0][block 1]...[block n]`, but the physical blocks backing them are
/// wherever the allocator had space. This table is what the attention kernel
/// would index through — the indirection that makes the cache non-contiguous
/// and therefore fragmentation-free.
class SequenceBlockTable {
public:
    SequenceBlockTable(SequenceId id, std::size_t block_size) : id_(id), block_size_(block_size) {}

    [[nodiscard]] SequenceId id() const noexcept { return id_; }
    [[nodiscard]] std::size_t token_count() const noexcept { return tokens_; }
    [[nodiscard]] const std::vector<BlockId>& blocks() const noexcept { return blocks_; }

    /// Tokens that fit in the blocks already held.
    [[nodiscard]] std::size_t capacity() const noexcept { return blocks_.size() * block_size_; }

    /// Unused slots in the final block. This is the *entire* internal
    /// fragmentation of a paged cache: at most `block_size - 1` tokens per
    /// sequence, against thousands for a contiguous cache sized to the maximum
    /// sequence length.
    [[nodiscard]] std::size_t slack() const noexcept { return capacity() - tokens_; }

    /// True when the next token needs a block the sequence does not yet hold.
    [[nodiscard]] bool needs_block() const noexcept { return tokens_ >= capacity(); }

    void append_block(BlockId block) { blocks_.push_back(block); }

    /// Records `count` more tokens. Throws if they do not fit — the caller must
    /// have allocated first, and silently overrunning would corrupt the block
    /// belonging to whichever sequence holds the next one.
    void add_tokens(std::size_t count);

    /// Replaces a logical block, for copy-on-write when a shared block must be
    /// written to.
    void replace_block(std::size_t logical_index, BlockId block);

    /// Physical block holding a given token, and its offset within it.
    [[nodiscard]] std::pair<BlockId, std::size_t> locate(std::size_t token_index) const;

private:
    SequenceId id_;
    std::size_t block_size_;
    std::size_t tokens_ = 0;
    std::vector<BlockId> blocks_;
};

}  // namespace cudaforge

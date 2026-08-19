#include "cudaforge/kv_cache.hpp"

#include <algorithm>
#include <stdexcept>

namespace cudaforge {

BlockAllocator::BlockAllocator(std::size_t block_count, std::size_t block_size)
    : total_(block_count), block_size_(block_size) {
    if (block_count == 0) {
        throw std::invalid_argument("BlockAllocator requires at least one block");
    }
    if (block_size == 0) {
        throw std::invalid_argument("BlockAllocator block_size must be non-zero");
    }

    free_list_.reserve(block_count);
    // Pushed in reverse so the first allocation returns block 0. Purely for
    // legibility in traces and tests; the allocator is order-agnostic.
    for (std::size_t i = block_count; i > 0; --i) {
        free_list_.push_back(static_cast<BlockId>(i - 1));
    }
    reference_counts_.assign(block_count, 0);
}

std::optional<BlockId> BlockAllocator::allocate() {
    std::lock_guard lock(mutex_);
    if (free_list_.empty()) {
        return std::nullopt;
    }
    const BlockId block = free_list_.back();
    free_list_.pop_back();
    reference_counts_[block] = 1;
    return block;
}

void BlockAllocator::add_reference(BlockId block) {
    std::lock_guard lock(mutex_);
    if (block >= reference_counts_.size()) {
        throw std::out_of_range("BlockAllocator::add_reference on an unknown block");
    }
    if (reference_counts_[block] == 0) {
        throw std::invalid_argument("BlockAllocator::add_reference on a free block");
    }
    ++reference_counts_[block];
}

void BlockAllocator::release(BlockId block) {
    std::lock_guard lock(mutex_);
    if (block >= reference_counts_.size()) {
        throw std::out_of_range("BlockAllocator::release on an unknown block");
    }
    if (reference_counts_[block] == 0) {
        // A double free here would silently hand the same block to two
        // sequences, and the resulting corruption would look like a model bug.
        throw std::invalid_argument("BlockAllocator::release on an already-free block");
    }
    if (--reference_counts_[block] == 0) {
        free_list_.push_back(block);
    }
}

std::uint32_t BlockAllocator::reference_count(BlockId block) const {
    std::lock_guard lock(mutex_);
    if (block >= reference_counts_.size()) {
        throw std::out_of_range("BlockAllocator::reference_count on an unknown block");
    }
    return reference_counts_[block];
}

bool BlockAllocator::is_writable(BlockId block) const {
    std::lock_guard lock(mutex_);
    if (block >= reference_counts_.size()) {
        throw std::out_of_range("BlockAllocator::is_writable on an unknown block");
    }
    return reference_counts_[block] == 1;
}

std::size_t BlockAllocator::free_blocks() const {
    std::lock_guard lock(mutex_);
    return free_list_.size();
}

std::size_t BlockAllocator::used_blocks() const {
    std::lock_guard lock(mutex_);
    return total_ - free_list_.size();
}

double BlockAllocator::utilisation() const {
    std::lock_guard lock(mutex_);
    return static_cast<double>(total_ - free_list_.size()) / static_cast<double>(total_);
}

void SequenceBlockTable::add_tokens(std::size_t count) {
    if (tokens_ + count > capacity()) {
        throw std::out_of_range(
            "SequenceBlockTable::add_tokens beyond the allocated blocks; allocate first");
    }
    tokens_ += count;
}

void SequenceBlockTable::replace_block(std::size_t logical_index, BlockId block) {
    if (logical_index >= blocks_.size()) {
        throw std::out_of_range("SequenceBlockTable::replace_block on an unheld block");
    }
    blocks_[logical_index] = block;
}

std::pair<BlockId, std::size_t> SequenceBlockTable::locate(std::size_t token_index) const {
    if (token_index >= tokens_) {
        throw std::out_of_range("SequenceBlockTable::locate past the end of the sequence");
    }
    // The whole point of the scheme: a logical token index resolves through the
    // table to an arbitrary physical block, so the cache need not be contiguous.
    return {blocks_[token_index / block_size_], token_index % block_size_};
}

}  // namespace cudaforge

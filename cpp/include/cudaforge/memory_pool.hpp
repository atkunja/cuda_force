#pragma once

#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace cudaforge {

/// A backend supplies raw allocation; the pool supplies reuse. Splitting them
/// this way is what makes the pool testable on a machine with no GPU: the
/// caching, size-class and accounting logic is exercised against
/// `HostAllocatorBackend` locally, and only the two-line device backend is
/// hardware-dependent.
template <typename B>
concept AllocatorBackend = requires(B backend, void* pointer, std::size_t bytes) {
    { backend.allocate(bytes) } -> std::same_as<void*>;
    { backend.deallocate(pointer, bytes) } noexcept;
    { B::name() } -> std::convertible_to<const char*>;
};

struct PoolStats {
    std::size_t bytes_reserved = 0;  ///< total obtained from the backend
    std::size_t bytes_in_use = 0;    ///< currently handed out to callers
    std::size_t peak_bytes_in_use = 0;
    std::uint64_t allocation_count = 0;  ///< calls to allocate()
    std::uint64_t backend_allocations = 0;
    std::uint64_t reuse_count = 0;  ///< allocations served from a free list
    std::uint64_t free_count = 0;
    std::size_t free_block_count = 0;

    [[nodiscard]] double reuse_rate() const {
        return allocation_count > 0
                   ? static_cast<double>(reuse_count) / static_cast<double>(allocation_count)
                   : 0.0;
    }
};

/// Caching allocator with power-of-two size classes.
///
/// ## Why this exists
///
/// `cudaMalloc` and `cudaFree` synchronise the device. In a serving loop that
/// allocates activation buffers per batch, that turns every allocation into a
/// pipeline stall, serialising work the stream scheduler exists to overlap.
/// Holding device memory and reusing it removes those stalls entirely.
///
/// ## Size classes
///
/// Requests are rounded up to the next power of two, which caps internal
/// fragmentation at just under 2x while making the free lists O(1) to index and
/// making blocks freely interchangeable within a class. A best-fit allocator
/// over exact sizes would waste less memory but needs a search on every
/// allocation and fragments over time as odd-sized holes accumulate. For
/// inference, where shapes repeat batch after batch, exact reuse dominates and
/// the rounding rarely costs anything after warmup.
///
/// ## What this is not
///
/// There is no splitting, no coalescing, no defragmentation, and no
/// stream-ordered semantics. Those are what make a production allocator hard,
/// and PyTorch's caching allocator and `cudaMallocAsync` already implement
/// them. See docs/memory-management.md for the comparison.
///
/// ## Thread safety
///
/// One mutex guards every free list and the statistics. Allocation is short and
/// uncontended in practice because it happens once per batch, not per kernel.
template <AllocatorBackend Backend>
class MemoryPool {
public:
    explicit MemoryPool(Backend backend = Backend{}, std::size_t min_block_bytes = 256)
        : backend_(std::move(backend)), min_block_bytes_(std::bit_ceil(min_block_bytes)) {
        if (min_block_bytes == 0) {
            throw std::invalid_argument("min_block_bytes must be non-zero");
        }
    }

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    MemoryPool(MemoryPool&&) = delete;
    MemoryPool& operator=(MemoryPool&&) = delete;

    /// Returns memory to the pool's owner. Any block still checked out at this
    /// point is leaked by the caller, not by the pool — the destructor releases
    /// the backing memory regardless, because holding it would leak for real.
    ~MemoryPool() { release_all(); }

    [[nodiscard]] void* allocate(std::size_t bytes) {
        if (bytes == 0) {
            return nullptr;
        }
        const std::size_t block_bytes = round_up(bytes);

        std::lock_guard lock(mutex_);
        stats_.allocation_count++;

        auto& free_list = free_lists_[block_bytes];
        if (!free_list.empty()) {
            void* pointer = free_list.back();
            free_list.pop_back();
            stats_.reuse_count++;
            stats_.free_block_count--;
            note_checkout(pointer, block_bytes);
            return pointer;
        }

        void* pointer = backend_.allocate(block_bytes);
        if (pointer == nullptr) {
            throw std::bad_alloc();
        }
        stats_.backend_allocations++;
        stats_.bytes_reserved += block_bytes;
        note_checkout(pointer, block_bytes);
        return pointer;
    }

    /// Returns a block to its free list. The size is looked up rather than
    /// taken from the caller, so a caller that reports the wrong size cannot
    /// corrupt the free lists.
    void deallocate(void* pointer) {
        if (pointer == nullptr) {
            return;
        }
        std::lock_guard lock(mutex_);

        auto entry = live_blocks_.find(pointer);
        if (entry == live_blocks_.end()) {
            throw std::invalid_argument("MemoryPool::deallocate on a pointer it did not hand out");
        }
        const std::size_t block_bytes = entry->second;
        live_blocks_.erase(entry);

        stats_.bytes_in_use -= block_bytes;
        stats_.free_count++;
        stats_.free_block_count++;
        free_lists_[block_bytes].push_back(pointer);
    }

    /// Hands cached blocks back to the backend. Useful before a phase that
    /// needs the memory for something else — for example, before loading a
    /// second model. Blocks currently checked out are untouched.
    void trim() {
        std::lock_guard lock(mutex_);
        for (auto& [block_bytes, free_list] : free_lists_) {
            for (void* pointer : free_list) {
                backend_.deallocate(pointer, block_bytes);
                stats_.bytes_reserved -= block_bytes;
            }
            free_list.clear();
        }
        stats_.free_block_count = 0;
    }

    [[nodiscard]] PoolStats stats() const {
        std::lock_guard lock(mutex_);
        return stats_;
    }

    [[nodiscard]] static const char* backend_name() { return Backend::name(); }

private:
    /// Rounds to a power of two, but never below the minimum block size.
    /// Without the floor, a workload allocating many 8-byte scratch buffers
    /// would create size classes far below any useful granularity and defeat
    /// reuse across slightly different shapes.
    [[nodiscard]] std::size_t round_up(std::size_t bytes) const {
        return std::max(min_block_bytes_, std::bit_ceil(bytes));
    }

    void note_checkout(void* pointer, std::size_t block_bytes) {
        live_blocks_[pointer] = block_bytes;
        stats_.bytes_in_use += block_bytes;
        stats_.peak_bytes_in_use = std::max(stats_.peak_bytes_in_use, stats_.bytes_in_use);
    }

    void release_all() noexcept {
        std::lock_guard lock(mutex_);
        for (auto& [block_bytes, free_list] : free_lists_) {
            for (void* pointer : free_list) {
                backend_.deallocate(pointer, block_bytes);
            }
            free_list.clear();
        }
        for (auto& [pointer, block_bytes] : live_blocks_) {
            backend_.deallocate(pointer, block_bytes);
        }
        live_blocks_.clear();
    }

    Backend backend_;
    std::size_t min_block_bytes_;

    mutable std::mutex mutex_;
    std::unordered_map<std::size_t, std::vector<void*>> free_lists_;
    std::unordered_map<void*, std::size_t> live_blocks_;
    PoolStats stats_;
};

/// Host backend. Exists so the pool's logic can be tested without a GPU, and so
/// pinned-memory staging buffers can reuse the same machinery.
class HostAllocatorBackend {
public:
    [[nodiscard]] void* allocate(std::size_t bytes) const { return std::malloc(bytes); }
    void deallocate(void* pointer, std::size_t /*bytes*/) const noexcept { std::free(pointer); }
    [[nodiscard]] static const char* name() { return "host"; }
};

using HostMemoryPool = MemoryPool<HostAllocatorBackend>;

}  // namespace cudaforge

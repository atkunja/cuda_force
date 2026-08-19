#include "cudaforge/kv_cache.hpp"

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <optional>
#include <set>
#include <thread>
#include <vector>

using cudaforge::BlockAllocator;
using cudaforge::BlockId;
using cudaforge::SequenceBlockTable;

TEST_CASE("the allocator rejects a degenerate configuration", "[kv]") {
    REQUIRE_THROWS_AS(BlockAllocator(0, 16), std::invalid_argument);
    REQUIRE_THROWS_AS(BlockAllocator(16, 0), std::invalid_argument);
}

TEST_CASE("allocation hands out distinct blocks", "[kv]") {
    BlockAllocator allocator(8, 16);
    std::set<BlockId> seen;
    for (int i = 0; i < 8; ++i) {
        const auto block = allocator.allocate();
        REQUIRE(block.has_value());
        // Handing the same block to two sequences would corrupt both, and the
        // damage would look like a model bug rather than an allocator bug.
        REQUIRE(seen.insert(*block).second);
    }
    REQUIRE(allocator.free_blocks() == 0);
}

TEST_CASE("exhaustion returns nullopt rather than throwing", "[kv]") {
    // Running out of cache is expected under load; the scheduler's response is
    // to preempt a sequence, not to unwind.
    BlockAllocator allocator(2, 16);
    REQUIRE(allocator.allocate().has_value());
    REQUIRE(allocator.allocate().has_value());
    REQUIRE_FALSE(allocator.allocate().has_value());
}

TEST_CASE("a released block is reusable", "[kv]") {
    BlockAllocator allocator(1, 16);
    const auto first = allocator.allocate();
    REQUIRE(first.has_value());
    REQUIRE_FALSE(allocator.allocate().has_value());

    allocator.release(*first);
    const auto second = allocator.allocate();
    REQUIRE(second.has_value());
    REQUIRE(*second == *first);
}

TEST_CASE("a shared block survives until its last referent releases it", "[kv][refcount]") {
    // Two sequences sharing a system prompt hold the same physical blocks; the
    // first to finish must not free them out from under the second.
    BlockAllocator allocator(4, 16);
    const auto block = allocator.allocate();
    REQUIRE(block.has_value());
    REQUIRE(allocator.reference_count(*block) == 1);

    allocator.add_reference(*block);
    REQUIRE(allocator.reference_count(*block) == 2);

    allocator.release(*block);
    REQUIRE(allocator.reference_count(*block) == 1);
    REQUIRE(allocator.used_blocks() == 1);

    allocator.release(*block);
    REQUIRE(allocator.reference_count(*block) == 0);
    REQUIRE(allocator.used_blocks() == 0);
}

TEST_CASE("a shared block is not writable", "[kv][refcount]") {
    // Appending to a block another sequence references would corrupt it; the
    // caller must copy-on-write first.
    BlockAllocator allocator(4, 16);
    const auto block = allocator.allocate();
    REQUIRE(allocator.is_writable(*block));

    allocator.add_reference(*block);
    REQUIRE_FALSE(allocator.is_writable(*block));

    allocator.release(*block);
    REQUIRE(allocator.is_writable(*block));
}

TEST_CASE("double free is rejected", "[kv][refcount]") {
    BlockAllocator allocator(4, 16);
    const auto block = allocator.allocate();
    allocator.release(*block);
    REQUIRE_THROWS_AS(allocator.release(*block), std::invalid_argument);
}

TEST_CASE("referencing a free block is rejected", "[kv][refcount]") {
    BlockAllocator allocator(4, 16);
    REQUIRE_THROWS_AS(allocator.add_reference(0), std::invalid_argument);
}

TEST_CASE("unknown block ids are rejected", "[kv]") {
    BlockAllocator allocator(4, 16);
    REQUIRE_THROWS_AS(allocator.release(99), std::out_of_range);
    REQUIRE_THROWS_AS(allocator.add_reference(99), std::out_of_range);
    REQUIRE_THROWS_AS(allocator.reference_count(99), std::out_of_range);
}

TEST_CASE("utilisation tracks the free list", "[kv]") {
    BlockAllocator allocator(4, 16);
    REQUIRE(allocator.utilisation() == 0.0);

    const auto first = allocator.allocate();
    const auto second = allocator.allocate();
    REQUIRE(allocator.utilisation() == 0.5);

    allocator.release(*first);
    allocator.release(*second);
    REQUIRE(allocator.utilisation() == 0.0);
}

// --- block tables ----------------------------------------------------------

TEST_CASE("a new sequence holds nothing", "[kv][table]") {
    const SequenceBlockTable table(1, 16);
    REQUIRE(table.token_count() == 0);
    REQUIRE(table.capacity() == 0);
    REQUIRE(table.needs_block());
}

TEST_CASE("a sequence grows one block at a time", "[kv][table]") {
    BlockAllocator allocator(8, 16);
    SequenceBlockTable table(1, 16);

    for (std::size_t token = 0; token < 40; ++token) {
        if (table.needs_block()) {
            const auto block = allocator.allocate();
            REQUIRE(block.has_value());
            table.append_block(*block);
        }
        table.add_tokens(1);
    }

    REQUIRE(table.token_count() == 40);
    // 40 tokens at 16 per block is three blocks, the last one part-filled.
    REQUIRE(table.blocks().size() == 3);
    REQUIRE(table.capacity() == 48);
}

TEST_CASE("internal fragmentation is under one block", "[kv][table]") {
    // The entire argument for paging: a contiguous cache sized for the longest
    // acceptable sequence wastes the difference, per slot. Here the waste is
    // bounded by the block size.
    constexpr std::size_t kBlockSize = 16;
    BlockAllocator allocator(64, kBlockSize);

    for (std::size_t length : {1U, 15U, 16U, 17U, 100U, 511U}) {
        SequenceBlockTable table(1, kBlockSize);
        for (std::size_t token = 0; token < length; ++token) {
            if (table.needs_block()) {
                const auto block = allocator.allocate();
                REQUIRE(block.has_value());
                table.append_block(*block);
            }
            table.add_tokens(1);
        }
        INFO("length " << length << " slack " << table.slack());
        REQUIRE(table.slack() < kBlockSize);
    }
}

TEST_CASE("adding tokens beyond the allocated blocks is rejected", "[kv][table]") {
    // Overrunning would write into whichever sequence holds the next physical
    // block, so this must fail loudly rather than silently.
    SequenceBlockTable table(1, 16);
    table.append_block(0);
    REQUIRE_NOTHROW(table.add_tokens(16));
    REQUIRE_THROWS_AS(table.add_tokens(1), std::out_of_range);
}

TEST_CASE("a token resolves through the table to a physical block", "[kv][table]") {
    SequenceBlockTable table(1, 4);
    table.append_block(7);
    table.append_block(3);
    table.add_tokens(8);

    // Logical order is 0..7; physical blocks are wherever the allocator had
    // space. That indirection is what removes the contiguity requirement.
    REQUIRE(table.locate(0) == std::pair<BlockId, std::size_t>{7, 0});
    REQUIRE(table.locate(3) == std::pair<BlockId, std::size_t>{7, 3});
    REQUIRE(table.locate(4) == std::pair<BlockId, std::size_t>{3, 0});
    REQUIRE(table.locate(7) == std::pair<BlockId, std::size_t>{3, 3});
}

TEST_CASE("locating past the end is rejected", "[kv][table]") {
    SequenceBlockTable table(1, 4);
    table.append_block(0);
    table.add_tokens(2);
    REQUIRE_THROWS_AS(table.locate(2), std::out_of_range);
}

TEST_CASE("copy-on-write replaces a shared block in place", "[kv][table]") {
    BlockAllocator allocator(8, 4);
    SequenceBlockTable shared(1, 4);
    SequenceBlockTable forked(2, 4);

    const auto prefix = allocator.allocate();
    REQUIRE(prefix.has_value());
    shared.append_block(*prefix);
    shared.add_tokens(4);

    // The fork shares the prefix rather than copying it.
    allocator.add_reference(*prefix);
    forked.append_block(*prefix);
    forked.add_tokens(4);
    REQUIRE(allocator.used_blocks() == 1);
    REQUIRE_FALSE(allocator.is_writable(*prefix));

    // Extending the fork requires its own copy of the shared block.
    const auto copy = allocator.allocate();
    REQUIRE(copy.has_value());
    forked.replace_block(0, *copy);
    allocator.release(*prefix);

    REQUIRE(shared.blocks()[0] == *prefix);
    REQUIRE(forked.blocks()[0] == *copy);
    REQUIRE(allocator.is_writable(*prefix));
    REQUIRE(allocator.is_writable(*copy));
}

TEST_CASE("replacing an unheld block is rejected", "[kv][table]") {
    SequenceBlockTable table(1, 4);
    REQUIRE_THROWS_AS(table.replace_block(0, 1), std::out_of_range);
}

TEST_CASE("concurrent allocation never double-issues a block", "[kv][stress]") {
    constexpr std::size_t kBlocks = 512;
    constexpr int kThreads = 8;

    BlockAllocator allocator(kBlocks, 16);
    std::vector<std::vector<BlockId>> claimed(kThreads);
    std::vector<std::thread> threads;

    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            while (true) {
                const auto block = allocator.allocate();
                if (!block) {
                    return;
                }
                claimed[static_cast<std::size_t>(t)].push_back(*block);
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }

    std::set<BlockId> all;
    std::size_t total = 0;
    for (const auto& per_thread : claimed) {
        total += per_thread.size();
        for (BlockId block : per_thread) {
            REQUIRE(all.insert(block).second);
        }
    }

    REQUIRE(total == kBlocks);
    REQUIRE(all.size() == kBlocks);
    REQUIRE(allocator.free_blocks() == 0);
}

TEST_CASE("concurrent allocate and release conserves every block", "[kv][stress]") {
    constexpr std::size_t kBlocks = 128;
    constexpr int kThreads = 8;
    constexpr int kRounds = 500;

    BlockAllocator allocator(kBlocks, 16);
    std::atomic<int> exhausted{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&] {
            for (int i = 0; i < kRounds; ++i) {
                const auto block = allocator.allocate();
                if (!block) {
                    exhausted.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                allocator.release(*block);
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }

    // Every block taken was given back; none leaked and none was double-freed.
    REQUIRE(allocator.free_blocks() == kBlocks);
    REQUIRE(allocator.used_blocks() == 0);
}

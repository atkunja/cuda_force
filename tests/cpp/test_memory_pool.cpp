#include "cudaforge/memory_pool.hpp"

#include "thread_assert.hpp"

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

using cudaforge::HostAllocatorBackend;
using cudaforge::HostMemoryPool;
using cudaforge::MemoryPool;
using cudaforge::test::ThreadAssert;

namespace {

/// Counts backend traffic so a test can prove the pool served a request from
/// its free list rather than going back to the allocator.
class CountingBackend {
public:
    struct Counters {
        std::atomic<std::uint64_t> allocations{0};
        std::atomic<std::uint64_t> deallocations{0};
        std::atomic<std::size_t> live_bytes{0};
    };

    explicit CountingBackend(Counters* counters) : counters_(counters) {}

    [[nodiscard]] void* allocate(std::size_t bytes) const {
        counters_->allocations.fetch_add(1);
        counters_->live_bytes.fetch_add(bytes);
        return std::malloc(bytes);
    }

    void deallocate(void* pointer, std::size_t bytes) const noexcept {
        counters_->deallocations.fetch_add(1);
        counters_->live_bytes.fetch_sub(bytes);
        std::free(pointer);
    }

    [[nodiscard]] static const char* name() { return "counting"; }

private:
    Counters* counters_;
};

}  // namespace

TEST_CASE("a zero-byte request yields nullptr", "[pool]") {
    HostMemoryPool pool;
    REQUIRE(pool.allocate(0) == nullptr);
    REQUIRE_NOTHROW(pool.deallocate(nullptr));
}

TEST_CASE("allocated memory is writable for the requested size", "[pool]") {
    HostMemoryPool pool;
    constexpr std::size_t kBytes = 4096;
    auto* buffer = static_cast<unsigned char*>(pool.allocate(kBytes));
    REQUIRE(buffer != nullptr);

    std::memset(buffer, 0xAB, kBytes);
    REQUIRE(buffer[0] == 0xAB);
    REQUIRE(buffer[kBytes - 1] == 0xAB);
    pool.deallocate(buffer);
}

TEST_CASE("a freed block is reused rather than reallocated", "[pool]") {
    CountingBackend::Counters counters;
    MemoryPool<CountingBackend> pool{CountingBackend{&counters}};

    void* first = pool.allocate(1024);
    pool.deallocate(first);
    void* second = pool.allocate(1024);

    REQUIRE(second == first);
    REQUIRE(counters.allocations.load() == 1);
    REQUIRE(pool.stats().reuse_count == 1);
    REQUIRE(pool.stats().backend_allocations == 1);

    pool.deallocate(second);
}

TEST_CASE("requests round up to a shared size class", "[pool]") {
    CountingBackend::Counters counters;
    MemoryPool<CountingBackend> pool{CountingBackend{&counters}};

    // 1100 and 2000 both round to the 2048-byte class, so the second call
    // reuses the first block. This is the fragmentation/reuse tradeoff the
    // size classes buy: up to 2x slack in exchange for exact reuse.
    void* first = pool.allocate(1100);
    pool.deallocate(first);
    void* second = pool.allocate(2000);

    REQUIRE(second == first);
    REQUIRE(counters.allocations.load() == 1);
    pool.deallocate(second);
}

TEST_CASE("different size classes do not share blocks", "[pool]") {
    CountingBackend::Counters counters;
    MemoryPool<CountingBackend> pool{CountingBackend{&counters}};

    void* small = pool.allocate(1024);
    pool.deallocate(small);
    void* large = pool.allocate(1024 * 1024);

    REQUIRE(large != small);
    REQUIRE(counters.allocations.load() == 2);
    pool.deallocate(large);
}

TEST_CASE("the minimum block size floors small requests", "[pool]") {
    CountingBackend::Counters counters;
    MemoryPool<CountingBackend> pool{CountingBackend{&counters}, /*min_block_bytes=*/512};

    void* first = pool.allocate(8);
    pool.deallocate(first);
    void* second = pool.allocate(300);

    REQUIRE(second == first);  // both floored to the 512-byte class
    REQUIRE(counters.allocations.load() == 1);
    pool.deallocate(second);
}

TEST_CASE("statistics track in-use and peak bytes", "[pool]") {
    HostMemoryPool pool;
    std::vector<void*> blocks;
    for (int i = 0; i < 4; ++i) {
        blocks.push_back(pool.allocate(1024));
    }

    const auto peak = pool.stats();
    REQUIRE(peak.bytes_in_use == 4 * 1024);
    REQUIRE(peak.peak_bytes_in_use == 4 * 1024);
    REQUIRE(peak.allocation_count == 4);

    for (void* block : blocks) {
        pool.deallocate(block);
    }

    const auto after = pool.stats();
    REQUIRE(after.bytes_in_use == 0);
    REQUIRE(after.peak_bytes_in_use == 4 * 1024);  // peak is a high-water mark
    REQUIRE(after.free_block_count == 4);
    REQUIRE(after.bytes_reserved == 4 * 1024);  // still held for reuse
}

TEST_CASE("reuse rate rises once the working set is warm", "[pool]") {
    HostMemoryPool pool;
    for (int round = 0; round < 100; ++round) {
        void* block = pool.allocate(8192);
        pool.deallocate(block);
    }
    const auto stats = pool.stats();
    REQUIRE(stats.allocation_count == 100);
    REQUIRE(stats.backend_allocations == 1);
    REQUIRE(stats.reuse_rate() > 0.98);
}

TEST_CASE("trim returns cached blocks to the backend", "[pool]") {
    CountingBackend::Counters counters;
    MemoryPool<CountingBackend> pool{CountingBackend{&counters}};

    void* block = pool.allocate(4096);
    pool.deallocate(block);
    REQUIRE(pool.stats().free_block_count == 1);

    pool.trim();
    REQUIRE(pool.stats().free_block_count == 0);
    REQUIRE(pool.stats().bytes_reserved == 0);
    REQUIRE(counters.deallocations.load() == 1);
}

TEST_CASE("trim leaves checked-out blocks alone", "[pool]") {
    CountingBackend::Counters counters;
    MemoryPool<CountingBackend> pool{CountingBackend{&counters}};

    void* held = pool.allocate(4096);
    pool.trim();
    REQUIRE(counters.deallocations.load() == 0);

    std::memset(held, 0, 4096);  // still valid
    pool.deallocate(held);
}

TEST_CASE("deallocating a foreign pointer is rejected", "[pool]") {
    HostMemoryPool pool;
    int stack_value = 0;
    REQUIRE_THROWS_AS(pool.deallocate(&stack_value), std::invalid_argument);
}

TEST_CASE("the destructor releases everything to the backend", "[pool]") {
    CountingBackend::Counters counters;
    {
        MemoryPool<CountingBackend> pool{CountingBackend{&counters}};
        void* freed = pool.allocate(1024);
        pool.deallocate(freed);
        (void)pool.allocate(2048);  // deliberately still checked out
    }
    REQUIRE(counters.live_bytes.load() == 0);
    REQUIRE(counters.allocations.load() == counters.deallocations.load());
}

TEST_CASE("concurrent allocation and release is safe", "[pool][stress]") {
    constexpr int kThreads = 8;
    constexpr int kPerThread = 500;

    HostMemoryPool pool;
    ThreadAssert errors;
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < kPerThread; ++i) {
                const std::size_t bytes = 256UL << ((t + i) % 5);
                void* block = pool.allocate(bytes);
                if (!errors.check(block != nullptr)) {
                    continue;
                }
                std::memset(block, t, bytes);
                pool.deallocate(block);
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }

    REQUIRE(errors.failures() == 0);
    const auto stats = pool.stats();
    REQUIRE(stats.allocation_count == kThreads * kPerThread);
    REQUIRE(stats.free_count == kThreads * kPerThread);
    REQUIRE(stats.bytes_in_use == 0);
}

TEST_CASE("the backend name is exposed for reporting", "[pool]") {
    REQUIRE(std::string(HostMemoryPool::backend_name()) == "host");
    REQUIRE(std::string(HostAllocatorBackend::name()) == "host");
}

// --- allocation failure ----------------------------------------------------

namespace {

/// Backend that refuses allocations after a configured number of successes.
/// Device OOM is the realistic case: `cudaMalloc` returns an error rather than
/// throwing, and the pool has to turn that into something callers can act on.
class FailingBackend {
public:
    struct State {
        std::atomic<int> allocations_before_failure{0};
        std::atomic<std::uint64_t> deallocations{0};
    };

    explicit FailingBackend(State* state) : state_(state) {}

    [[nodiscard]] void* allocate(std::size_t bytes) const {
        if (state_->allocations_before_failure.fetch_sub(1) <= 0) {
            return nullptr;  // as cudaMalloc failure is surfaced
        }
        return std::malloc(bytes);
    }

    void deallocate(void* pointer, std::size_t /*bytes*/) const noexcept {
        state_->deallocations.fetch_add(1);
        std::free(pointer);
    }

    [[nodiscard]] static const char* name() { return "failing"; }

private:
    State* state_;
};

}  // namespace

TEST_CASE("a backend failure surfaces as bad_alloc", "[pool][failure]") {
    // Returning null to the caller would be worse than throwing: a null device
    // pointer passed to a kernel is an illegal access thousands of lines later.
    FailingBackend::State state;
    state.allocations_before_failure = 0;
    MemoryPool<FailingBackend> pool{FailingBackend{&state}};

    REQUIRE_THROWS_AS(pool.allocate(1024), std::bad_alloc);
}

TEST_CASE("the pool is usable after a failed allocation", "[pool][failure]") {
    FailingBackend::State state;
    state.allocations_before_failure = 0;
    MemoryPool<FailingBackend> pool{FailingBackend{&state}};

    REQUIRE_THROWS_AS(pool.allocate(1024), std::bad_alloc);

    // The failure must not have left the mutex held or the accounting skewed.
    state.allocations_before_failure = 1;
    void* block = pool.allocate(1024);
    REQUIRE(block != nullptr);
    REQUIRE(pool.stats().bytes_in_use == 1024);
    pool.deallocate(block);
    REQUIRE(pool.stats().bytes_in_use == 0);
}

TEST_CASE("a failed allocation does not inflate the statistics", "[pool][failure]") {
    FailingBackend::State state;
    state.allocations_before_failure = 0;
    MemoryPool<FailingBackend> pool{FailingBackend{&state}};

    REQUIRE_THROWS_AS(pool.allocate(4096), std::bad_alloc);

    const auto stats = pool.stats();
    REQUIRE(stats.bytes_reserved == 0);
    REQUIRE(stats.bytes_in_use == 0);
    REQUIRE(stats.backend_allocations == 0);
    // The attempt is still counted; it happened.
    REQUIRE(stats.allocation_count == 1);
}

TEST_CASE("trimming then retrying is a viable recovery", "[pool][failure]") {
    // The documented response to device OOM: release cached blocks and retry.
    FailingBackend::State state;
    state.allocations_before_failure = 1;
    MemoryPool<FailingBackend> pool{FailingBackend{&state}};

    void* first = pool.allocate(4096);
    REQUIRE(first != nullptr);
    pool.deallocate(first);

    // A different size class, so the free list cannot satisfy it.
    REQUIRE_THROWS_AS(pool.allocate(1 << 20), std::bad_alloc);

    pool.trim();
    REQUIRE(state.deallocations.load() == 1);
    REQUIRE(pool.stats().bytes_reserved == 0);
}

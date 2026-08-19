#pragma once

#include <atomic>
#include <cstdint>

namespace cudaforge::test {

/// Records assertion failures from worker threads.
///
/// Catch2's `REQUIRE` family is not thread-safe: the macros manipulate
/// per-run state including an output redirect that asserts if two threads
/// activate it at once. Calling them off the main thread produces an abort
/// whose timing depends on the scheduler, so the resulting failure looks
/// intermittent and unrelated to the code under test.
///
/// Concurrency tests therefore funnel their checks through this counter and
/// assert once, on the main thread, after every worker has been joined.
class ThreadAssert {
public:
    /// Returns the condition so it can be used inline in a worker loop.
    bool check(bool condition) noexcept {
        if (!condition) {
            failures_.fetch_add(1, std::memory_order_relaxed);
        }
        return condition;
    }

    [[nodiscard]] std::uint64_t failures() const noexcept {
        return failures_.load(std::memory_order_relaxed);
    }

private:
    std::atomic<std::uint64_t> failures_{0};
};

}  // namespace cudaforge::test

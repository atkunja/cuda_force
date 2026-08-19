#include "cudaforge/thread_pool.hpp"

#include "thread_assert.hpp"

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

using cudaforge::ThreadPool;
using cudaforge::test::ThreadAssert;
using namespace std::chrono_literals;

TEST_CASE("pool rejects a zero worker count", "[pool]") {
    REQUIRE_THROWS_AS(ThreadPool(0), std::invalid_argument);
}

TEST_CASE("pool rejects an empty task", "[pool]") {
    ThreadPool pool(2);
    REQUIRE_THROWS_AS(pool.submit(ThreadPool::Task{}), std::invalid_argument);
}

TEST_CASE("every submitted task runs", "[pool]") {
    constexpr int kTasks = 1000;
    ThreadPool pool(4);
    std::atomic<int> counter{0};

    for (int i = 0; i < kTasks; ++i) {
        REQUIRE(pool.submit([&counter] { counter.fetch_add(1, std::memory_order_relaxed); }));
    }
    pool.shutdown();

    REQUIRE(counter.load() == kTasks);
    REQUIRE(pool.stats().completed == kTasks);
}

TEST_CASE("futures carry return values", "[pool]") {
    ThreadPool pool(3);
    std::vector<std::future<int>> futures;
    futures.reserve(50);

    for (int i = 0; i < 50; ++i) {
        futures.push_back(pool.submit_with_result([](int value) { return value * value; }, i));
    }

    int total = 0;
    for (std::future<int>& future : futures) {
        total += future.get();
    }
    REQUIRE(total == 40425);  // sum of i^2 for i in [0, 50)
}

TEST_CASE("futures propagate exceptions to the caller", "[pool]") {
    ThreadPool pool(2);
    auto future = pool.submit_with_result([]() -> int { throw std::runtime_error("boom"); });
    REQUIRE_THROWS_AS(future.get(), std::runtime_error);
}

TEST_CASE("a throwing task does not kill its worker", "[pool]") {
    ThreadPool pool(2);
    for (int i = 0; i < 20; ++i) {
        REQUIRE(pool.submit([] { throw std::runtime_error("expected"); }));
    }

    std::atomic<int> survivors{0};
    for (int i = 0; i < 20; ++i) {
        REQUIRE(pool.submit([&survivors] { survivors.fetch_add(1); }));
    }
    pool.shutdown();

    REQUIRE(survivors.load() == 20);
    REQUIRE(pool.stats().failed == 20);
    REQUIRE(pool.stats().completed == 20);
}

TEST_CASE("shutdown drains work already accepted", "[pool][shutdown]") {
    constexpr int kTasks = 200;
    std::atomic<int> counter{0};
    {
        ThreadPool pool(4, /*queue_capacity=*/kTasks);
        for (int i = 0; i < kTasks; ++i) {
            REQUIRE(pool.submit([&counter] {
                std::this_thread::sleep_for(100us);
                counter.fetch_add(1, std::memory_order_relaxed);
            }));
        }
    }  // destructor joins
    REQUIRE(counter.load() == kTasks);
}

TEST_CASE("shutdown is idempotent", "[pool][shutdown]") {
    ThreadPool pool(2);
    pool.shutdown();
    pool.shutdown();
    REQUIRE_FALSE(pool.submit([] {}));
}

TEST_CASE("submit_with_result on a stopped pool throws", "[pool][shutdown]") {
    ThreadPool pool(2);
    pool.shutdown();
    REQUIRE_THROWS_AS(pool.submit_with_result([] { return 1; }), std::runtime_error);
}

TEST_CASE("many producers submit concurrently without loss", "[pool][stress]") {
    constexpr int kProducers = 8;
    constexpr int kPerProducer = 500;

    std::atomic<int> counter{0};
    ThreadAssert errors;
    {
        ThreadPool pool(6, /*queue_capacity=*/64);
        std::vector<std::thread> producers;
        producers.reserve(kProducers);
        for (int p = 0; p < kProducers; ++p) {
            producers.emplace_back([&] {
                for (int i = 0; i < kPerProducer; ++i) {
                    errors.check(pool.submit([&counter] { counter.fetch_add(1); }));
                }
            });
        }
        for (std::thread& producer : producers) {
            producer.join();
        }
    }
    REQUIRE(errors.failures() == 0);
    REQUIRE(counter.load() == kProducers * kPerProducer);
}

TEST_CASE("a bounded queue applies backpressure rather than growing", "[pool][stress]") {
    // The pool is deliberately slower than the producer. With a bounded queue,
    // submit() blocks and depth stays capped; without one, depth would grow
    // until the producer finished.
    constexpr std::size_t kCapacity = 8;
    ThreadPool pool(1, kCapacity);

    std::atomic<std::size_t> max_depth{0};
    std::thread watcher([&] {
        for (int i = 0; i < 400; ++i) {
            const std::size_t depth = pool.stats().queue_depth;
            if (depth > max_depth.load()) {
                max_depth.store(depth);
            }
            std::this_thread::sleep_for(200us);
        }
    });

    for (int i = 0; i < 200; ++i) {
        REQUIRE(pool.submit([] { std::this_thread::sleep_for(200us); }));
    }
    pool.shutdown();
    watcher.join();

    REQUIRE(max_depth.load() <= kCapacity);
}

TEST_CASE("stats report the configured worker count", "[pool]") {
    ThreadPool pool(5);
    REQUIRE(pool.worker_count() == 5);
    REQUIRE(pool.stats().worker_count == 5);
}

// --- statistics ------------------------------------------------------------

TEST_CASE("submitted counts only accepted tasks", "[pool][stats]") {
    ThreadPool pool(2, /*queue_capacity=*/8);
    for (int i = 0; i < 20; ++i) {
        REQUIRE(pool.submit([] {}));
    }
    pool.shutdown();

    const auto stats = pool.stats();
    REQUIRE(stats.submitted == 20);
    REQUIRE(stats.completed == 20);
    REQUIRE(stats.failed == 0);
    // A rejected task is never submitted, so the counter must not move.
    REQUIRE_FALSE(pool.submit([] {}));
    REQUIRE(pool.stats().submitted == 20);
}

TEST_CASE("completed and failed partition the submitted tasks", "[pool][stats]") {
    ThreadPool pool(3);
    for (int i = 0; i < 30; ++i) {
        const bool should_throw = i % 3 == 0;
        REQUIRE(pool.submit([should_throw] {
            if (should_throw) {
                throw std::runtime_error("expected");
            }
        }));
    }
    pool.shutdown();

    const auto stats = pool.stats();
    REQUIRE(stats.submitted == 30);
    REQUIRE(stats.failed == 10);
    REQUIRE(stats.completed == 20);
    REQUIRE(stats.completed + stats.failed == stats.submitted);
}

TEST_CASE("the queue drains to empty after shutdown", "[pool][stats]") {
    ThreadPool pool(2, /*queue_capacity=*/64);
    for (int i = 0; i < 50; ++i) {
        REQUIRE(pool.submit([] {}));
    }
    pool.shutdown();
    REQUIRE(pool.stats().queue_depth == 0);
}

TEST_CASE("active workers return to zero when idle", "[pool][stats]") {
    ThreadPool pool(4);
    for (int i = 0; i < 20; ++i) {
        REQUIRE(pool.submit([] { std::this_thread::sleep_for(100us); }));
    }
    pool.shutdown();
    REQUIRE(pool.stats().active_workers == 0);
}

TEST_CASE("futures carry arguments through by value", "[pool]") {
    // The packaged task stores its arguments in a tuple; a dangling reference
    // here would be a use-after-free once the caller's frame returned.
    ThreadPool pool(2);
    auto future = pool.submit_with_result(
        [](std::string prefix, int value) { return prefix + std::to_string(value); },
        std::string("answer-"), 42);
    REQUIRE(future.get() == "answer-42");
}

TEST_CASE("a void-returning future is still waitable", "[pool]") {
    ThreadPool pool(2);
    std::atomic<bool> ran{false};
    auto future = pool.submit_with_result([&ran] { ran = true; });
    future.get();
    REQUIRE(ran.load());
}

TEST_CASE("many small tasks retire without loss", "[pool][stress]") {
    // A task count well above the queue capacity, so producers block and the
    // pool cycles the queue many times.
    constexpr int kTasks = 20'000;
    std::atomic<int> counter{0};
    {
        ThreadPool pool(4, /*queue_capacity=*/32);
        for (int i = 0; i < kTasks; ++i) {
            REQUIRE(pool.submit([&counter] { counter.fetch_add(1, std::memory_order_relaxed); }));
        }
    }
    REQUIRE(counter.load() == kTasks);
}

#include "cudaforge/thread_pool.hpp"

#include "thread_assert.hpp"

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

using cudaforge::test::ThreadAssert;
using cudaforge::ThreadPool;
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

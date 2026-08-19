#include "cudaforge/concurrent_queue.hpp"

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>

using cudaforge::ConcurrentQueue;
using cudaforge::QueueStatus;
using namespace std::chrono_literals;

TEST_CASE("queue rejects a zero capacity", "[queue]") {
    REQUIRE_THROWS_AS(ConcurrentQueue<int>(0), std::invalid_argument);
}

TEST_CASE("queue preserves fifo order on a single thread", "[queue]") {
    ConcurrentQueue<int> queue(8);
    for (int i = 0; i < 5; ++i) {
        REQUIRE(queue.push(i) == QueueStatus::Ok);
    }
    REQUIRE(queue.size() == 5);

    for (int expected = 0; expected < 5; ++expected) {
        int value = -1;
        REQUIRE(queue.pop(value) == QueueStatus::Ok);
        REQUIRE(value == expected);
    }
    REQUIRE(queue.empty());
}

TEST_CASE("try_push reports Full instead of blocking", "[queue]") {
    ConcurrentQueue<int> queue(2);
    REQUIRE(queue.try_push(1) == QueueStatus::Ok);
    REQUIRE(queue.try_push(2) == QueueStatus::Ok);
    REQUIRE(queue.try_push(3) == QueueStatus::Full);
    REQUIRE(queue.size() == 2);
}

TEST_CASE("try_pop reports Empty on an open empty queue", "[queue]") {
    ConcurrentQueue<int> queue(2);
    int value = 0;
    REQUIRE(queue.try_pop(value) == QueueStatus::Empty);
}

TEST_CASE("pop_for times out rather than waiting forever", "[queue]") {
    ConcurrentQueue<int> queue(4);
    int value = 0;

    const auto start = std::chrono::steady_clock::now();
    REQUIRE(queue.pop_for(value, 30ms) == QueueStatus::Timeout);
    const auto elapsed = std::chrono::steady_clock::now() - start;

    // Asserts only the lower bound. An upper bound would make the test flaky on
    // a loaded CI machine, where the scheduler can delay the wakeup arbitrarily.
    REQUIRE(elapsed >= 25ms);
}

TEST_CASE("a blocked producer is released by shutdown", "[queue][shutdown]") {
    ConcurrentQueue<int> queue(1);
    REQUIRE(queue.push(1) == QueueStatus::Ok);

    std::atomic<QueueStatus> observed{QueueStatus::Ok};
    std::thread producer([&] { observed = queue.push(2); });

    std::this_thread::sleep_for(20ms);  // let the producer reach the wait
    queue.shutdown();
    producer.join();

    REQUIRE(observed.load() == QueueStatus::Closed);
}

TEST_CASE("a blocked consumer is released by shutdown", "[queue][shutdown]") {
    ConcurrentQueue<int> queue(4);

    std::atomic<QueueStatus> observed{QueueStatus::Ok};
    std::thread consumer([&] {
        int value = 0;
        observed = queue.pop(value);
    });

    std::this_thread::sleep_for(20ms);
    queue.shutdown();
    consumer.join();

    REQUIRE(observed.load() == QueueStatus::Closed);
}

TEST_CASE("shutdown drains buffered items before reporting Closed", "[queue][shutdown]") {
    ConcurrentQueue<int> queue(4);
    REQUIRE(queue.push(10) == QueueStatus::Ok);
    REQUIRE(queue.push(20) == QueueStatus::Ok);
    queue.shutdown();

    int value = 0;
    REQUIRE(queue.pop(value) == QueueStatus::Ok);
    REQUIRE(value == 10);
    REQUIRE(queue.pop(value) == QueueStatus::Ok);
    REQUIRE(value == 20);
    REQUIRE(queue.pop(value) == QueueStatus::Closed);
}

TEST_CASE("shutdown is idempotent", "[queue][shutdown]") {
    ConcurrentQueue<int> queue(2);
    queue.shutdown();
    queue.shutdown();
    REQUIRE(queue.closed());
    REQUIRE(queue.push(1) == QueueStatus::Closed);
}

TEST_CASE("a closed queue rejects new work", "[queue][shutdown]") {
    ConcurrentQueue<int> queue(4);
    queue.shutdown();
    REQUIRE(queue.push(1) == QueueStatus::Closed);
    REQUIRE(queue.try_push(1) == QueueStatus::Closed);
}

TEST_CASE("the queue holds move-only payloads", "[queue]") {
    ConcurrentQueue<std::unique_ptr<int>> queue(4);
    REQUIRE(queue.push(std::make_unique<int>(42)) == QueueStatus::Ok);

    std::unique_ptr<int> value;
    REQUIRE(queue.pop(value) == QueueStatus::Ok);
    REQUIRE(value != nullptr);
    REQUIRE(*value == 42);
}

TEST_CASE("capacity is never exceeded under concurrent producers", "[queue][stress]") {
    constexpr std::size_t kCapacity = 4;
    constexpr int kProducers = 8;
    constexpr int kPerProducer = 500;

    ConcurrentQueue<int> queue(kCapacity);
    std::atomic<std::size_t> max_observed{0};
    std::atomic<bool> stop{false};

    // Independent observer rather than an assert inside the queue: the point is
    // that an outside thread can never catch the queue over capacity.
    std::thread watcher([&] {
        while (!stop.load(std::memory_order_relaxed)) {
            std::size_t depth = queue.size();
            std::size_t previous = max_observed.load(std::memory_order_relaxed);
            while (depth > previous &&
                   !max_observed.compare_exchange_weak(previous, depth,
                                                       std::memory_order_relaxed)) {
            }
        }
    });

    std::vector<std::thread> producers;
    producers.reserve(kProducers);
    for (int p = 0; p < kProducers; ++p) {
        producers.emplace_back([&] {
            for (int i = 0; i < kPerProducer; ++i) {
                REQUIRE(queue.push(i) == QueueStatus::Ok);
            }
        });
    }

    std::atomic<int> consumed{0};
    std::vector<std::thread> consumers;
    consumers.reserve(4);
    for (int cnt = 0; cnt < 4; ++cnt) {
        consumers.emplace_back([&] {
            int value = 0;
            while (queue.pop(value) == QueueStatus::Ok) {
                consumed.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    for (std::thread& producer : producers) {
        producer.join();
    }
    queue.shutdown();
    for (std::thread& consumer : consumers) {
        consumer.join();
    }
    stop = true;
    watcher.join();

    REQUIRE(consumed.load() == kProducers * kPerProducer);
    REQUIRE(max_observed.load() <= kCapacity);
}

TEST_CASE("every produced item is consumed exactly once", "[queue][stress]") {
    constexpr int kProducers = 6;
    constexpr int kPerProducer = 400;
    constexpr int kTotal = kProducers * kPerProducer;

    ConcurrentQueue<int> queue(16);
    std::vector<std::atomic<int>> seen(kTotal);
    for (std::atomic<int>& slot : seen) {
        slot.store(0);
    }

    std::vector<std::thread> producers;
    producers.reserve(kProducers);
    for (int p = 0; p < kProducers; ++p) {
        producers.emplace_back([&, p] {
            for (int i = 0; i < kPerProducer; ++i) {
                REQUIRE(queue.push(p * kPerProducer + i) == QueueStatus::Ok);
            }
        });
    }

    std::vector<std::thread> consumers;
    consumers.reserve(5);
    for (int cnt = 0; cnt < 5; ++cnt) {
        consumers.emplace_back([&] {
            int value = 0;
            while (queue.pop(value) == QueueStatus::Ok) {
                seen[static_cast<std::size_t>(value)].fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    for (std::thread& producer : producers) {
        producer.join();
    }
    queue.shutdown();
    for (std::thread& consumer : consumers) {
        consumer.join();
    }

    const int total = std::accumulate(seen.begin(), seen.end(), 0,
                                      [](int acc, const std::atomic<int>& slot) {
                                          return acc + slot.load();
                                      });
    REQUIRE(total == kTotal);
    for (const std::atomic<int>& slot : seen) {
        REQUIRE(slot.load() == 1);
    }
}

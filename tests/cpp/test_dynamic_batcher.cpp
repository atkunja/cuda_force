#include "cudaforge/dynamic_batcher.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

using cudaforge::Batch;
using cudaforge::BatchTrigger;
using cudaforge::DynamicBatcher;
using cudaforge::Metrics;
using cudaforge::Request;
using cudaforge::RuntimeConfig;
using namespace std::chrono_literals;

namespace {

/// Collects the batches a batcher produces so a test can assert on their sizes
/// and triggers after shutdown, when the batcher thread is joined and no
/// synchronisation is needed to read the vector.
class BatchCollector {
public:
    void operator()(Batch&& batch) {
        std::lock_guard lock(mutex_);
        sizes_.push_back(batch.size());
        triggers_.push_back(batch.trigger);
        total_ += batch.size();
    }

    [[nodiscard]] std::vector<std::size_t> sizes() const {
        std::lock_guard lock(mutex_);
        return sizes_;
    }
    [[nodiscard]] std::vector<BatchTrigger> triggers() const {
        std::lock_guard lock(mutex_);
        return triggers_;
    }
    [[nodiscard]] std::size_t total() const {
        std::lock_guard lock(mutex_);
        return total_;
    }

private:
    mutable std::mutex mutex_;
    std::vector<std::size_t> sizes_;
    std::vector<BatchTrigger> triggers_;
    std::size_t total_ = 0;
};

Request make_request(std::uint64_t id) { return Request(id, "prompt", {}); }

RuntimeConfig test_config(std::size_t max_batch, std::chrono::microseconds wait) {
    RuntimeConfig config;
    config.max_batch_size = max_batch;
    config.max_wait = wait;
    // The queue must be able to hold a full batch, so it is derived from the
    // batch size rather than fixed. A fixed 256 silently broke the large-batch
    // cases below.
    config.queue_capacity = std::max<std::size_t>(256, max_batch * 2);
    return config;
}

}  // namespace

TEST_CASE("the batcher requires a handler and metrics", "[batcher]") {
    auto metrics = std::make_shared<Metrics>();
    REQUIRE_THROWS_AS(DynamicBatcher(RuntimeConfig{}, nullptr, metrics), std::invalid_argument);
    REQUIRE_THROWS_AS(
        DynamicBatcher(RuntimeConfig{}, [](Batch&&) {}, nullptr), std::invalid_argument);
}

TEST_CASE("an invalid config is rejected at construction", "[batcher]") {
    RuntimeConfig config;
    config.max_batch_size = 0;
    auto metrics = std::make_shared<Metrics>();
    REQUIRE_THROWS_AS(DynamicBatcher(config, [](Batch&&) {}, metrics), std::invalid_argument);
}

TEST_CASE("a lone request is released after max_wait", "[batcher]") {
    auto collector = std::make_shared<BatchCollector>();
    auto metrics = std::make_shared<Metrics>();

    {
        DynamicBatcher batcher(test_config(16, 20ms),
                               [collector](Batch&& batch) { (*collector)(std::move(batch)); },
                               metrics);
        REQUIRE(batcher.submit(make_request(1)));
        std::this_thread::sleep_for(120ms);
    }

    const auto sizes = collector->sizes();
    REQUIRE(sizes.size() == 1);
    REQUIRE(sizes[0] == 1);
    REQUIRE(collector->triggers()[0] == BatchTrigger::Timeout);
}

TEST_CASE("a full batch is released without waiting", "[batcher]") {
    auto collector = std::make_shared<BatchCollector>();
    auto metrics = std::make_shared<Metrics>();

    // A long wait makes the assertion meaningful: if the batch were not closed
    // on size, this test would take five seconds instead of milliseconds.
    const auto start = std::chrono::steady_clock::now();
    {
        DynamicBatcher batcher(test_config(4, 5s),
                               [collector](Batch&& batch) { (*collector)(std::move(batch)); },
                               metrics);
        for (std::uint64_t i = 0; i < 4; ++i) {
            REQUIRE(batcher.submit(make_request(i)));
        }
        while (collector->total() < 4) {
            std::this_thread::sleep_for(1ms);
        }
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;

    REQUIRE(collector->sizes() == std::vector<std::size_t>{4});
    REQUIRE(collector->triggers()[0] == BatchTrigger::MaxSize);
    REQUIRE(elapsed < 4s);
}

TEST_CASE("no batch exceeds max_batch_size", "[batcher]") {
    constexpr std::size_t kMaxBatch = 8;
    constexpr std::uint64_t kRequests = 200;

    auto collector = std::make_shared<BatchCollector>();
    auto metrics = std::make_shared<Metrics>();
    {
        DynamicBatcher batcher(test_config(kMaxBatch, 2ms),
                               [collector](Batch&& batch) { (*collector)(std::move(batch)); },
                               metrics);
        for (std::uint64_t i = 0; i < kRequests; ++i) {
            REQUIRE(batcher.submit(make_request(i)));
        }
    }

    REQUIRE(collector->total() == kRequests);
    for (std::size_t size : collector->sizes()) {
        REQUIRE(size >= 1);
        REQUIRE(size <= kMaxBatch);
    }
}

TEST_CASE("the deadline is anchored to the oldest request", "[batcher]") {
    // A steady trickle of arrivals must not postpone execution indefinitely. If
    // the deadline were reset on every arrival, this batch would never close
    // while the producer keeps trickling.
    auto collector = std::make_shared<BatchCollector>();
    auto metrics = std::make_shared<Metrics>();

    {
        DynamicBatcher batcher(test_config(1000, 30ms),
                               [collector](Batch&& batch) { (*collector)(std::move(batch)); },
                               metrics);
        std::atomic<bool> stop{false};
        std::thread trickle([&] {
            std::uint64_t id = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                batcher.submit(make_request(id++));
                std::this_thread::sleep_for(2ms);
            }
        });

        std::this_thread::sleep_for(250ms);
        stop = true;
        trickle.join();
    }

    // Roughly 250ms of arrivals with a 30ms deadline: several closures, none of
    // which could have happened if the deadline slid forward on each arrival.
    REQUIRE(collector->sizes().size() >= 3);
    REQUIRE(collector->total() > 0);
}

TEST_CASE("shutdown drains queued requests", "[batcher][shutdown]") {
    constexpr std::uint64_t kRequests = 50;
    auto collector = std::make_shared<BatchCollector>();
    auto metrics = std::make_shared<Metrics>();

    {
        DynamicBatcher batcher(test_config(8, 500ms),
                               [collector](Batch&& batch) { (*collector)(std::move(batch)); },
                               metrics);
        for (std::uint64_t i = 0; i < kRequests; ++i) {
            REQUIRE(batcher.submit(make_request(i)));
        }
    }  // destructor shuts down and joins

    REQUIRE(collector->total() == kRequests);
}

TEST_CASE("submitting after shutdown fails", "[batcher][shutdown]") {
    auto metrics = std::make_shared<Metrics>();
    DynamicBatcher batcher(test_config(4, 5ms), [](Batch&&) {}, metrics);
    batcher.shutdown();
    REQUIRE_FALSE(batcher.submit(make_request(1)));
}

TEST_CASE("a throwing handler does not stall the batcher", "[batcher]") {
    std::atomic<int> calls{0};
    auto metrics = std::make_shared<Metrics>();
    {
        DynamicBatcher batcher(test_config(1, 1ms),
                               [&calls](Batch&&) {
                                   calls.fetch_add(1);
                                   throw std::runtime_error("handler failure");
                               },
                               metrics);
        for (std::uint64_t i = 0; i < 10; ++i) {
            REQUIRE(batcher.submit(make_request(i)));
        }
    }
    REQUIRE(calls.load() == 10);
}

TEST_CASE("concurrent producers all get their requests batched", "[batcher][stress]") {
    constexpr int kProducers = 8;
    constexpr int kPerProducer = 100;

    auto collector = std::make_shared<BatchCollector>();
    auto metrics = std::make_shared<Metrics>();
    {
        DynamicBatcher batcher(test_config(16, 3ms),
                               [collector](Batch&& batch) { (*collector)(std::move(batch)); },
                               metrics);
        std::vector<std::thread> producers;
        producers.reserve(kProducers);
        for (int p = 0; p < kProducers; ++p) {
            producers.emplace_back([&, p] {
                for (int i = 0; i < kPerProducer; ++i) {
                    REQUIRE(batcher.submit(make_request(
                        static_cast<std::uint64_t>(p * kPerProducer + i))));
                }
            });
        }
        for (std::thread& producer : producers) {
            producer.join();
        }
    }

    REQUIRE(collector->total() == kProducers * kPerProducer);

    const auto snapshot = metrics->snapshot();
    REQUIRE(snapshot.requests_received == kProducers * kPerProducer);
    REQUIRE(snapshot.batched_requests == kProducers * kPerProducer);
    REQUIRE(snapshot.average_batch_size > 1.0);
}

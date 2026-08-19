#include "cudaforge/metrics.hpp"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <thread>
#include <vector>

using cudaforge::Metrics;

TEST_CASE("a fresh registry reports zeroes", "[metrics]") {
    Metrics metrics;
    const auto snapshot = metrics.snapshot();
    REQUIRE(snapshot.requests_received == 0);
    REQUIRE(snapshot.requests_completed == 0);
    REQUIRE(snapshot.average_batch_size == 0.0);
    REQUIRE(snapshot.latency_p99_ns == 0);
}

TEST_CASE("counters accumulate", "[metrics]") {
    Metrics metrics;
    for (int i = 0; i < 10; ++i) {
        metrics.record_received();
    }
    metrics.record_rejected();
    metrics.record_failed();
    metrics.record_completion(1'000'000, 32);
    metrics.record_completion(2'000'000, 16);

    const auto snapshot = metrics.snapshot();
    REQUIRE(snapshot.requests_received == 10);
    REQUIRE(snapshot.requests_rejected == 1);
    REQUIRE(snapshot.requests_failed == 1);
    REQUIRE(snapshot.requests_completed == 2);
    REQUIRE(snapshot.tokens_generated == 48);
}

TEST_CASE("average batch size is the ratio of requests to batches", "[metrics]") {
    Metrics metrics;
    metrics.record_batch(8, /*closed_by_timeout=*/false);
    metrics.record_batch(4, /*closed_by_timeout=*/true);

    const auto snapshot = metrics.snapshot();
    REQUIRE(snapshot.batches_processed == 2);
    REQUIRE(snapshot.batched_requests == 12);
    REQUIRE(snapshot.average_batch_size == 6.0);
    REQUIRE(snapshot.batches_closed_by_size == 1);
    REQUIRE(snapshot.batches_closed_by_timeout == 1);
}

TEST_CASE("latency percentiles reflect recorded completions", "[metrics]") {
    Metrics metrics;
    for (int i = 0; i < 99; ++i) {
        metrics.record_completion(1'000'000, 1);
    }
    metrics.record_completion(500'000'000, 1);

    const auto snapshot = metrics.snapshot();
    REQUIRE(snapshot.latency_p50_ns < 2'000'000);
    REQUIRE(snapshot.latency_max_ns == 500'000'000);
    REQUIRE(snapshot.latency_p99_ns > snapshot.latency_p50_ns);
}

TEST_CASE("reset clears every counter", "[metrics]") {
    Metrics metrics;
    metrics.record_received();
    metrics.record_completion(1000, 5);
    metrics.record_batch(4, false);
    metrics.reset();

    const auto snapshot = metrics.snapshot();
    REQUIRE(snapshot.requests_received == 0);
    REQUIRE(snapshot.requests_completed == 0);
    REQUIRE(snapshot.batches_processed == 0);
    REQUIRE(snapshot.tokens_generated == 0);
}

TEST_CASE("json output carries the expected keys", "[metrics]") {
    Metrics metrics;
    metrics.record_received();
    metrics.record_completion(1'500'000, 12);
    metrics.record_batch(3, true);

    const std::string json = cudaforge::to_json(metrics.snapshot());
    REQUIRE(json.front() == '{');
    REQUIRE(json.back() == '}');
    REQUIRE(json.find("\"requests_completed\": 1") != std::string::npos);
    REQUIRE(json.find("\"tokens_generated\": 12") != std::string::npos);
    REQUIRE(json.find("\"latency_ms\"") != std::string::npos);
    REQUIRE(json.find("\"p99\"") != std::string::npos);
}

TEST_CASE("concurrent recording loses no counts", "[metrics][stress]") {
    constexpr int kThreads = 8;
    constexpr int kPerThread = 2000;

    Metrics metrics;
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&] {
            for (int i = 0; i < kPerThread; ++i) {
                metrics.record_received();
                metrics.record_completion(static_cast<std::uint64_t>(i + 1), 2);
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }

    const auto snapshot = metrics.snapshot();
    REQUIRE(snapshot.requests_received == kThreads * kPerThread);
    REQUIRE(snapshot.requests_completed == kThreads * kPerThread);
    REQUIRE(snapshot.tokens_generated == 2 * kThreads * kPerThread);
}

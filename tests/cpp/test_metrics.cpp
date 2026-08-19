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
    metrics.record_expired();
    metrics.record_expired();
    metrics.record_completion(1'000'000, 32);
    metrics.record_completion(2'000'000, 16);

    const auto snapshot = metrics.snapshot();
    REQUIRE(snapshot.requests_received == 10);
    REQUIRE(snapshot.requests_rejected == 1);
    REQUIRE(snapshot.requests_failed == 1);
    // Counted apart from rejection and failure: the three have different causes
    // and different remedies.
    REQUIRE(snapshot.requests_expired == 2);
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
    // 2% slow out of 100 samples. One slow sample would not move p99: the 99th
    // of 100 values is still a fast one, which is the definition working as
    // intended rather than a bug.
    Metrics metrics;
    for (int i = 0; i < 98; ++i) {
        metrics.record_completion(1'000'000, 1);
    }
    metrics.record_completion(500'000'000, 1);
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
    metrics.record_expired();
    metrics.reset();

    const auto snapshot = metrics.snapshot();
    REQUIRE(snapshot.requests_received == 0);
    REQUIRE(snapshot.requests_completed == 0);
    REQUIRE(snapshot.requests_expired == 0);
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
    REQUIRE(json.find("\"requests_expired\"") != std::string::npos);
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

// --- json shape ------------------------------------------------------------

namespace {

/// Balanced-delimiter check, enough to catch every way this emitter could
/// realistically break without adding a JSON parser as a test dependency.
bool json_is_balanced(const std::string& text) {
    int braces = 0;
    bool in_string = false;
    char previous = '\0';

    for (const char c : text) {
        if (in_string) {
            if (c == '"' && previous != '\\') {
                in_string = false;
            }
        } else if (c == '"') {
            in_string = true;
        } else if (c == '{') {
            ++braces;
        } else if (c == '}') {
            if (--braces < 0) {
                return false;
            }
        }
        previous = c;
    }
    return braces == 0 && !in_string;
}

}  // namespace

TEST_CASE("metrics json is structurally balanced", "[metrics][json]") {
    Metrics metrics;
    metrics.record_received();
    metrics.record_completion(1'234'567, 8);
    metrics.record_batch(4, true);
    metrics.record_expired();
    REQUIRE(json_is_balanced(cudaforge::to_json(metrics.snapshot())));
}

TEST_CASE("an empty snapshot still renders valid json", "[metrics][json]") {
    const std::string json = cudaforge::to_json(Metrics{}.snapshot());
    REQUIRE(json_is_balanced(json));
    REQUIRE(json.find("\"requests_received\": 0") != std::string::npos);
}

TEST_CASE("every counter appears in the json", "[metrics][json]") {
    // The field names are a contract shared with the Python registry so a
    // dashboard need not know which runtime produced a snapshot.
    const std::string json = cudaforge::to_json(Metrics{}.snapshot());
    for (const char* field : {"requests_received", "requests_completed", "requests_failed",
                              "requests_rejected", "requests_expired", "batches_processed",
                              "batches_closed_by_size", "batches_closed_by_timeout",
                              "average_batch_size", "queue_depth", "tokens_generated",
                              "uptime_seconds", "requests_per_second", "tokens_per_second",
                              "queue_delay_ms", "latency_ms"}) {
        INFO("field " << field);
        REQUIRE(json.find(std::string("\"") + field + "\"") != std::string::npos);
    }
}

TEST_CASE("durations are reported in milliseconds", "[metrics][json]") {
    // Nanoseconds are the recording unit; milliseconds are what a human reads,
    // and the conversion happens only at the reporting boundary.
    Metrics metrics;
    metrics.record_completion(2'000'000, 1);  // 2 ms

    const std::string json = cudaforge::to_json(metrics.snapshot());
    REQUIRE(json.find("\"max\": 2.0") != std::string::npos);
}

TEST_CASE("nested latency objects are present", "[metrics][json]") {
    const std::string json = cudaforge::to_json(Metrics{}.snapshot());
    REQUIRE(json.find("\"p50\"") != std::string::npos);
    REQUIRE(json.find("\"p95\"") != std::string::npos);
    REQUIRE(json.find("\"p99\"") != std::string::npos);
    REQUIRE(json.find("\"mean\"") != std::string::npos);
}

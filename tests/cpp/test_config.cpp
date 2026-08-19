#include "cudaforge/config.hpp"

#include <catch2/catch_test_macros.hpp>

#include <stdexcept>

using cudaforge::RuntimeConfig;

TEST_CASE("the default config is valid", "[config]") {
    REQUIRE_NOTHROW(RuntimeConfig{}.validate());
}

TEST_CASE("zero-valued sizing knobs are rejected", "[config]") {
    SECTION("max_batch_size") {
        RuntimeConfig config;
        config.max_batch_size = 0;
        REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
    }
    SECTION("queue_capacity") {
        RuntimeConfig config;
        config.queue_capacity = 0;
        REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
    }
    SECTION("worker_threads") {
        RuntimeConfig config;
        config.worker_threads = 0;
        REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
    }
    SECTION("cuda_streams") {
        RuntimeConfig config;
        config.cuda_streams = 0;
        REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
    }
}

TEST_CASE("a queue smaller than a batch is rejected", "[config]") {
    // Otherwise the batcher can never reach max_batch_size and throughput is
    // silently capped at the queue depth.
    RuntimeConfig config;
    config.max_batch_size = 32;
    config.queue_capacity = 16;
    REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
}

TEST_CASE("a negative wait is rejected", "[config]") {
    RuntimeConfig config;
    config.max_wait = std::chrono::microseconds{-1};
    REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
}

TEST_CASE("a zero wait is allowed and means execute immediately", "[config]") {
    RuntimeConfig config;
    config.max_wait = std::chrono::microseconds{0};
    REQUIRE_NOTHROW(config.validate());
}

// --- parity with the Python configuration ----------------------------------

TEST_CASE("the defaults match the python engine defaults", "[config][parity]") {
    // Both runtimes ship the same starting point, and docs/concurrency.md
    // describes one policy for both. Divergent defaults would make that
    // description wrong for whichever runtime a reader happened to be using.
    const RuntimeConfig config;
    REQUIRE(config.max_batch_size == 16);
    REQUIRE(config.max_wait == std::chrono::microseconds{5000});
    REQUIRE(config.queue_capacity == 1024);
    REQUIRE(config.worker_threads == 4);
    REQUIRE(config.cuda_streams == 4);
    REQUIRE(config.max_prompt_chars == 8192);
}

TEST_CASE("the default queue holds many full batches", "[config]") {
    // A queue only as deep as one batch would leave no headroom for arrivals
    // during execution, so the batcher would starve between batches.
    const RuntimeConfig config;
    REQUIRE(config.queue_capacity >= config.max_batch_size * 8);
}

TEST_CASE("a wait equal to the batch service time is representable", "[config]") {
    // The documented starting point: max_wait at roughly the single-request
    // service time. Sub-millisecond waits must round-trip exactly.
    RuntimeConfig config;
    config.max_wait = std::chrono::microseconds{250};
    REQUIRE_NOTHROW(config.validate());
    REQUIRE(config.max_wait.count() == 250);
}

TEST_CASE("a queue exactly the size of a batch is accepted", "[config]") {
    // The boundary: legal, but the batcher can never accumulate ahead.
    RuntimeConfig config;
    config.max_batch_size = 32;
    config.queue_capacity = 32;
    REQUIRE_NOTHROW(config.validate());
}

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

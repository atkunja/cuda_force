#include "cudaforge/latency_histogram.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <thread>
#include <vector>

using cudaforge::LatencyHistogram;

namespace {

/// The histogram's contract is bounded *relative* error, not exactness, so
/// assertions compare against a tolerance derived from the bucket layout rather
/// than against an exact value.
constexpr double kMaxRelativeError = 1.0 / static_cast<double>(LatencyHistogram::kSubBuckets);

bool within_tolerance(std::uint64_t reported, std::uint64_t expected) {
    if (expected == 0) {
        return reported == 0;
    }
    const auto slack = static_cast<double>(expected) * kMaxRelativeError;
    const auto diff = static_cast<double>(reported) - static_cast<double>(expected);
    return diff >= -slack && diff <= slack + 1.0;
}

}  // namespace

TEST_CASE("an empty histogram reports zeroes", "[histogram]") {
    LatencyHistogram histogram;
    REQUIRE(histogram.count() == 0);
    REQUIRE(histogram.percentile(0.5) == 0);
    REQUIRE(histogram.mean_ns() == 0.0);
    REQUIRE(histogram.min_ns() == 0);
}

TEST_CASE("small values are stored exactly", "[histogram]") {
    LatencyHistogram histogram;
    for (std::uint64_t value = 0; value < LatencyHistogram::kSubBuckets; ++value) {
        histogram.record(value);
    }
    REQUIRE(histogram.count() == LatencyHistogram::kSubBuckets);
    REQUIRE(histogram.min_ns() == 0);
    REQUIRE(histogram.max_ns() == LatencyHistogram::kSubBuckets - 1);
}

TEST_CASE("percentiles land within the documented tolerance", "[histogram]") {
    LatencyHistogram histogram;
    for (std::uint64_t value = 1; value <= 10000; ++value) {
        histogram.record(value);
    }

    REQUIRE(histogram.count() == 10000);
    REQUIRE(within_tolerance(histogram.percentile(0.50), 5000));
    REQUIRE(within_tolerance(histogram.percentile(0.95), 9500));
    REQUIRE(within_tolerance(histogram.percentile(0.99), 9900));
    REQUIRE(histogram.max_ns() == 10000);
    REQUIRE(histogram.min_ns() == 1);
}

TEST_CASE("percentiles are monotonic", "[histogram]") {
    LatencyHistogram histogram;
    for (std::uint64_t value = 1; value <= 5000; ++value) {
        histogram.record(value * 7);
    }
    REQUIRE(histogram.percentile(0.50) <= histogram.percentile(0.90));
    REQUIRE(histogram.percentile(0.90) <= histogram.percentile(0.99));
    REQUIRE(histogram.percentile(0.99) <= histogram.percentile(1.0));
}

TEST_CASE("a tail sample moves p99 but not p50", "[histogram]") {
    LatencyHistogram histogram;
    for (int i = 0; i < 990; ++i) {
        histogram.record(1000);
    }
    for (int i = 0; i < 10; ++i) {
        histogram.record(1'000'000);
    }

    REQUIRE(within_tolerance(histogram.percentile(0.50), 1000));
    REQUIRE(histogram.percentile(0.999) >= 900'000);
    REQUIRE(histogram.max_ns() == 1'000'000);
}

TEST_CASE("out of range quantiles are clamped", "[histogram]") {
    LatencyHistogram histogram;
    histogram.record(42);
    REQUIRE(histogram.percentile(-1.0) > 0);
    REQUIRE(histogram.percentile(2.0) > 0);
}

TEST_CASE("mean tracks the true mean exactly", "[histogram]") {
    // The running sum is not bucketed, so the mean is exact even though the
    // percentiles are approximate.
    LatencyHistogram histogram;
    for (std::uint64_t value = 1; value <= 100; ++value) {
        histogram.record(value);
    }
    REQUIRE(histogram.mean_ns() == 50.5);
}

TEST_CASE("reset clears every accumulator", "[histogram]") {
    LatencyHistogram histogram;
    for (int i = 0; i < 100; ++i) {
        histogram.record(1234);
    }
    histogram.reset();
    REQUIRE(histogram.count() == 0);
    REQUIRE(histogram.max_ns() == 0);
    REQUIRE(histogram.percentile(0.99) == 0);
}

TEST_CASE("concurrent recording loses no samples", "[histogram][stress]") {
    constexpr int kThreads = 8;
    constexpr int kPerThread = 5000;

    LatencyHistogram histogram;
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < kPerThread; ++i) {
                histogram.record(static_cast<std::uint64_t>((t + 1) * (i + 1)));
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    REQUIRE(histogram.count() == kThreads * kPerThread);
}

// --- bucket boundaries -----------------------------------------------------

TEST_CASE("bucket boundaries are monotonic across magnitudes", "[histogram][buckets]") {
    // A non-monotonic bucket layout would make percentiles jump backwards at
    // magnitude changes — the kind of defect that only shows up as a confusing
    // dashboard, never as an error.
    LatencyHistogram histogram;
    std::uint64_t previous = 0;

    for (std::uint64_t value = 1; value < (1ULL << 32); value = value + 1 + value / 8) {
        LatencyHistogram single;
        single.record(value);
        const std::uint64_t reported = single.percentile(1.0);

        INFO("value " << value << " reported " << reported);
        // The bucket's upper bound must be at least the value it holds, and
        // must not decrease as the value grows.
        REQUIRE(reported >= value);
        REQUIRE(reported >= previous);
        previous = reported;
    }
}

TEST_CASE("relative error stays within the documented bound", "[histogram][buckets]") {
    constexpr double kBound = 1.0 / static_cast<double>(LatencyHistogram::kSubBuckets);

    for (std::uint64_t value = LatencyHistogram::kSubBuckets; value < (1ULL << 30);
         value = value + 1 + value / 5) {
        LatencyHistogram single;
        single.record(value);
        const auto reported = static_cast<double>(single.percentile(1.0));
        const auto exact = static_cast<double>(value);

        INFO("value " << value << " reported " << reported);
        REQUIRE((reported - exact) / exact <= kBound);
    }
}

TEST_CASE("values below the sub-bucket count are exact", "[histogram][buckets]") {
    // The bottom magnitude maps one-to-one, so there is no rounding at all.
    for (std::uint64_t value = 0; value < LatencyHistogram::kSubBuckets; ++value) {
        LatencyHistogram single;
        single.record(value);
        REQUIRE(single.percentile(1.0) == value);
    }
}

TEST_CASE("an enormous value is clamped rather than overflowing", "[histogram][buckets]") {
    // Past the top bucket the index is clamped; the alternative is writing out
    // of bounds, which a sanitizer would catch but a release build would not.
    LatencyHistogram histogram;
    histogram.record(UINT64_MAX / 2);
    REQUIRE(histogram.count() == 1);
    REQUIRE(histogram.max_ns() == UINT64_MAX / 2);
}

TEST_CASE("min and max bracket every recorded sample", "[histogram][buckets]") {
    LatencyHistogram histogram;
    for (std::uint64_t value : {5ULL, 500ULL, 50'000ULL, 5'000'000ULL}) {
        histogram.record(value);
    }
    REQUIRE(histogram.min_ns() == 5);
    REQUIRE(histogram.max_ns() == 5'000'000);
    // Unlike the percentiles, min and max are stored exactly.
    REQUIRE(histogram.percentile(0.0) <= histogram.max_ns());
}

TEST_CASE("a single sample is its own percentile at every quantile",
          "[histogram][buckets]") {
    LatencyHistogram histogram;
    histogram.record(123'456);
    const std::uint64_t reported = histogram.percentile(0.5);
    for (double quantile : {0.0, 0.25, 0.5, 0.9, 0.99, 1.0}) {
        REQUIRE(histogram.percentile(quantile) == reported);
    }
}

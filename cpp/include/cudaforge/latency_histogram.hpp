#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace cudaforge {

/// Fixed-memory latency histogram with logarithmically spaced buckets.
///
/// Storing every sample would give exact percentiles but grows without bound
/// under sustained load, which is precisely when latency reporting matters
/// most. This trades a bounded relative error for O(1) memory and O(1) record.
///
/// Buckets are `kSubBuckets` linear divisions within each power-of-two
/// magnitude, so the worst-case relative error of any reported percentile is
/// `1 / kSubBuckets`. At 16 sub-buckets that is 6.25%, which is well inside the
/// run-to-run variance of the measurements being bucketed. This is the same
/// shape as HdrHistogram, deliberately simplified: no auto-resize, no
/// concurrent writer striping, one lock.
class LatencyHistogram {
public:
    static constexpr std::size_t kSubBuckets = 16;
    static constexpr std::size_t kMagnitudes = 40;  // covers 1ns .. ~2.4 hours
    static constexpr std::size_t kBucketCount = kSubBuckets * kMagnitudes;

    void record(std::uint64_t value_ns) {
        const std::size_t index = bucket_index(value_ns);
        std::lock_guard lock(mutex_);
        ++counts_[index];
        ++total_count_;
        sum_ns_ += value_ns;
        min_ns_ = std::min(min_ns_, value_ns);
        max_ns_ = std::max(max_ns_, value_ns);
    }

    /// `quantile` in [0, 1]. Returns 0 when no samples have been recorded.
    [[nodiscard]] std::uint64_t percentile(double quantile) const {
        std::lock_guard lock(mutex_);
        if (total_count_ == 0) {
            return 0;
        }
        const double clamped = std::clamp(quantile, 0.0, 1.0);

        // Rank is 1-based: the p-th percentile is the smallest value whose
        // cumulative count reaches ceil(p * N). Using a 0-based rank here would
        // report the bucket below the true percentile for exact boundaries.
        auto rank = static_cast<std::uint64_t>(clamped * static_cast<double>(total_count_));
        if (rank == 0) {
            rank = 1;
        }

        std::uint64_t cumulative = 0;
        for (std::size_t i = 0; i < kBucketCount; ++i) {
            cumulative += counts_[i];
            if (cumulative >= rank) {
                return bucket_upper_bound(i);
            }
        }
        return max_ns_;
    }

    [[nodiscard]] std::uint64_t count() const {
        std::lock_guard lock(mutex_);
        return total_count_;
    }

    [[nodiscard]] double mean_ns() const {
        std::lock_guard lock(mutex_);
        if (total_count_ == 0) {
            return 0.0;
        }
        return static_cast<double>(sum_ns_) / static_cast<double>(total_count_);
    }

    [[nodiscard]] std::uint64_t min_ns() const {
        std::lock_guard lock(mutex_);
        return total_count_ == 0 ? 0 : min_ns_;
    }

    [[nodiscard]] std::uint64_t max_ns() const {
        std::lock_guard lock(mutex_);
        return max_ns_;
    }

    void reset() {
        std::lock_guard lock(mutex_);
        counts_.fill(0);
        total_count_ = 0;
        sum_ns_ = 0;
        min_ns_ = UINT64_MAX;
        max_ns_ = 0;
    }

private:
    /// Values below `kSubBuckets` map one-to-one so sub-nanosecond-resolution
    /// buckets are not wasted on the bottom magnitude.
    static std::size_t bucket_index(std::uint64_t value) {
        if (value < kSubBuckets) {
            return static_cast<std::size_t>(value);
        }
        const int magnitude = 63 - __builtin_clzll(value);
        const int shift = magnitude - static_cast<int>(kSubBucketBits);
        const auto sub = static_cast<std::size_t>((value >> shift) & (kSubBuckets - 1));
        const auto base = static_cast<std::size_t>(magnitude - kSubBucketBits + 1) * kSubBuckets;
        return std::min(base + sub, kBucketCount - 1);
    }

    static std::uint64_t bucket_upper_bound(std::size_t index) {
        if (index < kSubBuckets) {
            return index;
        }
        const auto magnitude = static_cast<int>(index / kSubBuckets) + kSubBucketBits - 1;
        const auto sub = static_cast<std::uint64_t>(index % kSubBuckets);
        const int shift = magnitude - static_cast<int>(kSubBucketBits);
        return ((static_cast<std::uint64_t>(kSubBuckets) + sub + 1) << shift) - 1;
    }

    static constexpr int kSubBucketBits = 4;  // log2(kSubBuckets)
    static_assert(1U << kSubBucketBits == kSubBuckets);

    mutable std::mutex mutex_;
    std::array<std::uint64_t, kBucketCount> counts_{};
    std::uint64_t total_count_ = 0;
    std::uint64_t sum_ns_ = 0;
    std::uint64_t min_ns_ = UINT64_MAX;
    std::uint64_t max_ns_ = 0;
};

}  // namespace cudaforge

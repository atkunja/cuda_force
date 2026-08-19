#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace cudaforge::bench {

using Clock = std::chrono::steady_clock;

/// Wall-clock timer. Deliberately not `high_resolution_clock`: on some
/// implementations that is an alias for `system_clock`, which can step
/// backwards when NTP adjusts the wall clock and produce negative durations.
class Timer {
public:
    void start() { start_ = Clock::now(); }

    [[nodiscard]] double elapsed_seconds() const {
        return std::chrono::duration<double>(Clock::now() - start_).count();
    }

    [[nodiscard]] double elapsed_ms() const { return elapsed_seconds() * 1e3; }

private:
    Clock::time_point start_ = Clock::now();
};

/// Exact percentiles over a collected sample vector.
///
/// The runtime uses a bucketed histogram because it must be O(1) in memory
/// under sustained load. A benchmark knows its sample count up front and runs
/// for a bounded time, so it can afford to sort and report exact values —
/// which also makes it a useful cross-check on the histogram's approximation.
inline double percentile(std::vector<double>& samples, double quantile) {
    if (samples.empty()) {
        return 0.0;
    }
    std::sort(samples.begin(), samples.end());
    const auto position = quantile * static_cast<double>(samples.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = std::min(lower + 1, samples.size() - 1);
    const double weight = position - static_cast<double>(lower);
    return samples[lower] * (1.0 - weight) + samples[upper] * weight;
}

inline double mean(const std::vector<double>& samples) {
    if (samples.empty()) {
        return 0.0;
    }
    double total = 0.0;
    for (double sample : samples) {
        total += sample;
    }
    return total / static_cast<double>(samples.size());
}

/// Minimal JSON emitter. A dependency-free benchmark binary is one fewer thing
/// that can fail to build on the GPU machine where the numbers actually matter.
class JsonWriter {
public:
    explicit JsonWriter(std::ostream& out) : out_(out) {
        out_ << std::fixed << std::setprecision(4);
    }

    void begin_object() {
        out_ << indent() << "{\n";
        ++depth_;
        first_.push_back(true);
    }

    void end_object() {
        --depth_;
        first_.pop_back();
        out_ << "\n" << indent() << "}";
    }

    void begin_array(const std::string& key) {
        separator();
        out_ << indent() << quote(key) << ": [\n";
        ++depth_;
        first_.push_back(true);
    }

    void end_array() {
        --depth_;
        first_.pop_back();
        out_ << "\n" << indent() << "]";
    }

    void field(const std::string& key, double value) {
        separator();
        out_ << indent() << quote(key) << ": " << value;
    }

    void field(const std::string& key, std::uint64_t value) {
        separator();
        out_ << indent() << quote(key) << ": " << value;
    }

    void field(const std::string& key, const std::string& value) {
        separator();
        out_ << indent() << quote(key) << ": " << quote(value);
    }

    void array_element_begin() {
        separator();
        out_ << indent() << "{\n";
        ++depth_;
        first_.push_back(true);
    }

    void array_element_end() {
        --depth_;
        first_.pop_back();
        out_ << "\n" << indent() << "}";
    }

    void finish() { out_ << "\n"; }

private:
    void separator() {
        if (first_.empty()) {
            return;
        }
        if (first_.back()) {
            first_.back() = false;
        } else {
            out_ << ",\n";
        }
    }

    [[nodiscard]] std::string indent() const {
        return std::string(static_cast<std::size_t>(depth_) * 2, ' ');
    }
    static std::string quote(const std::string& value) { return "\"" + value + "\""; }

    std::ostream& out_;
    int depth_ = 0;
    std::vector<bool> first_;
};

/// Keeps a computed value from being eliminated by the optimiser. Without this,
/// a benchmark whose result is unused can be deleted wholesale and will report
/// an impossibly good number.
template <typename T>
inline void keep(T&& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

}  // namespace cudaforge::bench

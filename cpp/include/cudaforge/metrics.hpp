#pragma once

#include <atomic>
#include <cstdint>
#include <string>

#include "cudaforge/latency_histogram.hpp"

namespace cudaforge {

/// Point-in-time view of the runtime, suitable for a /metrics endpoint.
struct MetricsSnapshot {
    std::uint64_t requests_received = 0;
    std::uint64_t requests_completed = 0;
    std::uint64_t requests_failed = 0;
    std::uint64_t requests_rejected = 0;
    std::uint64_t requests_expired = 0;

    std::uint64_t batches_processed = 0;
    std::uint64_t batched_requests = 0;
    std::uint64_t batches_closed_by_size = 0;
    std::uint64_t batches_closed_by_timeout = 0;

    std::uint64_t tokens_generated = 0;
    std::size_t queue_depth = 0;

    double average_batch_size = 0.0;
    double uptime_seconds = 0.0;
    double requests_per_second = 0.0;
    double tokens_per_second = 0.0;

    std::uint64_t queue_delay_p50_ns = 0;
    std::uint64_t queue_delay_p95_ns = 0;
    std::uint64_t queue_delay_p99_ns = 0;

    std::uint64_t latency_p50_ns = 0;
    std::uint64_t latency_p95_ns = 0;
    std::uint64_t latency_p99_ns = 0;
    std::uint64_t latency_max_ns = 0;
    double latency_mean_ns = 0.0;
};

/// Central counter and latency registry.
///
/// Counters are relaxed atomics. The recording sites sit on the per-request hot
/// path, so a mutex here would introduce contention proportional to throughput
/// — a metrics system that degrades the thing it measures. Relaxed ordering is
/// correct because no counter publishes data to another thread; each is read
/// only for reporting.
///
/// The consequence is that `snapshot()` is not an atomic view: two counters may
/// be read either side of an update. Derived values are therefore computed
/// defensively (see `average_batch_size`) and no invariant is asserted across
/// fields.
class Metrics {
public:
    Metrics() : started_(Clock::now()) {}

    void record_received() noexcept { received_.fetch_add(1, std::memory_order_relaxed); }
    void record_rejected() noexcept { rejected_.fetch_add(1, std::memory_order_relaxed); }
    void record_failed() noexcept { failed_.fetch_add(1, std::memory_order_relaxed); }

    /// A request dropped at dequeue because its deadline had passed. Counted
    /// apart from `rejected` (refused at admission) and `failed` (execution
    /// error): a rising expiry count means the queue is deeper than clients
    /// will wait for, which calls for shedding earlier rather than for more
    /// capacity.
    void record_expired() noexcept { expired_.fetch_add(1, std::memory_order_relaxed); }

    void record_queue_delay(std::uint64_t nanos) { queue_delay_.record(nanos); }

    void record_completion(std::uint64_t latency_ns, std::uint32_t tokens) noexcept {
        completed_.fetch_add(1, std::memory_order_relaxed);
        tokens_.fetch_add(tokens, std::memory_order_relaxed);
        latency_.record(latency_ns);
    }

    void record_batch(std::size_t batch_size, bool closed_by_timeout) noexcept {
        batches_.fetch_add(1, std::memory_order_relaxed);
        batched_requests_.fetch_add(batch_size, std::memory_order_relaxed);
        if (closed_by_timeout) {
            timeout_closures_.fetch_add(1, std::memory_order_relaxed);
        } else {
            size_closures_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    void set_queue_depth(std::size_t depth) noexcept {
        queue_depth_.store(depth, std::memory_order_relaxed);
    }

    [[nodiscard]] MetricsSnapshot snapshot() const {
        const auto elapsed = std::chrono::duration<double>(Clock::now() - started_).count();
        const auto batches = batches_.load(std::memory_order_relaxed);
        const auto batched = batched_requests_.load(std::memory_order_relaxed);
        const auto completed = completed_.load(std::memory_order_relaxed);
        const auto tokens = tokens_.load(std::memory_order_relaxed);

        MetricsSnapshot snap;
        snap.requests_received = received_.load(std::memory_order_relaxed);
        snap.requests_completed = completed;
        snap.requests_failed = failed_.load(std::memory_order_relaxed);
        snap.requests_rejected = rejected_.load(std::memory_order_relaxed);
        snap.requests_expired = expired_.load(std::memory_order_relaxed);

        snap.batches_processed = batches;
        snap.batched_requests = batched;
        snap.batches_closed_by_size = size_closures_.load(std::memory_order_relaxed);
        snap.batches_closed_by_timeout = timeout_closures_.load(std::memory_order_relaxed);

        snap.tokens_generated = tokens;
        snap.queue_depth = queue_depth_.load(std::memory_order_relaxed);

        snap.average_batch_size =
            batches > 0 ? static_cast<double>(batched) / static_cast<double>(batches) : 0.0;
        snap.uptime_seconds = elapsed;
        if (elapsed > 0.0) {
            snap.requests_per_second = static_cast<double>(completed) / elapsed;
            snap.tokens_per_second = static_cast<double>(tokens) / elapsed;
        }

        snap.queue_delay_p50_ns = queue_delay_.percentile(0.50);
        snap.queue_delay_p95_ns = queue_delay_.percentile(0.95);
        snap.queue_delay_p99_ns = queue_delay_.percentile(0.99);

        snap.latency_p50_ns = latency_.percentile(0.50);
        snap.latency_p95_ns = latency_.percentile(0.95);
        snap.latency_p99_ns = latency_.percentile(0.99);
        snap.latency_max_ns = latency_.max_ns();
        snap.latency_mean_ns = latency_.mean_ns();
        return snap;
    }

    /// Not safe to call concurrently with `snapshot()` — `started_` is a plain
    /// member and resetting it races a reader. Intended for test setup and for
    /// benchmark phase boundaries, where the runtime is quiesced.
    void reset() {
        received_.store(0, std::memory_order_relaxed);
        completed_.store(0, std::memory_order_relaxed);
        failed_.store(0, std::memory_order_relaxed);
        rejected_.store(0, std::memory_order_relaxed);
        expired_.store(0, std::memory_order_relaxed);
        batches_.store(0, std::memory_order_relaxed);
        batched_requests_.store(0, std::memory_order_relaxed);
        size_closures_.store(0, std::memory_order_relaxed);
        timeout_closures_.store(0, std::memory_order_relaxed);
        tokens_.store(0, std::memory_order_relaxed);
        queue_depth_.store(0, std::memory_order_relaxed);
        latency_.reset();
        queue_delay_.reset();
        started_ = Clock::now();
    }

private:
    using Clock = std::chrono::steady_clock;

    std::atomic<std::uint64_t> received_{0};
    std::atomic<std::uint64_t> completed_{0};
    std::atomic<std::uint64_t> failed_{0};
    std::atomic<std::uint64_t> rejected_{0};
    std::atomic<std::uint64_t> expired_{0};
    std::atomic<std::uint64_t> batches_{0};
    std::atomic<std::uint64_t> batched_requests_{0};
    std::atomic<std::uint64_t> size_closures_{0};
    std::atomic<std::uint64_t> timeout_closures_{0};
    std::atomic<std::uint64_t> tokens_{0};
    std::atomic<std::size_t> queue_depth_{0};

    LatencyHistogram latency_;
    LatencyHistogram queue_delay_;
    Clock::time_point started_;
};

std::string to_json(const MetricsSnapshot& snapshot);

}  // namespace cudaforge

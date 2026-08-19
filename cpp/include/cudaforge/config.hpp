#pragma once

#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace cudaforge {

/// Runtime tuning knobs, mirroring `inference/configs/*.yaml`.
///
/// The two batching parameters trade throughput against tail latency and pull
/// in opposite directions:
///
/// - `max_batch_size` raises GPU efficiency (one kernel launch amortised over
///   more rows) but a larger batch takes longer to fill and longer to execute,
///   so every request in it waits longer.
/// - `max_wait` caps how long the batcher will hold an early arrival hostage
///   waiting for company. It is the direct upper bound on batching-induced
///   queue delay, and therefore the main p99 lever.
///
/// A sensible starting point is `max_wait` at roughly the single-request
/// service time: below that the batcher rarely fills a batch, above it the
/// added delay exceeds the throughput it buys.
struct RuntimeConfig {
    std::size_t max_batch_size = 16;
    std::chrono::microseconds max_wait{5000};
    std::size_t queue_capacity = 1024;
    std::size_t worker_threads = 4;
    std::size_t cuda_streams = 4;

    /// Applied at ingress. Zero disables the check.
    std::size_t max_prompt_chars = 8192;

    void validate() const {
        if (max_batch_size == 0) {
            throw std::invalid_argument("max_batch_size must be non-zero");
        }
        if (queue_capacity == 0) {
            throw std::invalid_argument("queue_capacity must be non-zero");
        }
        if (worker_threads == 0) {
            throw std::invalid_argument("worker_threads must be non-zero");
        }
        if (cuda_streams == 0) {
            throw std::invalid_argument("cuda_streams must be non-zero");
        }
        if (max_wait.count() < 0) {
            throw std::invalid_argument("max_wait must not be negative");
        }
        // A queue that cannot hold a full batch guarantees the batcher never
        // reaches max_batch_size, silently capping throughput at the queue
        // depth. Catch it at construction rather than in a benchmark.
        if (queue_capacity < max_batch_size) {
            throw std::invalid_argument("queue_capacity must be at least max_batch_size");
        }
    }
};

}  // namespace cudaforge

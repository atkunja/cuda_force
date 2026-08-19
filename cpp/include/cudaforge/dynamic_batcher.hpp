#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <thread>

#include "cudaforge/concurrent_queue.hpp"
#include "cudaforge/config.hpp"
#include "cudaforge/metrics.hpp"
#include "cudaforge/request.hpp"

namespace cudaforge {

/// Aggregates concurrently arriving requests into batches.
///
/// ## The decision being made
///
/// GPU inference is dominated by weight movement, not arithmetic: running one
/// request through a transformer layer reads the same weights as running
/// sixteen. Batching amortises that read, so throughput rises close to linearly
/// with batch size until the kernel becomes compute-bound.
///
/// The cost is latency. A request that arrives when the queue is empty would be
/// fastest if executed immediately; batching makes it wait for neighbours. The
/// batcher therefore closes a batch on whichever comes first:
///
/// 1. it holds `max_batch_size` requests, or
/// 2. the *oldest* request in it has waited `max_wait`.
///
/// The deadline is anchored to the oldest request, not reset on each arrival.
/// Resetting it would let a steady arrival stream postpone execution
/// indefinitely — the classic Nagle-style starvation bug — so no request can
/// ever be delayed by more than `max_wait` plus the batch's own service time.
///
/// ## Threading
///
/// A single batcher thread owns batch formation. Formation is inherently
/// serial: two threads racing to claim requests from the same queue would need
/// a lock that serialises them anyway, and would make the deadline
/// non-deterministic. Parallelism lives downstream, in the executor.
class DynamicBatcher {
public:
    /// Invoked on the batcher thread. Should hand work off rather than block:
    /// time spent here is time no batch is being formed.
    using BatchHandler = std::function<void(Batch&&)>;

    /// Invoked for each request dropped because its deadline had already
    /// passed. Optional, but an owner that tracks outstanding work needs it:
    /// without it a dropped request simply vanishes and whoever is waiting on
    /// it waits out their own timeout with no explanation.
    using ExpiryHandler = std::function<void(const Request&)>;

    DynamicBatcher(RuntimeConfig config, BatchHandler handler, std::shared_ptr<Metrics> metrics,
                   ExpiryHandler on_expired = nullptr);

    DynamicBatcher(const DynamicBatcher&) = delete;
    DynamicBatcher& operator=(const DynamicBatcher&) = delete;
    DynamicBatcher(DynamicBatcher&&) = delete;
    DynamicBatcher& operator=(DynamicBatcher&&) = delete;

    ~DynamicBatcher();

    /// Blocks while the queue is full, applying backpressure to the caller.
    /// Returns false once shutdown has begun.
    bool submit(Request request);

    /// Rejects instead of blocking when the queue is full. Ingress paths that
    /// prefer shedding load to queueing it use this.
    QueueStatus try_submit(Request request);

    /// Stops accepting work, drains whatever is queued into final batches, and
    /// joins the batcher thread. Idempotent.
    void shutdown() noexcept;

    [[nodiscard]] std::size_t queue_depth() const { return queue_.size(); }
    [[nodiscard]] const RuntimeConfig& config() const noexcept { return config_; }

private:
    void run();

    /// Blocks for the first request, then fills opportunistically until the
    /// size limit or the oldest request's deadline. Returns an empty batch only
    /// when the queue is closed and drained.
    Batch collect_batch();

    /// Discards a request whose deadline has passed. Returns true if dropped.
    bool drop_if_expired(const Request& request);

    RuntimeConfig config_;
    BatchHandler handler_;
    ExpiryHandler on_expired_;
    std::shared_ptr<Metrics> metrics_;
    ConcurrentQueue<Request> queue_;
    std::atomic<bool> stopping_{false};
    std::thread worker_;
};

}  // namespace cudaforge

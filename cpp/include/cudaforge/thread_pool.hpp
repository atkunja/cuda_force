#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "cudaforge/concurrent_queue.hpp"

namespace cudaforge {

/// Snapshot of pool activity. Every field is read from a relaxed atomic, so the
/// fields are individually accurate but not mutually consistent — `submitted`
/// and `completed` may be read either side of a task finishing. That is
/// acceptable for reporting and is why no invariant is derived from the pair.
struct ThreadPoolStats {
    std::uint64_t submitted = 0;
    std::uint64_t completed = 0;
    std::uint64_t failed = 0;
    std::size_t active_workers = 0;
    std::size_t queue_depth = 0;
    std::size_t worker_count = 0;
};

/// Fixed-size worker pool over a bounded task queue.
///
/// The queue is bounded on purpose: an unbounded task queue lets a fast
/// producer accumulate work faster than the pool retires it, which shows up as
/// memory growth first and unbounded latency second. Bounding it makes
/// `submit` block, pushing backpressure onto the caller where it can be
/// observed and handled.
///
/// Counters are `std::atomic` rather than mutex-protected because they are
/// independent single words on the hot path; taking the queue's mutex to bump a
/// counter would serialise workers that otherwise never interact. `relaxed`
/// ordering suffices — nothing else is published through these counters, and no
/// reader uses them to establish happens-before.
class ThreadPool {
public:
    using Task = std::function<void()>;

    explicit ThreadPool(std::size_t worker_count, std::size_t queue_capacity = 1024)
        : tasks_(queue_capacity) {
        if (worker_count == 0) {
            throw std::invalid_argument("ThreadPool requires at least one worker");
        }
        workers_.reserve(worker_count);
        for (std::size_t i = 0; i < worker_count; ++i) {
            workers_.emplace_back([this] { run(); });
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    /// Drains outstanding work before returning. Destruction is a join point,
    /// not a cancellation point — a task already queued still runs.
    ~ThreadPool() { shutdown(); }

    /// Blocks while the task queue is full. Returns false if the pool is
    /// shutting down, in which case the task is not queued and never runs.
    bool submit(Task task) {
        if (!task) {
            throw std::invalid_argument("ThreadPool::submit received an empty task");
        }
        if (tasks_.push(std::move(task)) != QueueStatus::Ok) {
            return false;
        }
        submitted_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    /// Submits and hands back a future for the result.
    ///
    /// The packaged task is heap-allocated behind a `shared_ptr` because
    /// `std::function` requires a copyable target and `std::packaged_task` is
    /// move-only. The indirection is one allocation per submission; for tasks
    /// coarse enough to be worth a thread hop, that is not the bottleneck.
    template <typename F, typename... Args>
    auto submit_with_result(F&& fn, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>> {
        using Result = std::invoke_result_t<F, Args...>;

        auto packaged = std::make_shared<std::packaged_task<Result()>>(
            [fn = std::forward<F>(fn),
             tuple = std::make_tuple(std::forward<Args>(args)...)]() mutable -> Result {
                return std::apply(std::move(fn), std::move(tuple));
            });

        std::future<Result> future = packaged->get_future();
        if (!submit([packaged] { (*packaged)(); })) {
            throw std::runtime_error("ThreadPool::submit_with_result on a stopped pool");
        }
        return future;
    }

    /// Idempotent. Closes the queue, waits for every worker to finish the work
    /// already accepted, and joins. Called by the destructor.
    void shutdown() noexcept {
        if (stopping_.exchange(true, std::memory_order_acq_rel)) {
            return;
        }
        tasks_.shutdown();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    [[nodiscard]] ThreadPoolStats stats() const {
        return ThreadPoolStats{
            .submitted = submitted_.load(std::memory_order_relaxed),
            .completed = completed_.load(std::memory_order_relaxed),
            .failed = failed_.load(std::memory_order_relaxed),
            .active_workers = active_.load(std::memory_order_relaxed),
            .queue_depth = tasks_.size(),
            .worker_count = workers_.size(),
        };
    }

    [[nodiscard]] std::size_t worker_count() const noexcept { return workers_.size(); }

private:
    void run() {
        Task task;
        while (tasks_.pop(task) == QueueStatus::Ok) {
            active_.fetch_add(1, std::memory_order_relaxed);
            try {
                task();
                completed_.fetch_add(1, std::memory_order_relaxed);
            } catch (...) {
                // A throwing task must not take the worker down with it, or the
                // pool silently loses capacity. The exception is counted and
                // swallowed here; callers who need the error should use
                // submit_with_result, where it is stored in the future.
                failed_.fetch_add(1, std::memory_order_relaxed);
            }
            active_.fetch_sub(1, std::memory_order_relaxed);
            task = nullptr;  // release captured state before blocking again
        }
    }

    ConcurrentQueue<Task> tasks_;
    std::vector<std::thread> workers_;

    std::atomic<std::uint64_t> submitted_{0};
    std::atomic<std::uint64_t> completed_{0};
    std::atomic<std::uint64_t> failed_{0};
    std::atomic<std::size_t> active_{0};
    std::atomic<bool> stopping_{false};
};

}  // namespace cudaforge

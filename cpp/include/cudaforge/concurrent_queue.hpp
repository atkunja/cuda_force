#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

namespace cudaforge {

/// Outcome of a queue operation. Callers must distinguish "the queue is closed"
/// from "the wait expired": the first is terminal and should unwind the caller,
/// the second is routine and should be retried.
enum class QueueStatus : std::uint8_t {
    Ok,
    Timeout,
    Closed,  ///< shutdown() was called and (for pop) the queue is drained
    Full,    ///< try_push only
    Empty,   ///< try_pop only
};

/// Bounded, blocking, multi-producer multi-consumer queue.
///
/// ## Why a mutex at all
///
/// The invariant being protected is not a single word — it is the pair
/// (`buffer_`, `closed_`) plus the relationship between them. A producer must
/// observe "not closed AND not full" and then insert, with no window in between
/// where another thread could close the queue or steal the last slot. Atomics
/// cannot express that composite check-then-act, so a mutex owns both fields
/// and every predicate is evaluated under it.
///
/// ## Why condition variables rather than polling
///
/// A consumer waiting on an empty queue has nothing useful to do. Spinning
/// burns a core and, on a machine where producers and consumers outnumber
/// cores, actively steals cycles from the producer it is waiting for. Sleeping
/// in a retry loop adds latency quantised to the sleep interval. A condition
/// variable parks the thread in the kernel and the producer's `notify_one`
/// wakes exactly one waiter at the moment the state changes.
///
/// ## Spurious wakeups
///
/// `wait` may return without a matching notify. Every wait here therefore uses
/// the predicate overload, which re-checks the condition in a loop and only
/// returns once it genuinely holds. There is no bare `wait(lock)` in this file,
/// and there must not be: a bare wait would let a consumer proceed to pop from
/// an empty deque.
///
/// ## Lock ownership
///
/// `std::unique_lock` is used rather than `std::lock_guard` because
/// `condition_variable::wait` must be able to release and reacquire the mutex.
/// Notifications are issued *after* the lock is released where practical, so a
/// woken thread does not immediately block on a mutex the notifier still holds.
///
/// ## Shutdown
///
/// `shutdown()` is idempotent. It sets `closed_` and broadcasts to every
/// waiter. Producers blocked in `push` return `Closed` and their payload is
/// left untouched. Consumers continue to drain buffered items — a closed queue
/// with items still in it returns `Ok`, not `Closed` — so in-flight work is not
/// dropped. Only once the buffer is empty does `pop` return `Closed`.
///
/// ## Exception safety
///
/// The only operation that can throw is the move/copy of `T` into or out of the
/// buffer. Insertion happens as the last step under the lock; if it throws, the
/// deque is unchanged and no notification has been sent, so the queue is still
/// consistent. Extraction moves out and then pops, so a throwing move leaves
/// the element in place rather than losing it.
template <typename T>
class ConcurrentQueue {
public:
    explicit ConcurrentQueue(std::size_t capacity) : capacity_(capacity) {
        if (capacity_ == 0) {
            throw std::invalid_argument("ConcurrentQueue capacity must be non-zero");
        }
    }

    ConcurrentQueue(const ConcurrentQueue&) = delete;
    ConcurrentQueue& operator=(const ConcurrentQueue&) = delete;
    ConcurrentQueue(ConcurrentQueue&&) = delete;
    ConcurrentQueue& operator=(ConcurrentQueue&&) = delete;
    ~ConcurrentQueue() = default;

    /// Blocks while the queue is full. Returns `Closed` if shutdown happens
    /// first, leaving `value` untouched so the caller can react to the rejection.
    QueueStatus push(T value) {
        {
            std::unique_lock lock(mutex_);
            not_full_.wait(lock, [this] { return closed_ || buffer_.size() < capacity_; });
            if (closed_) {
                return QueueStatus::Closed;
            }
            buffer_.push_back(std::move(value));
        }
        not_empty_.notify_one();
        return QueueStatus::Ok;
    }

    /// Non-blocking insert. Returns `Full` rather than waiting, which is what an
    /// ingress path wanting to shed load instead of applying backpressure needs.
    QueueStatus try_push(T value) {
        {
            std::unique_lock lock(mutex_);
            if (closed_) {
                return QueueStatus::Closed;
            }
            if (buffer_.size() >= capacity_) {
                return QueueStatus::Full;
            }
            buffer_.push_back(std::move(value));
        }
        not_empty_.notify_one();
        return QueueStatus::Ok;
    }

    /// Blocks until an item is available or the queue is closed and drained.
    QueueStatus pop(T& out) {
        {
            std::unique_lock lock(mutex_);
            not_empty_.wait(lock, [this] { return closed_ || !buffer_.empty(); });
            if (buffer_.empty()) {
                return QueueStatus::Closed;  // only reachable when closed_
            }
            out = std::move(buffer_.front());
            buffer_.pop_front();
        }
        not_full_.notify_one();
        return QueueStatus::Ok;
    }

    /// Bounded wait. `Timeout` is distinct from `Closed` so a worker can use the
    /// timeout as a heartbeat tick without mistaking it for shutdown.
    template <typename Rep, typename Period>
    QueueStatus pop_for(T& out, const std::chrono::duration<Rep, Period>& timeout) {
        {
            std::unique_lock lock(mutex_);
            const bool ready = not_empty_.wait_for(
                lock, timeout, [this] { return closed_ || !buffer_.empty(); });
            if (!ready) {
                return QueueStatus::Timeout;
            }
            if (buffer_.empty()) {
                return QueueStatus::Closed;
            }
            out = std::move(buffer_.front());
            buffer_.pop_front();
        }
        not_full_.notify_one();
        return QueueStatus::Ok;
    }

    QueueStatus try_pop(T& out) {
        {
            std::unique_lock lock(mutex_);
            if (buffer_.empty()) {
                return closed_ ? QueueStatus::Closed : QueueStatus::Empty;
            }
            out = std::move(buffer_.front());
            buffer_.pop_front();
        }
        not_full_.notify_one();
        return QueueStatus::Ok;
    }

    /// Wakes every waiter. Safe to call repeatedly and from any thread.
    void shutdown() noexcept {
        {
            std::lock_guard lock(mutex_);
            if (closed_) {
                return;
            }
            closed_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    [[nodiscard]] bool closed() const {
        std::lock_guard lock(mutex_);
        return closed_;
    }

    /// Instantaneous depth. Inherently stale the moment it is returned; useful
    /// for metrics, never for control flow.
    [[nodiscard]] std::size_t size() const {
        std::lock_guard lock(mutex_);
        return buffer_.size();
    }

    [[nodiscard]] bool empty() const {
        std::lock_guard lock(mutex_);
        return buffer_.empty();
    }

    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

private:
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::deque<T> buffer_;
    std::size_t capacity_;
    bool closed_ = false;
};

}  // namespace cudaforge

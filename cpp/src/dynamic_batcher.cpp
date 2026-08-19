#include "cudaforge/dynamic_batcher.hpp"

#include <stdexcept>
#include <utility>

namespace cudaforge {

DynamicBatcher::DynamicBatcher(RuntimeConfig config, BatchHandler handler,
                               std::shared_ptr<Metrics> metrics, ExpiryHandler on_expired)
    : config_(config),
      handler_(std::move(handler)),
      on_expired_(std::move(on_expired)),
      metrics_(std::move(metrics)),
      queue_(config.queue_capacity) {
    config_.validate();
    if (!handler_) {
        throw std::invalid_argument("DynamicBatcher requires a batch handler");
    }
    if (!metrics_) {
        throw std::invalid_argument("DynamicBatcher requires a metrics registry");
    }
    // Started last: the thread immediately touches every member above, so
    // construction of those must be complete before it runs.
    worker_ = std::thread([this] { run(); });
}

DynamicBatcher::~DynamicBatcher() {
    shutdown();
}

bool DynamicBatcher::submit(Request request) {
    metrics_->record_received();
    const QueueStatus status = queue_.push(std::move(request));
    metrics_->set_queue_depth(queue_.size());
    if (status != QueueStatus::Ok) {
        metrics_->record_rejected();
        return false;
    }
    return true;
}

QueueStatus DynamicBatcher::try_submit(Request request) {
    metrics_->record_received();
    const QueueStatus status = queue_.try_push(std::move(request));
    metrics_->set_queue_depth(queue_.size());
    if (status != QueueStatus::Ok) {
        metrics_->record_rejected();
    }
    return status;
}

void DynamicBatcher::shutdown() noexcept {
    if (stopping_.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    // Closing the queue both unblocks producers and lets the batcher thread's
    // pop return Closed once drained, which is how run() exits.
    queue_.shutdown();
    if (worker_.joinable()) {
        worker_.join();
    }
}

void DynamicBatcher::run() {
    while (true) {
        Batch batch = collect_batch();
        if (batch.empty()) {
            return;  // queue closed and drained
        }

        metrics_->record_batch(batch.size(), batch.trigger == BatchTrigger::Timeout);
        metrics_->set_queue_depth(queue_.size());

        for (Request& request : batch.requests) {
            metrics_->record_queue_delay(static_cast<std::uint64_t>(request.queue_delay().count()));
        }

        try {
            handler_(std::move(batch));
        } catch (...) {
            // A throwing handler must not kill the batcher, or every subsequent
            // request stalls in the queue until shutdown. The failure is
            // attributed to the requests that were in flight.
            metrics_->record_failed();
        }
    }
}

bool DynamicBatcher::drop_if_expired(const Request& request) {
    if (!request.expired()) {
        return false;
    }
    metrics_->record_expired();
    if (on_expired_) {
        try {
            on_expired_(request);
        } catch (...) {
            // The handler is caller-supplied; a throwing one must not stop the
            // only thread that drains the queue.
            metrics_->record_failed();
        }
    }
    return true;
}

Batch DynamicBatcher::collect_batch() {
    Batch batch;
    batch.requests.reserve(config_.max_batch_size);

    // Loops past expired requests so the batch always has at least one live
    // member; an empty batch is reserved for "queue closed and drained".
    Request first;
    while (true) {
        if (queue_.pop(first) != QueueStatus::Ok) {
            return batch;  // closed and drained
        }
        first.dequeued = Clock::now();
        if (!drop_if_expired(first)) {
            break;
        }
    }

    // Anchored to the first request, never extended. This is what bounds
    // batching-induced queue delay at max_wait regardless of arrival rate.
    const TimePoint deadline = first.dequeued + config_.max_wait;
    batch.requests.push_back(std::move(first));

    while (batch.requests.size() < config_.max_batch_size) {
        const auto remaining = deadline - Clock::now();
        if (remaining <= Duration::zero()) {
            batch.trigger = BatchTrigger::Timeout;
            break;
        }

        Request next;
        const QueueStatus status = queue_.pop_for(next, remaining);
        if (status == QueueStatus::Timeout) {
            batch.trigger = BatchTrigger::Timeout;
            break;
        }
        if (status == QueueStatus::Closed) {
            // Drain whatever is still buffered rather than abandoning it, then
            // let the next collect_batch() call observe the empty queue.
            batch.trigger = BatchTrigger::Shutdown;
            while (batch.requests.size() < config_.max_batch_size &&
                   queue_.try_pop(next) == QueueStatus::Ok) {
                next.dequeued = Clock::now();
                if (drop_if_expired(next)) {
                    continue;
                }
                batch.requests.push_back(std::move(next));
            }
            break;
        }

        next.dequeued = Clock::now();
        if (drop_if_expired(next)) {
            continue;
        }
        batch.requests.push_back(std::move(next));
    }

    if (batch.requests.size() >= config_.max_batch_size) {
        batch.trigger = BatchTrigger::MaxSize;
    }
    batch.formed = Clock::now();
    return batch;
}

}  // namespace cudaforge

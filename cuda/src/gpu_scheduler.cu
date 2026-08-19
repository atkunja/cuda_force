#include "cudaforge/gpu_scheduler.cuh"

#include <stdexcept>

#include "cudaforge/cuda_error.cuh"

namespace cudaforge {

GpuScheduler::GpuScheduler(std::size_t stream_count, int device) : device_(device) {
    if (stream_count == 0) {
        throw std::invalid_argument("GpuScheduler requires at least one stream");
    }

    CUDAFORGE_CHECK(cudaSetDevice(device_));

    // Streams are created at the highest available priority band so that
    // inference work is not preempted by lower-priority background streams a
    // host application may also be running on this device.
    int lowest = 0;
    int highest = 0;
    CUDAFORGE_CHECK(cudaDeviceGetStreamPriorityRange(&lowest, &highest));

    streams_.reserve(stream_count);
    completion_events_.reserve(stream_count);
    for (std::size_t i = 0; i < stream_count; ++i) {
        streams_.emplace_back(highest);
        // Ordering-only events: these are waited on, never timed, so disabling
        // timing removes a timestamp write from every record.
        completion_events_.emplace_back(CudaEvent::Purpose::Ordering);
    }
    stats_.resize(stream_count);
}

GpuScheduler::~GpuScheduler() {
    // Destroying a stream with work still queued is undefined. Draining first
    // is the only safe order, and the destructor must not throw, so the failure
    // is swallowed — at this point the process is tearing down regardless.
    try {
        synchronize_all();
    } catch (const CudaError&) {
    }
}

StreamLease GpuScheduler::acquire() {
    const std::uint64_t ticket = next_stream_.fetch_add(1, std::memory_order_relaxed);
    const std::size_t index = static_cast<std::size_t>(ticket % streams_.size());
    return StreamLease(streams_[index], completion_events_[index], index);
}

void GpuScheduler::copy_to_device(const StreamLease& lease, void* device, const void* host,
                                  std::size_t bytes) {
    if (bytes == 0) {
        return;
    }
    CUDAFORGE_CHECK(cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, lease.stream()));

    std::lock_guard lock(stats_mutex_);
    stats_[lease.index()].bytes_to_device += bytes;
}

void GpuScheduler::copy_to_host(const StreamLease& lease, void* host, const void* device,
                                std::size_t bytes) {
    if (bytes == 0) {
        return;
    }
    CUDAFORGE_CHECK(cudaMemcpyAsync(host, device, bytes, cudaMemcpyDeviceToHost, lease.stream()));

    std::lock_guard lock(stats_mutex_);
    stats_[lease.index()].bytes_to_host += bytes;
}

void GpuScheduler::chain(const StreamLease& waiter, const StreamLease& signaller) {
    stream_wait_event(waiter.stream(), signaller.completion_event());
}

void GpuScheduler::note_dispatch(const StreamLease& lease) {
    std::lock_guard lock(stats_mutex_);
    stats_[lease.index()].batches_dispatched++;
}

void GpuScheduler::synchronize_all() {
    for (const CudaStream& stream : streams_) {
        stream.synchronize();
    }
}

SchedulerStats GpuScheduler::stats() const {
    std::lock_guard lock(stats_mutex_);

    SchedulerStats snapshot;
    snapshot.stream_count = streams_.size();
    snapshot.per_stream = stats_;
    for (const StreamStats& entry : stats_) {
        snapshot.total_dispatches += entry.batches_dispatched;
        snapshot.total_bytes_to_device += entry.bytes_to_device;
        snapshot.total_bytes_to_host += entry.bytes_to_host;
    }
    return snapshot;
}

}  // namespace cudaforge

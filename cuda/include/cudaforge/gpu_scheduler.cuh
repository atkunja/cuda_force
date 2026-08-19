#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "cudaforge/cuda_raii.cuh"

namespace cudaforge {

struct StreamStats {
    std::uint64_t batches_dispatched = 0;
    std::uint64_t bytes_to_device = 0;
    std::uint64_t bytes_to_host = 0;
};

struct SchedulerStats {
    std::size_t stream_count = 0;
    std::uint64_t total_dispatches = 0;
    std::uint64_t total_bytes_to_device = 0;
    std::uint64_t total_bytes_to_host = 0;
    std::vector<StreamStats> per_stream;
};

/// Exclusive use of one stream, with its completion event.
///
/// Returned by `GpuScheduler::acquire`. Holding it means no other thread will
/// be handed the same stream, which is what allows a worker to issue a whole
/// H2D → kernel → D2H sequence and rely on the stream's own ordering rather
/// than on locks.
class StreamLease {
public:
    StreamLease(CudaStream& stream, CudaEvent& done, std::size_t index)
        : stream_(&stream), done_(&done), index_(index) {}

    [[nodiscard]] cudaStream_t stream() const noexcept { return stream_->get(); }
    [[nodiscard]] CudaEvent& completion_event() const noexcept { return *done_; }
    [[nodiscard]] std::size_t index() const noexcept { return index_; }

    /// Records the lease's event on its stream. The event completes once every
    /// operation issued to the stream so far has finished.
    void record_completion() const { done_->record(stream_->get()); }

    /// Blocks the calling host thread until this stream's work is done. Other
    /// streams are unaffected.
    void synchronize() const { stream_->synchronize(); }

    [[nodiscard]] bool complete() const { return stream_->query(); }

private:
    CudaStream* stream_;
    CudaEvent* done_;
    std::size_t index_;
};

/// Owns a fixed pool of CUDA streams and hands them out to worker threads.
///
/// ## What overlapping actually requires
///
/// A GPU can run a kernel, a host-to-device copy and a device-to-host copy at
/// the same time, because copy engines are separate hardware from the SMs. That
/// only happens if three conditions all hold:
///
///   1. The operations are issued to **different** streams. Same-stream work is
///      strictly ordered by definition.
///   2. Host memory involved in the copies is **pinned**. Copies from pageable
///      memory are staged synchronously through a driver buffer and cannot
///      overlap. See `PinnedBuffer`.
///   3. No device-wide synchronisation intervenes. A single
///      `cudaDeviceSynchronize` is a barrier across every stream and collapses
///      the pipeline — which is why it appears nowhere in this project.
///
/// With N streams and per-batch work split into copy-in, compute and copy-out,
/// batch `i`'s compute overlaps batch `i+1`'s copy-in and batch `i-1`'s
/// copy-out. The steady-state throughput gain approaches the ratio of total
/// work to compute-only work.
///
/// ## Assignment policy
///
/// Round-robin over a counter. The alternative — querying each stream and
/// picking an idle one — costs a driver call per acquisition and, because
/// `cudaStreamQuery` reports only whether *all* work is done, tends to pile
/// short batches onto whichever stream finished first. Round-robin is
/// predictable, contention-free, and produces the even distribution the overlap
/// argument above assumes.
class GpuScheduler {
public:
    /// `stream_count` should be at least 3 for copy-in/compute/copy-out overlap
    /// to have anywhere to go. Beyond about 8 the returns flatten: the copy
    /// engines saturate and additional streams only add scheduling overhead.
    explicit GpuScheduler(std::size_t stream_count, int device = 0);

    GpuScheduler(const GpuScheduler&) = delete;
    GpuScheduler& operator=(const GpuScheduler&) = delete;
    GpuScheduler(GpuScheduler&&) = delete;
    GpuScheduler& operator=(GpuScheduler&&) = delete;

    ~GpuScheduler();

    /// Next stream in round-robin order. Never blocks.
    [[nodiscard]] StreamLease acquire();

    /// Asynchronous host-to-device copy on the lease's stream.
    /// `host` must be pinned for this to overlap with anything.
    void copy_to_device(const StreamLease& lease, void* device, const void* host,
                        std::size_t bytes);

    void copy_to_host(const StreamLease& lease, void* host, const void* device,
                      std::size_t bytes);

    /// Makes `waiter` wait for `signaller`'s recorded completion event without
    /// blocking the host. This is the only cross-stream dependency mechanism
    /// used here.
    static void chain(const StreamLease& waiter, const StreamLease& signaller);

    void note_dispatch(const StreamLease& lease);

    /// Waits for every stream. Used only at shutdown — calling it per batch
    /// would serialise the pipeline.
    void synchronize_all();

    [[nodiscard]] SchedulerStats stats() const;
    [[nodiscard]] std::size_t stream_count() const noexcept { return streams_.size(); }
    [[nodiscard]] int device() const noexcept { return device_; }

private:
    int device_;
    std::vector<CudaStream> streams_;
    std::vector<CudaEvent> completion_events_;

    /// Relaxed: the counter only needs to distribute work, and no data is
    /// published through it. A stronger ordering would add a barrier per
    /// acquisition for no benefit.
    std::atomic<std::uint64_t> next_stream_{0};

    mutable std::mutex stats_mutex_;
    std::vector<StreamStats> stats_;
};

/// Device backend for `MemoryPool`.
///
/// This is the reason the pool is templated. `cudaMalloc` and `cudaFree`
/// synchronise the device: every call is a barrier that drains the pipeline the
/// scheduler above works to fill. Allocating once and reusing removes those
/// barriers from the steady-state path entirely.
class DeviceAllocatorBackend {
public:
    [[nodiscard]] void* allocate(std::size_t bytes) const {
        void* pointer = nullptr;
        if (cudaMalloc(&pointer, bytes) != cudaSuccess) {
            // Returning null rather than throwing lets MemoryPool decide
            // whether to trim its caches and retry, which is the recoverable
            // response to device OOM.
            return nullptr;
        }
        return pointer;
    }

    void deallocate(void* pointer, std::size_t /*bytes*/) const noexcept {
        static_cast<void>(cudaFree(pointer));
    }

    [[nodiscard]] static const char* name() { return "device"; }
};

/// Pinned-host backend, for staging buffers on the copy path.
class PinnedAllocatorBackend {
public:
    [[nodiscard]] void* allocate(std::size_t bytes) const {
        void* pointer = nullptr;
        if (cudaHostAlloc(&pointer, bytes, cudaHostAllocDefault) != cudaSuccess) {
            return nullptr;
        }
        return pointer;
    }

    void deallocate(void* pointer, std::size_t /*bytes*/) const noexcept {
        static_cast<void>(cudaFreeHost(pointer));
    }

    [[nodiscard]] static const char* name() { return "pinned"; }
};

}  // namespace cudaforge

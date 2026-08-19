#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

#include "cudaforge/cuda_error.cuh"

namespace cudaforge {

/// Owning CUDA stream.
///
/// Streams are the unit of ordering: work issued to one stream runs in issue
/// order, work in different streams may overlap. Leaking one leaks a driver
/// resource that outlives the process's use of it, and the usual cause is an
/// exception thrown between `cudaStreamCreate` and the matching destroy. RAII
/// removes that path entirely.
///
/// Created with `cudaStreamNonBlocking` rather than the default: a default
/// stream created with `cudaStreamCreate` implicitly synchronises with the
/// legacy default stream, so any library call that touches the default stream
/// would serialise every stream this scheduler owns — silently undoing the
/// overlap it exists to produce.
class CudaStream {
public:
    CudaStream() { CUDAFORGE_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }

    /// Priority-aware construction. Lower numeric value means higher priority;
    /// the valid range is device-specific and must come from
    /// `cudaDeviceGetStreamPriorityRange`.
    explicit CudaStream(int priority) {
        CUDAFORGE_CHECK(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, priority));
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    CudaStream(CudaStream&& other) noexcept : stream_(std::exchange(other.stream_, nullptr)) {}

    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            destroy();
            stream_ = std::exchange(other.stream_, nullptr);
        }
        return *this;
    }

    ~CudaStream() { destroy(); }

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }
    operator cudaStream_t() const noexcept {
        return stream_;
    }  // NOLINT(google-explicit-constructor)

    /// Blocks the calling host thread until every operation issued to this
    /// stream has completed. Scoped to one stream, so other streams keep
    /// running — unlike `cudaDeviceSynchronize`, which is a device-wide
    /// barrier and appears nowhere in this project's execution path.
    void synchronize() const { CUDAFORGE_CHECK(cudaStreamSynchronize(stream_)); }

    /// Non-blocking completion test, for a scheduler that wants to poll rather
    /// than park a host thread.
    [[nodiscard]] bool query() const {
        const cudaError_t status = cudaStreamQuery(stream_);
        if (status == cudaSuccess) {
            return true;
        }
        if (status == cudaErrorNotReady) {
            return false;
        }
        detail::throw_cuda_error(status, "cudaStreamQuery", __FILE__, __LINE__);
    }

private:
    void destroy() noexcept {
        if (stream_ != nullptr) {
            // Destructors must not throw. A failure here means the context is
            // already broken, and the next checked call will report it with
            // better information than a terminate() from here would.
            static_cast<void>(cudaStreamDestroy(stream_));
            stream_ = nullptr;
        }
    }

    cudaStream_t stream_ = nullptr;
};

/// Owning CUDA event.
///
/// Events serve two distinct purposes and the right creation flags differ:
///   - **Timing**: needs the default flags, which include timing support.
///   - **Ordering**: should pass `cudaEventDisableTiming`, which skips
///     recording a timestamp and makes record/wait measurably cheaper.
///
/// Using a timing-enabled event purely for ordering is a common and easy
/// inefficiency, so the constructor makes the choice explicit.
class CudaEvent {
public:
    enum class Purpose : unsigned { Timing, Ordering };

    explicit CudaEvent(Purpose purpose = Purpose::Ordering) {
        const unsigned flags =
            purpose == Purpose::Timing ? cudaEventDefault : cudaEventDisableTiming;
        CUDAFORGE_CHECK(cudaEventCreateWithFlags(&event_, flags));
        timing_enabled_ = purpose == Purpose::Timing;
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    CudaEvent(CudaEvent&& other) noexcept
        : event_(std::exchange(other.event_, nullptr)), timing_enabled_(other.timing_enabled_) {}

    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            destroy();
            event_ = std::exchange(other.event_, nullptr);
            timing_enabled_ = other.timing_enabled_;
        }
        return *this;
    }

    ~CudaEvent() { destroy(); }

    [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

    /// Marks a point in a stream's execution. The event completes when every
    /// operation issued to that stream before this call has completed.
    void record(cudaStream_t stream) { CUDAFORGE_CHECK(cudaEventRecord(event_, stream)); }

    void synchronize() const { CUDAFORGE_CHECK(cudaEventSynchronize(event_)); }

    [[nodiscard]] bool query() const {
        const cudaError_t status = cudaEventQuery(event_);
        if (status == cudaSuccess) {
            return true;
        }
        if (status == cudaErrorNotReady) {
            return false;
        }
        detail::throw_cuda_error(status, "cudaEventQuery", __FILE__, __LINE__);
    }

    /// Milliseconds between two recorded events, measured on the device.
    ///
    /// This is the correct way to time GPU work. A host-side timer around a
    /// kernel launch measures the launch, not the execution, because the launch
    /// is asynchronous — and adding a synchronise to fix that measures the
    /// synchronisation too.
    [[nodiscard]] static float elapsed_ms(const CudaEvent& start, const CudaEvent& stop) {
        float milliseconds = 0.0F;
        CUDAFORGE_CHECK(cudaEventElapsedTime(&milliseconds, start.event_, stop.event_));
        return milliseconds;
    }

    [[nodiscard]] bool timing_enabled() const noexcept { return timing_enabled_; }

private:
    void destroy() noexcept {
        if (event_ != nullptr) {
            static_cast<void>(cudaEventDestroy(event_));
            event_ = nullptr;
        }
    }

    cudaEvent_t event_ = nullptr;
    bool timing_enabled_ = false;
};

/// Makes `stream` wait until `event` completes, without blocking the host.
///
/// This is how cross-stream dependencies are expressed. The alternative —
/// synchronising the host and then issuing the dependent work — idles the GPU
/// for a full host round trip on every dependency.
inline void stream_wait_event(cudaStream_t stream, const CudaEvent& event) {
    CUDAFORGE_CHECK(cudaStreamWaitEvent(stream, event.get(), 0));
}

/// Owning device allocation, for buffers whose lifetime matches a scope.
/// Buffers reused across batches go through `MemoryPool` instead.
template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count_ > 0) {
            CUDAFORGE_CHECK(cudaMalloc(&data_, count_ * sizeof(T)));
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), count_(std::exchange(other.count_, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            destroy();
            data_ = std::exchange(other.data_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }

    ~DeviceBuffer() { destroy(); }

    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return count_ * sizeof(T); }
    [[nodiscard]] bool empty() const noexcept { return count_ == 0; }

    /// Bounds-checked because the alternative is a device-side buffer overrun.
    /// That does not fault at the copy: it corrupts whatever allocation happens
    /// to follow, and surfaces later as wrong numbers or an illegal access in
    /// an unrelated kernel.
    void copy_from_host(const T* host, std::size_t count, cudaStream_t stream) {
        if (count > count_) {
            throw std::out_of_range("DeviceBuffer::copy_from_host exceeds the allocation");
        }
        CUDAFORGE_CHECK(
            cudaMemcpyAsync(data_, host, count * sizeof(T), cudaMemcpyHostToDevice, stream));
    }

    void copy_to_host(T* host, std::size_t count, cudaStream_t stream) const {
        if (count > count_) {
            throw std::out_of_range("DeviceBuffer::copy_to_host exceeds the allocation");
        }
        CUDAFORGE_CHECK(
            cudaMemcpyAsync(host, data_, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
    }

    void fill_zero(cudaStream_t stream) {
        CUDAFORGE_CHECK(cudaMemsetAsync(data_, 0, bytes(), stream));
    }

private:
    void destroy() noexcept {
        if (data_ != nullptr) {
            static_cast<void>(cudaFree(data_));
            data_ = nullptr;
            count_ = 0;
        }
    }

    T* data_ = nullptr;
    std::size_t count_ = 0;
};

/// Page-locked host allocation.
///
/// `cudaMemcpyAsync` from pageable memory is not actually asynchronous: the
/// driver must stage through an internal pinned buffer, which serialises the
/// copy against the host thread and against the stream. Only pinned memory
/// allows a true DMA transfer that overlaps with kernel execution, which is the
/// entire premise of the scheduler's copy/compute pipelining.
///
/// Pinned memory is not free — it cannot be paged out, so over-allocating it
/// degrades the whole system. Staging buffers are therefore sized to the
/// maximum batch and reused, not allocated per request.
template<typename T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;

    explicit PinnedBuffer(std::size_t count) : count_(count) {
        if (count_ > 0) {
            CUDAFORGE_CHECK(cudaHostAlloc(&data_, count_ * sizeof(T), cudaHostAllocDefault));
        }
    }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    PinnedBuffer(PinnedBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), count_(std::exchange(other.count_, 0)) {}

    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
        if (this != &other) {
            destroy();
            data_ = std::exchange(other.data_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }

    ~PinnedBuffer() { destroy(); }

    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return count_ * sizeof(T); }

    [[nodiscard]] T& operator[](std::size_t index) noexcept { return data_[index]; }
    [[nodiscard]] const T& operator[](std::size_t index) const noexcept { return data_[index]; }

private:
    void destroy() noexcept {
        if (data_ != nullptr) {
            static_cast<void>(cudaFreeHost(data_));
            data_ = nullptr;
            count_ = 0;
        }
    }

    T* data_ = nullptr;
    std::size_t count_ = 0;
};

}  // namespace cudaforge

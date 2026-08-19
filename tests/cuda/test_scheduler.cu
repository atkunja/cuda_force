#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <thread>
#include <vector>

#include "cudaforge/gpu_scheduler.cuh"
#include "cudaforge/memory_pool.hpp"
#include "cudaforge/reduction.cuh"

using namespace cudaforge;

TEST_CASE("the scheduler rejects a zero stream count", "[cuda][scheduler]") {
    REQUIRE_THROWS_AS(GpuScheduler(0), std::invalid_argument);
}

TEST_CASE("stream assignment is round-robin", "[cuda][scheduler]") {
    // Round-robin rather than idle-polling: querying each stream costs a driver
    // call per acquisition and tends to pile short batches onto whichever
    // stream finished first, defeating the even distribution overlap needs.
    GpuScheduler scheduler(4);
    for (std::size_t expected = 0; expected < 12; ++expected) {
        REQUIRE(scheduler.acquire().index() == expected % 4);
    }
}

TEST_CASE("leases hand out distinct streams", "[cuda][scheduler]") {
    GpuScheduler scheduler(3);
    const auto first = scheduler.acquire();
    const auto second = scheduler.acquire();
    const auto third = scheduler.acquire();

    REQUIRE(first.stream() != second.stream());
    REQUIRE(second.stream() != third.stream());
    REQUIRE(scheduler.acquire().stream() == first.stream());
}

TEST_CASE("an event records and completes", "[cuda][scheduler]") {
    GpuScheduler scheduler(2);
    const auto lease = scheduler.acquire();

    DeviceBuffer<float> buffer(1024);
    buffer.fill_zero(lease.stream());
    lease.record_completion();
    lease.synchronize();

    REQUIRE(lease.complete());
    REQUIRE(lease.completion_event().query());
}

TEST_CASE("ordering events do not support timing", "[cuda][scheduler]") {
    // Timing-enabled events write a timestamp on every record. Using one purely
    // for ordering is a common and avoidable inefficiency.
    const CudaEvent ordering(CudaEvent::Purpose::Ordering);
    const CudaEvent timing(CudaEvent::Purpose::Timing);
    REQUIRE_FALSE(ordering.timing_enabled());
    REQUIRE(timing.timing_enabled());
}

TEST_CASE("device timing reports a positive interval", "[cuda][scheduler]") {
    // Timing GPU work with a host clock measures the launch, not the execution,
    // because the launch is asynchronous. CUDA events measure on the device.
    CudaStream stream;
    CudaEvent start(CudaEvent::Purpose::Timing);
    CudaEvent stop(CudaEvent::Purpose::Timing);

    DeviceBuffer<float> input(1 << 20);
    DeviceBuffer<float> output(1);
    input.fill_zero(stream);
    output.fill_zero(stream);

    start.record(stream);
    launch_reduce_sum(input.data(), output.data(), input.size(),
                      ReductionKernel::WarpOptimised, stream);
    stop.record(stream);
    stream.synchronize();

    REQUIRE(CudaEvent::elapsed_ms(start, stop) > 0.0F);
}

TEST_CASE("chaining makes one stream wait for another", "[cuda][scheduler]") {
    GpuScheduler scheduler(2);
    const auto producer = scheduler.acquire();
    const auto consumer = scheduler.acquire();

    DeviceBuffer<float> buffer(1 << 20);
    buffer.fill_zero(producer.stream());
    producer.record_completion();

    // Without this the consumer's work could begin before the producer's write
    // lands. Expressing the dependency with an event keeps the host free,
    // unlike synchronising and then issuing.
    GpuScheduler::chain(consumer, producer);

    DeviceBuffer<float> output(1);
    output.fill_zero(consumer.stream());
    launch_reduce_sum(buffer.data(), output.data(), buffer.size(),
                      ReductionKernel::WarpOptimised, consumer.stream());
    consumer.synchronize();

    float result = -1.0F;
    output.copy_to_host(&result, 1, consumer.stream());
    consumer.synchronize();
    REQUIRE(result == 0.0F);
}

TEST_CASE("copy statistics are attributed per stream", "[cuda][scheduler]") {
    GpuScheduler scheduler(2);
    PinnedBuffer<float> host(256);
    DeviceBuffer<float> device(256);

    const auto lease = scheduler.acquire();
    scheduler.copy_to_device(lease, device.data(), host.data(), device.bytes());
    scheduler.copy_to_host(lease, host.data(), device.data(), device.bytes());
    scheduler.note_dispatch(lease);
    lease.synchronize();

    const auto stats = scheduler.stats();
    REQUIRE(stats.stream_count == 2);
    REQUIRE(stats.total_bytes_to_device == device.bytes());
    REQUIRE(stats.total_bytes_to_host == device.bytes());
    REQUIRE(stats.total_dispatches == 1);
    REQUIRE(stats.per_stream[lease.index()].batches_dispatched == 1);
}

TEST_CASE("the device memory pool reuses allocations", "[cuda][pool]") {
    // Each avoided allocation is a cudaMalloc that would have synchronised the
    // device, draining the pipeline the scheduler works to fill.
    MemoryPool<DeviceAllocatorBackend> pool;

    void* first = pool.allocate(1 << 20);
    pool.deallocate(first);
    void* second = pool.allocate(1 << 20);

    REQUIRE(second == first);
    REQUIRE(pool.stats().backend_allocations == 1);
    REQUIRE(pool.stats().reuse_count == 1);
    pool.deallocate(second);
}

TEST_CASE("pooled device memory is usable by kernels", "[cuda][pool]") {
    MemoryPool<DeviceAllocatorBackend> pool;
    constexpr std::size_t kCount = 1 << 16;

    auto* input = static_cast<float*>(pool.allocate(kCount * sizeof(float)));
    auto* output = static_cast<float*>(pool.allocate(sizeof(float)));

    CudaStream stream;
    CUDAFORGE_CHECK(cudaMemsetAsync(input, 0, kCount * sizeof(float), stream));
    CUDAFORGE_CHECK(cudaMemsetAsync(output, 0, sizeof(float), stream));
    launch_reduce_sum(input, output, kCount, ReductionKernel::WarpOptimised, stream);

    float result = -1.0F;
    CUDAFORGE_CHECK(
        cudaMemcpyAsync(&result, output, sizeof(float), cudaMemcpyDeviceToHost, stream));
    stream.synchronize();

    REQUIRE(result == 0.0F);
    pool.deallocate(input);
    pool.deallocate(output);
}

TEST_CASE("the pinned pool reports its backend", "[cuda][pool]") {
    REQUIRE(std::string(MemoryPool<PinnedAllocatorBackend>::backend_name()) == "pinned");
    REQUIRE(std::string(MemoryPool<DeviceAllocatorBackend>::backend_name()) == "device");
}

TEST_CASE("a cuda error carries its status code", "[cuda][error]") {
    // Branching on the specific failure matters: device OOM is recoverable by
    // trimming a cache and retrying, an illegal address is not.
    const CudaError oom(cudaErrorMemoryAllocation, "test");
    REQUIRE(oom.status() == cudaErrorMemoryAllocation);
    REQUIRE_FALSE(oom.is_sticky());

    const CudaError fatal(cudaErrorIllegalAddress, "test");
    REQUIRE(fatal.is_sticky());
}

TEST_CASE("concurrent workers can share the scheduler", "[cuda][scheduler][stress]") {
    constexpr int kThreads = 8;
    constexpr int kPerThread = 50;

    GpuScheduler scheduler(4);
    std::atomic<std::uint64_t> failures{0};

    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([&] {
            for (int i = 0; i < kPerThread; ++i) {
                const auto lease = scheduler.acquire();
                DeviceBuffer<float> buffer(1024);
                DeviceBuffer<float> result(1);
                buffer.fill_zero(lease.stream());
                result.fill_zero(lease.stream());
                launch_reduce_sum(buffer.data(), result.data(), buffer.size(),
                                  ReductionKernel::WarpOptimised, lease.stream());
                scheduler.note_dispatch(lease);
                lease.synchronize();

                float value = -1.0F;
                result.copy_to_host(&value, 1, lease.stream());
                lease.synchronize();
                if (value != 0.0F) {
                    failures.fetch_add(1);
                }
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }

    REQUIRE(failures.load() == 0);
    REQUIRE(scheduler.stats().total_dispatches == kThreads * kPerThread);
}

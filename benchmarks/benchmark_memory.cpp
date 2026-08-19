// Measures what the caching pool buys over calling the backend directly.
//
// On the host backend the saving is modest, because malloc already caches.
// The number to look at is `reuse_rate` and the ratio of backend calls to
// allocations: those carry over unchanged to the device backend, where each
// avoided call is a `cudaMalloc` that would have synchronised the device.
// That is the actual argument for the pool, and it cannot be measured here.

#include "bench_common.hpp"
#include "cudaforge/memory_pool.hpp"

#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

using cudaforge::HostAllocatorBackend;
using cudaforge::HostMemoryPool;
using namespace cudaforge::bench;

namespace {

/// Shapes an inference workload actually allocates: activations for a few
/// batch sizes, repeated. The repetition is the point — it is what a pool
/// exploits and what a fresh malloc per batch cannot.
const std::vector<std::size_t> kWorkloadSizes = {
    4 * 1024, 64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024,
};

double time_pool(std::size_t iterations, std::size_t threads, HostMemoryPool& pool) {
    Timer timer;
    timer.start();

    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (std::size_t t = 0; t < threads; ++t) {
        workers.emplace_back([&, t] {
            for (std::size_t i = 0; i < iterations; ++i) {
                const std::size_t bytes = kWorkloadSizes[(i + t) % kWorkloadSizes.size()];
                void* block = pool.allocate(bytes);
                keep(block);
                pool.deallocate(block);
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
    return timer.elapsed_seconds();
}

double time_backend(std::size_t iterations, std::size_t threads) {
    HostAllocatorBackend backend;
    Timer timer;
    timer.start();

    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (std::size_t t = 0; t < threads; ++t) {
        workers.emplace_back([&, t] {
            for (std::size_t i = 0; i < iterations; ++i) {
                const std::size_t bytes = kWorkloadSizes[(i + t) % kWorkloadSizes.size()];
                void* block = backend.allocate(bytes);
                keep(block);
                backend.deallocate(block, bytes);
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
    return timer.elapsed_seconds();
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t iterations = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 20'000;
    const std::vector<std::size_t> thread_counts = {1, 2, 4, 8};

    JsonWriter writer(std::cout);
    writer.begin_object();
    writer.field("benchmark", std::string("memory_pool"));
    writer.field("backend", std::string(HostMemoryPool::backend_name()));
    writer.field("note",
                 std::string("host backend only; the device backend is where avoided allocator "
                             "calls translate into avoided device synchronisation"));
    writer.field("iterations_per_thread", static_cast<std::uint64_t>(iterations));
    writer.begin_array("cases");

    for (std::size_t threads : thread_counts) {
        HostMemoryPool pool;
        // Warm up so the steady-state path is what gets timed, not the initial
        // population of the free lists.
        (void)time_pool(kWorkloadSizes.size() * 4, threads, pool);

        const double pool_seconds = time_pool(iterations, threads, pool);
        const double backend_seconds = time_backend(iterations, threads);
        const auto stats = pool.stats();

        writer.array_element_begin();
        writer.field("threads", static_cast<std::uint64_t>(threads));
        writer.field("pool_seconds", pool_seconds);
        writer.field("raw_backend_seconds", backend_seconds);
        writer.field("pool_allocations", stats.allocation_count);
        writer.field("backend_allocations", stats.backend_allocations);
        writer.field("reuse_rate", stats.reuse_rate());
        writer.field("peak_bytes_in_use", static_cast<std::uint64_t>(stats.peak_bytes_in_use));
        writer.field("bytes_reserved", static_cast<std::uint64_t>(stats.bytes_reserved));
        writer.array_element_end();
    }

    writer.end_array();
    writer.end_object();
    writer.finish();
    return 0;
}

// Measures how the bounded queue behaves as producer and consumer counts vary.
//
// The interesting quantity is not raw throughput but where it stops scaling.
// A single mutex serialises every push and pop, so beyond some thread count
// the queue becomes the bottleneck and adding threads makes things worse.
// Locating that point is what tells you whether the runtime needs sharding.

#include "bench_common.hpp"
#include "cudaforge/concurrent_queue.hpp"

#include <atomic>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

using cudaforge::ConcurrentQueue;
using cudaforge::QueueStatus;
using namespace cudaforge::bench;

namespace {

struct Result {
    std::size_t producers;
    std::size_t consumers;
    std::size_t capacity;
    std::uint64_t items;
    double seconds;
    double items_per_second;
};

Result run_case(std::size_t producers, std::size_t consumers, std::size_t capacity,
                std::uint64_t items_per_producer) {
    ConcurrentQueue<std::uint64_t> queue(capacity);
    std::atomic<std::uint64_t> consumed{0};
    std::atomic<bool> go{false};

    std::vector<std::thread> consumer_threads;
    consumer_threads.reserve(consumers);
    for (std::size_t c = 0; c < consumers; ++c) {
        consumer_threads.emplace_back([&] {
            std::uint64_t value = 0;
            std::uint64_t local = 0;
            while (queue.pop(value) == QueueStatus::Ok) {
                keep(value);
                ++local;
            }
            consumed.fetch_add(local, std::memory_order_relaxed);
        });
    }

    std::vector<std::thread> producer_threads;
    producer_threads.reserve(producers);
    for (std::size_t p = 0; p < producers; ++p) {
        producer_threads.emplace_back([&] {
            // Spin on the start flag so every producer begins at the same
            // instant; staggered starts understate contention.
            while (!go.load(std::memory_order_acquire)) {
            }
            for (std::uint64_t i = 0; i < items_per_producer; ++i) {
                queue.push(i);
            }
        });
    }

    Timer timer;
    timer.start();
    go.store(true, std::memory_order_release);

    for (std::thread& thread : producer_threads) {
        thread.join();
    }
    queue.shutdown();
    for (std::thread& thread : consumer_threads) {
        thread.join();
    }
    const double seconds = timer.elapsed_seconds();

    const std::uint64_t total = consumed.load();
    return Result{producers, consumers, capacity, total, seconds,
                  seconds > 0.0 ? static_cast<double>(total) / seconds : 0.0};
}

}  // namespace

int main(int argc, char** argv) {
    const std::uint64_t items_per_producer =
        argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 200'000;

    const std::vector<std::pair<std::size_t, std::size_t>> shapes = {
        {1, 1}, {2, 1}, {1, 2}, {2, 2}, {4, 2}, {4, 4}, {8, 4}, {8, 8},
    };
    const std::vector<std::size_t> capacities = {16, 256, 4096};

    JsonWriter writer(std::cout);
    writer.begin_object();
    writer.field("benchmark", std::string("concurrent_queue"));
    writer.field("hardware_threads",
                 static_cast<std::uint64_t>(std::thread::hardware_concurrency()));
    writer.field("items_per_producer", items_per_producer);
    writer.begin_array("cases");

    for (std::size_t capacity : capacities) {
        for (const auto& [producers, consumers] : shapes) {
            const Result result = run_case(producers, consumers, capacity, items_per_producer);
            writer.array_element_begin();
            writer.field("producers", static_cast<std::uint64_t>(result.producers));
            writer.field("consumers", static_cast<std::uint64_t>(result.consumers));
            writer.field("capacity", static_cast<std::uint64_t>(result.capacity));
            writer.field("items", result.items);
            writer.field("seconds", result.seconds);
            writer.field("items_per_second", result.items_per_second);
            writer.array_element_end();
        }
    }

    writer.end_array();
    writer.end_object();
    writer.finish();
    return 0;
}

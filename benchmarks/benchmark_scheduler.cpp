// Measures the dynamic batcher under a controlled arrival process.
//
// The two quantities that matter are in tension: average batch size (which
// drives GPU efficiency) and queue delay (which is the batcher's contribution
// to tail latency). Sweeping arrival rate against max_wait shows where the
// configured wait stops buying batch size and starts only buying latency.
//
// Execution is simulated with a sleep rather than a real model, so the numbers
// here describe the *scheduler*, not inference throughput. That separation is
// deliberate: it keeps the benchmark runnable on a machine with no GPU and
// isolates the variable under study.

#include "bench_common.hpp"
#include "cudaforge/dynamic_batcher.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using cudaforge::Batch;
using cudaforge::DynamicBatcher;
using cudaforge::Metrics;
using cudaforge::Request;
using cudaforge::RuntimeConfig;
using namespace cudaforge::bench;
using namespace std::chrono_literals;

namespace {

/// Per-row service cost of the simulated model, plus a fixed launch overhead.
/// The fixed component is what batching amortises; without it, batching would
/// show no benefit and the benchmark would be measuring nothing.
constexpr auto kFixedBatchCost = 400us;
constexpr auto kPerRequestCost = 25us;

struct CaseResult {
    std::size_t producers;
    std::size_t max_batch;
    std::uint64_t max_wait_us;
    std::uint64_t requests;
    double seconds;
    double requests_per_second;
    double average_batch_size;
    double queue_delay_p50_ms;
    double queue_delay_p99_ms;
    double timeout_closure_fraction;
};

CaseResult run_case(std::size_t producers, std::size_t requests_per_producer,
                    std::size_t max_batch, std::chrono::microseconds max_wait) {
    RuntimeConfig config;
    config.max_batch_size = max_batch;
    config.max_wait = max_wait;
    config.queue_capacity = std::max<std::size_t>(1024, max_batch * 4);

    auto metrics = std::make_shared<Metrics>();
    std::atomic<std::uint64_t> executed{0};

    Timer timer;
    {
        DynamicBatcher batcher(
            config,
            [&](Batch&& batch) {
                std::this_thread::sleep_for(kFixedBatchCost +
                                            kPerRequestCost * batch.size());
                executed.fetch_add(batch.size(), std::memory_order_relaxed);
            },
            metrics);

        std::atomic<bool> go{false};
        std::vector<std::thread> threads;
        threads.reserve(producers);
        for (std::size_t p = 0; p < producers; ++p) {
            threads.emplace_back([&, p] {
                while (!go.load(std::memory_order_acquire)) {
                }
                for (std::size_t i = 0; i < requests_per_producer; ++i) {
                    batcher.submit(Request(p * requests_per_producer + i, "benchmark", {}));
                }
            });
        }

        timer.start();
        go.store(true, std::memory_order_release);
        for (std::thread& thread : threads) {
            thread.join();
        }
    }  // shutdown drains the remainder before the timer is read

    const double seconds = timer.elapsed_seconds();
    const auto snapshot = metrics->snapshot();
    const double closures = static_cast<double>(snapshot.batches_processed);

    return CaseResult{
        producers,
        max_batch,
        static_cast<std::uint64_t>(max_wait.count()),
        executed.load(),
        seconds,
        seconds > 0.0 ? static_cast<double>(executed.load()) / seconds : 0.0,
        snapshot.average_batch_size,
        static_cast<double>(snapshot.queue_delay_p50_ns) / 1e6,
        static_cast<double>(snapshot.queue_delay_p99_ns) / 1e6,
        closures > 0.0 ? static_cast<double>(snapshot.batches_closed_by_timeout) / closures : 0.0,
    };
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t per_producer = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 2000;

    const std::vector<std::size_t> producer_counts = {1, 2, 4, 8};
    const std::vector<std::size_t> batch_sizes = {1, 4, 16, 32};
    const std::vector<std::chrono::microseconds> waits = {500us, 2000us, 5000us};

    JsonWriter writer(std::cout);
    writer.begin_object();
    writer.field("benchmark", std::string("dynamic_batcher"));
    writer.field("note",
                 std::string("execution is simulated with a sleep; these measure the scheduler, "
                             "not model throughput"));
    writer.field("fixed_batch_cost_us", static_cast<std::uint64_t>(kFixedBatchCost.count()));
    writer.field("per_request_cost_us", static_cast<std::uint64_t>(kPerRequestCost.count()));
    writer.field("hardware_threads",
                 static_cast<std::uint64_t>(std::thread::hardware_concurrency()));
    writer.begin_array("cases");

    for (std::size_t producers : producer_counts) {
        for (std::size_t max_batch : batch_sizes) {
            for (std::chrono::microseconds wait : waits) {
                const CaseResult result = run_case(producers, per_producer, max_batch, wait);
                writer.array_element_begin();
                writer.field("producers", static_cast<std::uint64_t>(result.producers));
                writer.field("max_batch_size", static_cast<std::uint64_t>(result.max_batch));
                writer.field("max_wait_us", result.max_wait_us);
                writer.field("requests", result.requests);
                writer.field("seconds", result.seconds);
                writer.field("requests_per_second", result.requests_per_second);
                writer.field("average_batch_size", result.average_batch_size);
                writer.field("queue_delay_p50_ms", result.queue_delay_p50_ms);
                writer.field("queue_delay_p99_ms", result.queue_delay_p99_ms);
                writer.field("timeout_closure_fraction", result.timeout_closure_fraction);
                writer.array_element_end();
            }
        }
    }

    writer.end_array();
    writer.end_object();
    writer.finish();
    return 0;
}

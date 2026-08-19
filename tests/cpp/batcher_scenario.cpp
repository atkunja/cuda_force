// Runs a batching scenario and reports what happened, as JSON.
//
// The documentation claims the C++ and Python batchers implement the same
// policy. Nothing verified that: each is tested against its own expectations,
// which is exactly how two implementations drift apart while both suites stay
// green.
//
// This binary exists so a Python test can run the identical scenario through
// both and compare. It is not a test itself — it is the C++ half of one, which
// is why it lives beside the tests but is not linked into the test binary.
//
//   batcher_scenario <max_batch> <max_wait_us> <producers> <per_producer> <gap_us>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "cudaforge/dynamic_batcher.hpp"

using namespace cudaforge;
using namespace std::chrono_literals;

namespace {

struct Observed {
    std::vector<std::size_t> sizes;
    std::vector<int> triggers;  // 0 = MaxSize, 1 = Timeout, 2 = Shutdown
    std::mutex mutex;
};

int trigger_code(BatchTrigger trigger) {
    switch (trigger) {
        case BatchTrigger::MaxSize:
            return 0;
        case BatchTrigger::Timeout:
            return 1;
        case BatchTrigger::Shutdown:
            return 2;
    }
    return -1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "usage: batcher_scenario <max_batch> <max_wait_us> <producers> "
                     "<per_producer> <gap_us>\n";
        return 2;
    }

    const auto max_batch = static_cast<std::size_t>(std::strtoul(argv[1], nullptr, 10));
    const auto max_wait_us = std::strtol(argv[2], nullptr, 10);
    const auto producers = static_cast<std::size_t>(std::strtoul(argv[3], nullptr, 10));
    const auto per_producer = static_cast<std::size_t>(std::strtoul(argv[4], nullptr, 10));
    const auto gap_us = std::strtol(argv[5], nullptr, 10);

    RuntimeConfig config;
    config.max_batch_size = max_batch;
    config.max_wait = std::chrono::microseconds(max_wait_us);
    config.queue_capacity = std::max<std::size_t>(1024, max_batch * 4);

    auto observed = std::make_shared<Observed>();
    auto metrics = std::make_shared<Metrics>();

    {
        DynamicBatcher batcher(
            config,
            [observed](Batch&& batch) {
                std::lock_guard lock(observed->mutex);
                observed->sizes.push_back(batch.size());
                observed->triggers.push_back(trigger_code(batch.trigger));
            },
            metrics);

        std::vector<std::thread> threads;
        threads.reserve(producers);
        for (std::size_t p = 0; p < producers; ++p) {
            threads.emplace_back([&, p] {
                for (std::size_t i = 0; i < per_producer; ++i) {
                    batcher.submit(Request(p * per_producer + i, "scenario", {}));
                    if (gap_us > 0) {
                        std::this_thread::sleep_for(std::chrono::microseconds(gap_us));
                    }
                }
            });
        }
        for (std::thread& thread : threads) {
            thread.join();
        }
    }  // shutdown drains before the results are read

    const auto snapshot = metrics->snapshot();

    std::lock_guard lock(observed->mutex);
    std::size_t total = 0;
    std::size_t largest = 0;
    for (std::size_t size : observed->sizes) {
        total += size;
        largest = std::max(largest, size);
    }

    std::cout << "{\n";
    std::cout << "  \"implementation\": \"cpp\",\n";
    std::cout << "  \"max_batch_size\": " << max_batch << ",\n";
    std::cout << "  \"max_wait_us\": " << max_wait_us << ",\n";
    std::cout << "  \"submitted\": " << producers * per_producer << ",\n";
    std::cout << "  \"batched\": " << total << ",\n";
    std::cout << "  \"batches\": " << observed->sizes.size() << ",\n";
    std::cout << "  \"largest_batch\": " << largest << ",\n";
    std::cout << "  \"closed_by_size\": " << snapshot.batches_closed_by_size << ",\n";
    std::cout << "  \"closed_by_timeout\": " << snapshot.batches_closed_by_timeout << ",\n";
    std::cout << "  \"sizes\": [";
    for (std::size_t i = 0; i < observed->sizes.size(); ++i) {
        std::cout << (i ? ", " : "") << observed->sizes[i];
    }
    std::cout << "]\n}\n";
    return 0;
}

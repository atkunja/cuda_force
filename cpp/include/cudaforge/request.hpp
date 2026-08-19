#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace cudaforge {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
using Duration = std::chrono::nanoseconds;

/// Monotonically increasing identifier assigned at ingress.
using RequestId = std::uint64_t;

/// Generation knobs carried per request. Requests batched together must agree
/// on nothing except tensor-shape compatibility; sampling is applied per row,
/// so these may differ within a batch.
struct GenerationParams {
    std::uint32_t max_new_tokens = 64;
    float temperature = 1.0F;
    float top_p = 1.0F;
    std::uint32_t top_k = 0;  // 0 disables top-k
    std::uint64_t seed = 0;   // 0 means "use the engine's global seed"
};

/// A unit of work travelling through queue -> batcher -> scheduler.
///
/// Timestamps are recorded at each hop rather than derived, because the gaps
/// between them are exactly the quantities the metrics layer reports:
/// `enqueued -> dequeued` is queue delay, `dequeued -> completed` is service
/// time, and the sum is end-to-end latency.
struct Request {
    RequestId id = 0;
    std::string prompt;
    GenerationParams params;

    TimePoint enqueued{};
    TimePoint dequeued{};

    /// Point past which this request is worthless. A default-constructed value
    /// means no deadline, which is why `expired()` tests for it explicitly
    /// rather than comparing against the epoch.
    TimePoint deadline{};

    Request() = default;

    Request(RequestId request_id, std::string request_prompt, GenerationParams generation)
        : id(request_id),
          prompt(std::move(request_prompt)),
          params(generation),
          enqueued(Clock::now()) {}

    [[nodiscard]] Duration queue_delay() const noexcept {
        return std::chrono::duration_cast<Duration>(dequeued - enqueued);
    }

    [[nodiscard]] bool has_deadline() const noexcept { return deadline != TimePoint{}; }

    /// True once the deadline has passed.
    ///
    /// Tested at dequeue rather than at admission. Running a request whose
    /// client has already given up spends capacity that the requests still
    /// being waited on need — under overload that is exactly backwards, and it
    /// deepens the backlog that caused the timeouts in the first place.
    [[nodiscard]] bool expired(TimePoint now = Clock::now()) const noexcept {
        return has_deadline() && now >= deadline;
    }
};

/// Result of executing a request. `error` being non-empty means generation
/// failed; `text` is then unspecified.
struct Response {
    RequestId id = 0;
    std::string text;
    std::string error;

    std::uint32_t prompt_tokens = 0;
    std::uint32_t generated_tokens = 0;

    Duration queue_time{};
    Duration inference_time{};

    [[nodiscard]] Duration total_latency() const noexcept { return queue_time + inference_time; }
    [[nodiscard]] bool ok() const noexcept { return error.empty(); }
};

/// Why the batcher stopped accumulating. Recorded because a batcher that only
/// ever closes on `Timeout` is starved and one that only ever closes on
/// `MaxSize` is saturated — both are actionable, and indistinguishable from
/// batch size alone.
enum class BatchTrigger : std::uint8_t {
    MaxSize,   ///< reached max_batch_size
    Timeout,   ///< oldest request hit max_wait
    Shutdown,  ///< drained during graceful shutdown
};

struct Batch {
    std::vector<Request> requests;
    BatchTrigger trigger = BatchTrigger::MaxSize;
    TimePoint formed{};

    [[nodiscard]] std::size_t size() const noexcept { return requests.size(); }
    [[nodiscard]] bool empty() const noexcept { return requests.empty(); }
};

}  // namespace cudaforge

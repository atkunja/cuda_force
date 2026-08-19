#include "cudaforge/request.hpp"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <utility>

using cudaforge::Batch;
using cudaforge::BatchTrigger;
using cudaforge::Clock;
using cudaforge::Duration;
using cudaforge::GenerationParams;
using cudaforge::Request;
using cudaforge::Response;
using namespace std::chrono_literals;

TEST_CASE("a constructed request stamps its arrival time", "[request]") {
    const auto before = Clock::now();
    const Request request(1, "prompt", {});
    const auto after = Clock::now();

    REQUIRE(request.id == 1);
    REQUIRE(request.prompt == "prompt");
    REQUIRE(request.enqueued >= before);
    REQUIRE(request.enqueued <= after);
}

TEST_CASE("queue delay is the gap between the two stamps", "[request]") {
    Request request(1, "prompt", {});
    std::this_thread::sleep_for(5ms);
    request.dequeued = Clock::now();

    // Only a lower bound is asserted. An upper bound would make this flaky on a
    // loaded machine, where the scheduler can delay the wake arbitrarily.
    REQUIRE(request.queue_delay() >= std::chrono::milliseconds(4));
}

TEST_CASE("the prompt is moved, not copied", "[request]") {
    std::string prompt(4096, 'x');
    const char* original = prompt.data();

    const Request request(1, std::move(prompt), {});

    // A copy here would be a per-request allocation on the ingress path.
    REQUIRE(request.prompt.data() == original);
}

TEST_CASE("generation defaults are sane", "[request]") {
    const GenerationParams params;
    REQUIRE(params.max_new_tokens > 0);
    REQUIRE(params.temperature == 1.0F);
    REQUIRE(params.top_p == 1.0F);
    REQUIRE(params.top_k == 0);  // 0 disables top-k
    REQUIRE(params.seed == 0);   // 0 means "use the engine's global seed"
}

TEST_CASE("a response reports total latency as the sum of its parts", "[response]") {
    Response response;
    response.queue_time = std::chrono::milliseconds(3);
    response.inference_time = std::chrono::milliseconds(42);

    REQUIRE(response.total_latency() == std::chrono::milliseconds(45));
}

TEST_CASE("a response is ok until an error is set", "[response]") {
    Response response;
    REQUIRE(response.ok());

    response.error = "model failure";
    REQUIRE_FALSE(response.ok());
}

TEST_CASE("an empty batch reports zero size", "[batch]") {
    const Batch batch;
    REQUIRE(batch.empty());
    REQUIRE(batch.size() == 0);
    REQUIRE(batch.trigger == BatchTrigger::MaxSize);
}

TEST_CASE("a batch reports the number of requests it holds", "[batch]") {
    Batch batch;
    for (std::uint64_t i = 0; i < 5; ++i) {
        batch.requests.emplace_back(i, "prompt", GenerationParams{});
    }
    REQUIRE(batch.size() == 5);
    REQUIRE_FALSE(batch.empty());
}

TEST_CASE("requests carry independent generation settings", "[batch]") {
    // Rows in a batch need only be shape-compatible; sampling is per row, so
    // members may disagree on temperature and token budget.
    Batch batch;
    batch.requests.emplace_back(1, "a", GenerationParams{.max_new_tokens = 8});
    batch.requests.emplace_back(2, "b", GenerationParams{.max_new_tokens = 64});

    REQUIRE(batch.requests[0].params.max_new_tokens == 8);
    REQUIRE(batch.requests[1].params.max_new_tokens == 64);
}

TEST_CASE("a default request has a zero queue delay", "[request]") {
    // Both stamps are default-constructed, so the difference is exactly zero
    // rather than an arbitrary large value.
    const Request request;
    REQUIRE(request.queue_delay() == Duration::zero());
}

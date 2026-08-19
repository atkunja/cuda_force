#include "cudaforge/metrics.hpp"

#include <sstream>

namespace cudaforge {
namespace {

/// Nanoseconds are the recording unit; milliseconds are what a human reads.
/// The conversion happens at the reporting boundary so no precision is lost in
/// the histogram itself.
double to_ms(std::uint64_t nanos) { return static_cast<double>(nanos) / 1e6; }

}  // namespace

std::string to_json(const MetricsSnapshot& snapshot) {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(3);

    out << "{\n";
    out << "  \"requests_received\": " << snapshot.requests_received << ",\n";
    out << "  \"requests_completed\": " << snapshot.requests_completed << ",\n";
    out << "  \"requests_failed\": " << snapshot.requests_failed << ",\n";
    out << "  \"requests_rejected\": " << snapshot.requests_rejected << ",\n";
    out << "  \"batches_processed\": " << snapshot.batches_processed << ",\n";
    out << "  \"batches_closed_by_size\": " << snapshot.batches_closed_by_size << ",\n";
    out << "  \"batches_closed_by_timeout\": " << snapshot.batches_closed_by_timeout << ",\n";
    out << "  \"average_batch_size\": " << snapshot.average_batch_size << ",\n";
    out << "  \"queue_depth\": " << snapshot.queue_depth << ",\n";
    out << "  \"tokens_generated\": " << snapshot.tokens_generated << ",\n";
    out << "  \"uptime_seconds\": " << snapshot.uptime_seconds << ",\n";
    out << "  \"requests_per_second\": " << snapshot.requests_per_second << ",\n";
    out << "  \"tokens_per_second\": " << snapshot.tokens_per_second << ",\n";
    out << "  \"queue_delay_ms\": {\n";
    out << "    \"p50\": " << to_ms(snapshot.queue_delay_p50_ns) << ",\n";
    out << "    \"p95\": " << to_ms(snapshot.queue_delay_p95_ns) << ",\n";
    out << "    \"p99\": " << to_ms(snapshot.queue_delay_p99_ns) << "\n";
    out << "  },\n";
    out << "  \"latency_ms\": {\n";
    out << "    \"mean\": " << snapshot.latency_mean_ns / 1e6 << ",\n";
    out << "    \"p50\": " << to_ms(snapshot.latency_p50_ns) << ",\n";
    out << "    \"p95\": " << to_ms(snapshot.latency_p95_ns) << ",\n";
    out << "    \"p99\": " << to_ms(snapshot.latency_p99_ns) << ",\n";
    out << "    \"max\": " << to_ms(snapshot.latency_max_ns) << "\n";
    out << "  }\n";
    out << "}";
    return out.str();
}

}  // namespace cudaforge

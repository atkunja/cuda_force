// Validates the latency histogram's accuracy claim against exact percentiles.
//
// LatencyHistogram trades exactness for O(1) memory, and documents a worst-case
// relative error of 1/kSubBuckets — 6.25% at 16 sub-buckets. That is a claim,
// and claims about numerical behaviour should be measured rather than asserted.
//
// This runs the same samples through both the bucketed histogram and an exact
// sorted vector, and reports the observed error across several distributions.
// The distributions matter: a bucketed histogram's error depends on where the
// samples land relative to bucket boundaries, so a single well-behaved input
// would flatter it.
//
// Also reports record throughput, since a metrics system that degrades the
// thing it measures is not useful.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "bench_common.hpp"
#include "cudaforge/latency_histogram.hpp"

using cudaforge::LatencyHistogram;
using namespace cudaforge::bench;

namespace {

struct Distribution {
    std::string name;
    std::vector<std::uint64_t> samples;
};

/// Seeded explicitly so a run is reproducible; an unseeded generator would make
/// a marginal result appear and disappear between runs.
std::vector<Distribution> build_distributions(std::size_t count) {
    std::mt19937_64 engine(20260819);
    std::vector<Distribution> distributions;

    {
        // Uniform across four orders of magnitude: exercises many buckets.
        std::uniform_int_distribution<std::uint64_t> uniform(1, 10'000'000);
        std::vector<std::uint64_t> samples(count);
        for (auto& sample : samples) {
            sample = uniform(engine);
        }
        distributions.push_back({"uniform_1ns_10ms", std::move(samples)});
    }

    {
        // Log-normal: the shape real latency actually takes — a tight body with
        // a long right tail.
        std::lognormal_distribution<double> lognormal(13.0, 1.0);
        std::vector<std::uint64_t> samples(count);
        for (auto& sample : samples) {
            sample = static_cast<std::uint64_t>(std::max(1.0, lognormal(engine)));
        }
        distributions.push_back({"lognormal", std::move(samples)});
    }

    {
        // Bimodal: a fast path plus a 2% slow path. This is the case percentile
        // reporting exists for, and the one a naive bucketing gets worst.
        std::normal_distribution<double> fast(1'000'000.0, 50'000.0);
        std::normal_distribution<double> slow(50'000'000.0, 5'000'000.0);
        std::uniform_real_distribution<double> pick(0.0, 1.0);
        std::vector<std::uint64_t> samples(count);
        for (auto& sample : samples) {
            const double value = pick(engine) < 0.02 ? slow(engine) : fast(engine);
            sample = static_cast<std::uint64_t>(std::max(1.0, value));
        }
        distributions.push_back({"bimodal_2pct_tail", std::move(samples)});
    }

    {
        // Every sample identical: the histogram must not invent spread.
        distributions.push_back({"constant", std::vector<std::uint64_t>(count, 1'234'567)});
    }

    return distributions;
}

/// Exact percentile over a sorted copy, using the same rank convention as the
/// histogram: the smallest value whose cumulative count reaches ceil(p * N).
std::uint64_t exact_percentile(std::vector<std::uint64_t> samples, double quantile) {
    if (samples.empty()) {
        return 0;
    }
    std::sort(samples.begin(), samples.end());
    auto rank = static_cast<std::size_t>(quantile * static_cast<double>(samples.size()));
    if (rank == 0) {
        rank = 1;
    }
    return samples[std::min(rank - 1, samples.size() - 1)];
}

double relative_error(std::uint64_t reported, std::uint64_t exact) {
    if (exact == 0) {
        return reported == 0 ? 0.0 : 1.0;
    }
    return std::fabs(static_cast<double>(reported) - static_cast<double>(exact)) /
           static_cast<double>(exact);
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t count = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 200'000;
    const std::vector<double> quantiles = {0.50, 0.90, 0.95, 0.99, 0.999};

    JsonWriter writer(std::cout);
    writer.begin_object();
    writer.field("benchmark", std::string("latency_histogram"));
    writer.field("sub_buckets", static_cast<std::uint64_t>(LatencyHistogram::kSubBuckets));
    writer.field("documented_max_relative_error",
                 1.0 / static_cast<double>(LatencyHistogram::kSubBuckets));
    writer.field("samples_per_distribution", static_cast<std::uint64_t>(count));
    writer.begin_array("distributions");

    for (const Distribution& distribution : build_distributions(count)) {
        LatencyHistogram histogram;

        Timer timer;
        timer.start();
        for (std::uint64_t sample : distribution.samples) {
            histogram.record(sample);
        }
        const double record_seconds = timer.elapsed_seconds();

        double worst_error = 0.0;

        writer.array_element_begin();
        writer.field("name", distribution.name);
        writer.field("records_per_second",
                     static_cast<double>(count) / std::max(record_seconds, 1e-12));
        writer.begin_array("percentiles");

        for (double quantile : quantiles) {
            const std::uint64_t reported = histogram.percentile(quantile);
            const std::uint64_t exact = exact_percentile(distribution.samples, quantile);
            const double error = relative_error(reported, exact);
            worst_error = std::max(worst_error, error);

            writer.array_element_begin();
            writer.field("quantile", quantile);
            writer.field("histogram_ns", reported);
            writer.field("exact_ns", exact);
            writer.field("relative_error", error);
            writer.array_element_end();
        }

        writer.end_array();
        writer.field("worst_relative_error", worst_error);
        // The claim this benchmark exists to check.
        writer.field("within_documented_bound",
                     std::string(worst_error <= 1.0 / LatencyHistogram::kSubBuckets ? "true"
                                                                                    : "false"));
        writer.array_element_end();
    }

    writer.end_array();
    writer.end_object();
    writer.finish();
    return 0;
}

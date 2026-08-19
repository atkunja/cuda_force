// How many concurrent sequences fit in a fixed KV cache, paged versus
// contiguous.
//
// This is measurable without a GPU because it is a bookkeeping question, not a
// kernel one. The cache size is fixed; the only variable is how the space is
// carved up, and the answer is entirely determined by the sequence-length
// distribution.
//
// The contiguous baseline is the standard implementation: reserve
// `max_sequence_length` tokens per slot, because a sequence may grow to that
// and cannot be moved once it has started. The paged scheme allocates one block
// at a time and wastes at most `block_size - 1` tokens per sequence.
//
// Also reported: allocator throughput, since a per-token allocation on the
// decode path has to be cheap to be viable.

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "bench_common.hpp"
#include "cudaforge/kv_cache.hpp"

using cudaforge::BlockAllocator;
using cudaforge::SequenceBlockTable;
using namespace cudaforge::bench;

namespace {

/// Sequence-length distributions, because the answer depends entirely on which
/// one a deployment actually sees.
struct Workload {
    std::string name;
    std::string description;
    std::vector<std::size_t> lengths;
};

std::vector<Workload> build_workloads(std::size_t count, std::size_t max_length) {
    // Seeded explicitly: an unseeded generator would make the reported ratios
    // move between runs for no reason.
    std::mt19937_64 engine(20260819);
    std::vector<Workload> workloads;

    {
        // Chat: most turns short, a long tail. The case paging helps most.
        std::lognormal_distribution<double> lognormal(4.6, 0.9);  // median ~100
        std::vector<std::size_t> lengths(count);
        for (auto& length : lengths) {
            length =
                std::clamp(static_cast<std::size_t>(lognormal(engine)), std::size_t{1}, max_length);
        }
        workloads.push_back({"chat_lognormal", "short turns with a long tail", std::move(lengths)});
    }

    {
        // Uniform over the whole permitted range.
        std::uniform_int_distribution<std::size_t> uniform(1, max_length);
        std::vector<std::size_t> lengths(count);
        for (auto& length : lengths) {
            length = uniform(engine);
        }
        workloads.push_back({"uniform", "uniform over the accepted range", std::move(lengths)});
    }

    {
        // Every sequence at the maximum: the case where paging cannot help,
        // included so the comparison is not stacked.
        workloads.push_back({"all_maximum", "every sequence at the limit",
                             std::vector<std::size_t>(count, max_length)});
    }

    {
        // Short prompts, as in classification or embedding workloads.
        std::uniform_int_distribution<std::size_t> uniform(16, 128);
        std::vector<std::size_t> lengths(count);
        for (auto& length : lengths) {
            length = uniform(engine);
        }
        workloads.push_back({"short_prompts", "16-128 tokens", std::move(lengths)});
    }

    return workloads;
}

struct Occupancy {
    std::size_t sequences_admitted;
    std::size_t tokens_held;
    double waste_fraction;
};

/// Contiguous: one reservation of `max_length` per sequence, regardless of how
/// long the sequence turns out to be.
Occupancy contiguous_occupancy(const std::vector<std::size_t>& lengths, std::size_t cache_tokens,
                               std::size_t max_length) {
    const std::size_t slots = cache_tokens / max_length;
    const std::size_t admitted = std::min(slots, lengths.size());
    const std::size_t tokens = std::accumulate(
        lengths.begin(), lengths.begin() + static_cast<long>(admitted), std::size_t{0});
    const std::size_t reserved = admitted * max_length;
    return {admitted, tokens,
            reserved > 0 ? 1.0 - static_cast<double>(tokens) / static_cast<double>(reserved) : 0.0};
}

/// Paged: blocks are taken one at a time as the sequence grows, so only the
/// final block of each sequence is partly wasted.
Occupancy paged_occupancy(const std::vector<std::size_t>& lengths, std::size_t cache_tokens,
                          std::size_t block_size) {
    BlockAllocator allocator(cache_tokens / block_size, block_size);

    std::size_t admitted = 0;
    std::size_t tokens = 0;
    std::size_t reserved = 0;

    for (std::size_t length : lengths) {
        SequenceBlockTable table(admitted, block_size);
        bool fitted = true;

        for (std::size_t token = 0; token < length; ++token) {
            if (table.needs_block()) {
                const auto block = allocator.allocate();
                if (!block) {
                    fitted = false;
                    break;
                }
                table.append_block(*block);
            }
            table.add_tokens(1);
        }

        if (!fitted) {
            // Blocks already taken by the rejected sequence stay allocated; a
            // real scheduler would release them. Counting them as held is the
            // pessimistic choice, which is the right one for this comparison.
            break;
        }
        ++admitted;
        tokens += length;
        reserved += table.capacity();
    }

    return {admitted, tokens,
            reserved > 0 ? 1.0 - static_cast<double>(tokens) / static_cast<double>(reserved) : 0.0};
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t cache_tokens =
        argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 1u << 20;  // ~1M token slots
    constexpr std::size_t kMaxLength = 2048;
    constexpr std::size_t kCandidates = 20000;

    JsonWriter writer(std::cout);
    writer.begin_object();
    writer.field("benchmark", std::string("kv_cache_occupancy"));
    writer.field("note", std::string("bookkeeping only; no device memory is allocated and no "
                                     "attention kernel reads these blocks"));
    writer.field("cache_tokens", static_cast<std::uint64_t>(cache_tokens));
    writer.field("max_sequence_length", static_cast<std::uint64_t>(kMaxLength));
    writer.begin_array("workloads");

    for (const Workload& workload : build_workloads(kCandidates, kMaxLength)) {
        const Occupancy contiguous =
            contiguous_occupancy(workload.lengths, cache_tokens, kMaxLength);

        writer.array_element_begin();
        writer.field("workload", workload.name);
        writer.field("description", workload.description);
        writer.field("mean_length",
                     static_cast<double>(std::accumulate(workload.lengths.begin(),
                                                         workload.lengths.end(), std::size_t{0})) /
                         static_cast<double>(workload.lengths.size()));
        writer.field("contiguous_sequences",
                     static_cast<std::uint64_t>(contiguous.sequences_admitted));
        writer.field("contiguous_waste", contiguous.waste_fraction);

        writer.begin_array("paged");
        for (std::size_t block_size : {8U, 16U, 32U, 64U}) {
            const Occupancy paged = paged_occupancy(workload.lengths, cache_tokens, block_size);

            writer.array_element_begin();
            writer.field("block_size", static_cast<std::uint64_t>(block_size));
            writer.field("sequences", static_cast<std::uint64_t>(paged.sequences_admitted));
            writer.field("waste", paged.waste_fraction);
            writer.field("sequences_ratio",
                         contiguous.sequences_admitted > 0
                             ? static_cast<double>(paged.sequences_admitted) /
                                   static_cast<double>(contiguous.sequences_admitted)
                             : 0.0);
            writer.array_element_end();
        }
        writer.end_array();
        writer.array_element_end();
    }

    writer.end_array();

    // Allocator throughput. A per-token allocation on the decode path has to be
    // cheap for the scheme to be viable at all.
    writer.begin_array("allocator_throughput");
    for (std::size_t blocks : {1024U, 65536U}) {
        BlockAllocator allocator(blocks, 16);
        std::vector<cudaforge::BlockId> held;
        held.reserve(blocks);

        Timer timer;
        timer.start();
        constexpr int kRounds = 100;
        for (int round = 0; round < kRounds; ++round) {
            while (const auto block = allocator.allocate()) {
                held.push_back(*block);
            }
            for (cudaforge::BlockId block : held) {
                allocator.release(block);
            }
            held.clear();
        }
        const double seconds = timer.elapsed_seconds();

        writer.array_element_begin();
        writer.field("pool_blocks", static_cast<std::uint64_t>(blocks));
        writer.field("operations_per_second", static_cast<double>(kRounds) *
                                                  static_cast<double>(blocks) * 2.0 /
                                                  std::max(seconds, 1e-12));
        writer.array_element_end();
    }
    writer.end_array();

    writer.end_object();
    writer.finish();
    return 0;
}

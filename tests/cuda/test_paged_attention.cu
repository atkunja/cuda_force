#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

#include "cuda_test_utils.cuh"
#include "cudaforge/paged_attention.cuh"

using namespace cudaforge;
using namespace cudaforge::test;

namespace {

/// A paged cache and the tables that address it, built so the physical layout
/// deliberately disagrees with the logical one.
struct PagedFixture {
    int sequences;
    int heads;
    int kv_heads;
    int head_dim;
    int block_size;
    int max_blocks;
    std::vector<float> query;
    std::vector<float> k_cache;
    std::vector<float> v_cache;
    std::vector<DeviceBlockId> tables;
    std::vector<int> context;
};

/// Physical slot of a (block, offset, kv_head) triple.
std::size_t slot_of(const PagedFixture& fixture, DeviceBlockId block, int offset, int kv_head) {
    return ((static_cast<std::size_t>(block) * static_cast<std::size_t>(fixture.block_size) +
             static_cast<std::size_t>(offset)) *
                static_cast<std::size_t>(fixture.kv_heads) +
            static_cast<std::size_t>(kv_head)) *
           static_cast<std::size_t>(fixture.head_dim);
}

/// Attention computed on the host, in double, gathering through the same block
/// table the kernel is given.
std::vector<float> reference_paged_attention(const PagedFixture& fixture, float scale) {
    const int group = fixture.heads / fixture.kv_heads;
    std::vector<float> out(static_cast<std::size_t>(fixture.sequences) * fixture.heads *
                           fixture.head_dim);

    for (int sequence = 0; sequence < fixture.sequences; ++sequence) {
        const int context = fixture.context[static_cast<std::size_t>(sequence)];
        for (int head = 0; head < fixture.heads; ++head) {
            const int kv_head = head / group;
            const std::size_t row =
                (static_cast<std::size_t>(sequence) * fixture.heads + head) * fixture.head_dim;

            std::vector<double> scores(static_cast<std::size_t>(std::max(context, 0)));
            for (int token = 0; token < context; ++token) {
                const DeviceBlockId block =
                    fixture.tables[static_cast<std::size_t>(sequence) * fixture.max_blocks +
                                   static_cast<std::size_t>(token / fixture.block_size)];
                const std::size_t slot =
                    slot_of(fixture, block, token % fixture.block_size, kv_head);

                double dot = 0.0;
                for (int d = 0; d < fixture.head_dim; ++d) {
                    dot += static_cast<double>(fixture.query[row + static_cast<std::size_t>(d)]) *
                           static_cast<double>(fixture.k_cache[slot + static_cast<std::size_t>(d)]);
                }
                scores[static_cast<std::size_t>(token)] = dot * static_cast<double>(scale);
            }

            if (context <= 0) {
                for (int d = 0; d < fixture.head_dim; ++d) {
                    out[row + static_cast<std::size_t>(d)] = 0.0F;
                }
                continue;
            }

            const double peak = *std::max_element(scores.begin(), scores.end());
            double total = 0.0;
            for (double& score : scores) {
                score = std::exp(score - peak);
                total += score;
            }

            for (int d = 0; d < fixture.head_dim; ++d) {
                double sum = 0.0;
                for (int token = 0; token < context; ++token) {
                    const DeviceBlockId block =
                        fixture.tables[static_cast<std::size_t>(sequence) * fixture.max_blocks +
                                       static_cast<std::size_t>(token / fixture.block_size)];
                    const std::size_t slot =
                        slot_of(fixture, block, token % fixture.block_size, kv_head);
                    sum += scores[static_cast<std::size_t>(token)] *
                           static_cast<double>(fixture.v_cache[slot + static_cast<std::size_t>(d)]);
                }
                out[row + static_cast<std::size_t>(d)] = static_cast<float>(sum / total);
            }
        }
    }
    return out;
}

/// Block tables that are deliberately not the identity.
///
/// A kernel that ignored the indirection and read token `t` from slot `t` would
/// agree with the reference on an identity table, and every test here would
/// pass while the paged cache did nothing. Scrambling the assignment is what
/// makes these tests capable of failing.
PagedFixture make_fixture(int sequences, int heads, int kv_heads, int head_dim, int block_size,
                          const std::vector<int>& context, unsigned seed) {
    PagedFixture fixture;
    fixture.sequences = sequences;
    fixture.heads = heads;
    fixture.kv_heads = kv_heads;
    fixture.head_dim = head_dim;
    fixture.block_size = block_size;
    fixture.context = context;

    int needed = 0;
    for (int length : context) {
        needed += (length + block_size - 1) / block_size;
    }
    fixture.max_blocks = 0;
    for (int length : context) {
        fixture.max_blocks =
            std::max(fixture.max_blocks, (length + block_size - 1) / block_size);
    }
    fixture.max_blocks = std::max(fixture.max_blocks, 1);

    // More physical blocks than any sequence needs, handed out in reverse so a
    // sequence's logical order never matches its physical order.
    const int total_blocks = needed + 3;
    fixture.k_cache = random_vector(static_cast<std::size_t>(total_blocks) * block_size * kv_heads *
                                        head_dim,
                                    seed);
    fixture.v_cache = random_vector(static_cast<std::size_t>(total_blocks) * block_size * kv_heads *
                                        head_dim,
                                    seed + 17);
    fixture.query = random_vector(
        static_cast<std::size_t>(sequences) * heads * head_dim, seed + 91);

    fixture.tables.assign(static_cast<std::size_t>(sequences) * fixture.max_blocks, 0);
    DeviceBlockId next = static_cast<DeviceBlockId>(total_blocks - 1);
    for (int sequence = 0; sequence < sequences; ++sequence) {
        const int blocks =
            (context[static_cast<std::size_t>(sequence)] + block_size - 1) / block_size;
        for (int logical = 0; logical < blocks; ++logical) {
            fixture.tables[static_cast<std::size_t>(sequence) * fixture.max_blocks +
                           static_cast<std::size_t>(logical)] = next;
            next = (next == 0) ? static_cast<DeviceBlockId>(total_blocks - 1)
                               : static_cast<DeviceBlockId>(next - 1);
        }
    }
    return fixture;
}

std::vector<float> run_paged_attention(const PagedFixture& fixture, float scale) {
    CudaStream stream;
    DeviceBuffer<float> query(fixture.query.size());
    DeviceBuffer<float> k_cache(fixture.k_cache.size());
    DeviceBuffer<float> v_cache(fixture.v_cache.size());
    DeviceBuffer<DeviceBlockId> tables(fixture.tables.size());
    DeviceBuffer<int> context(fixture.context.size());
    DeviceBuffer<float> out(fixture.query.size());

    query.copy_from_host(fixture.query.data(), fixture.query.size(), stream);
    k_cache.copy_from_host(fixture.k_cache.data(), fixture.k_cache.size(), stream);
    v_cache.copy_from_host(fixture.v_cache.data(), fixture.v_cache.size(), stream);
    tables.copy_from_host(fixture.tables.data(), fixture.tables.size(), stream);
    context.copy_from_host(fixture.context.data(), fixture.context.size(), stream);

    launch_paged_attention(query.data(), k_cache.data(), v_cache.data(), tables.data(),
                           context.data(), out.data(), fixture.sequences, fixture.heads,
                           fixture.kv_heads, fixture.head_dim, fixture.block_size,
                           fixture.max_blocks, scale, stream);

    std::vector<float> host(fixture.query.size());
    out.copy_to_host(host.data(), host.size(), stream);
    stream.synchronize();
    return host;
}

}  // namespace

TEST_CASE("paged attention matches a host reference", "[cuda][paged]") {
    struct Case {
        int sequences;
        int heads;
        int kv_heads;
        int head_dim;
        int block_size;
        std::vector<int> context;
    };

    // Context lengths deliberately include a partial final block, a length that
    // is an exact multiple of the block size, and a single token.
    const std::vector<Case> cases = {
        {1, 1, 1, 32, 4, {1}},
        {1, 2, 2, 64, 8, {8}},
        {3, 2, 2, 32, 4, {5, 12, 1}},
        {2, 4, 2, 64, 16, {33, 7}},
        {2, 8, 2, 128, 8, {17, 64}},
    };

    for (const Case& item : cases) {
        const PagedFixture fixture =
            make_fixture(item.sequences, item.heads, item.kv_heads, item.head_dim, item.block_size,
                         item.context, static_cast<unsigned>(item.head_dim * 7 + item.block_size));
        const float scale = 1.0F / std::sqrt(static_cast<float>(item.head_dim));

        const std::vector<float> expected = reference_paged_attention(fixture, scale);
        const std::vector<float> actual = run_paged_attention(fixture, scale);

        REQUIRE(actual.size() == expected.size());
        for (std::size_t i = 0; i < expected.size(); ++i) {
            INFO("head_dim " << item.head_dim << " block_size " << item.block_size << " index " << i);
            REQUIRE(actual[i] == Catch::Approx(expected[i]).margin(2e-5));
        }
    }
}

TEST_CASE("paged attention actually follows the block table", "[cuda][paged]") {
    // The property that makes every other test here meaningful. Reading the
    // same cache through a different table must give a different answer; if it
    // does not, the kernel is ignoring the indirection and the paged cache is
    // decorative.
    PagedFixture fixture = make_fixture(1, 1, 1, 32, 4, {8}, 3);
    const float scale = 1.0F / std::sqrt(32.0F);
    const std::vector<float> original = run_paged_attention(fixture, scale);

    REQUIRE(fixture.tables.size() >= 2);
    std::swap(fixture.tables[0], fixture.tables[1]);
    const std::vector<float> swapped = run_paged_attention(fixture, scale);

    bool differs = false;
    for (std::size_t i = 0; i < original.size(); ++i) {
        if (std::fabs(original[i] - swapped[i]) > 1e-6F) {
            differs = true;
            break;
        }
    }
    REQUIRE(differs);

    // And swapping back restores it exactly, so the difference above is the
    // table and not run-to-run noise.
    std::swap(fixture.tables[0], fixture.tables[1]);
    const std::vector<float> restored = run_paged_attention(fixture, scale);
    for (std::size_t i = 0; i < original.size(); ++i) {
        REQUIRE(restored[i] == Catch::Approx(original[i]).margin(1e-6));
    }
}

TEST_CASE("paged attention handles an empty sequence", "[cuda][paged]") {
    // A sequence admitted but not yet holding a token shares the launch with
    // sequences that do. It must write zeros rather than read a block table
    // entry that was never filled in.
    const PagedFixture fixture = make_fixture(2, 2, 2, 32, 4, {0, 6}, 11);
    const float scale = 1.0F / std::sqrt(32.0F);

    const std::vector<float> expected = reference_paged_attention(fixture, scale);
    const std::vector<float> actual = run_paged_attention(fixture, scale);

    for (std::size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(actual[i] == Catch::Approx(expected[i]).margin(2e-5));
    }
    for (int d = 0; d < fixture.head_dim * fixture.heads; ++d) {
        REQUIRE(actual[static_cast<std::size_t>(d)] == 0.0F);
    }
}

TEST_CASE("grouped-query attention shares kv heads across query heads", "[cuda][paged]") {
    // num_heads a multiple of num_kv_heads is the GQA case every current model
    // uses. Checked separately because getting the group arithmetic wrong is
    // silent: it reads a valid KV head, just the wrong one.
    const PagedFixture fixture = make_fixture(2, 8, 2, 64, 8, {20, 9}, 5);
    const float scale = 1.0F / std::sqrt(64.0F);

    const std::vector<float> expected = reference_paged_attention(fixture, scale);
    const std::vector<float> actual = run_paged_attention(fixture, scale);

    for (std::size_t i = 0; i < expected.size(); ++i) {
        INFO("index " << i);
        REQUIRE(actual[i] == Catch::Approx(expected[i]).margin(2e-5));
    }
}

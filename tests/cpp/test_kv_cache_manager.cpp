#include "cudaforge/kv_cache_manager.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <vector>

using cudaforge::AdmissionResult;
using cudaforge::KVCacheManager;
using cudaforge::PreemptionPolicy;
using cudaforge::SequenceId;

namespace {

constexpr std::size_t kBlockSize = 16;

/// Fills the cache with `count` sequences of `tokens` each, returning their ids.
std::vector<SequenceId> fill(KVCacheManager& manager, std::size_t count, std::size_t tokens) {
    std::vector<SequenceId> admitted;
    for (std::size_t i = 0; i < count; ++i) {
        const auto sequence = static_cast<SequenceId>(i + 1);
        if (manager.admit(sequence, tokens).ok()) {
            admitted.push_back(sequence);
        }
    }
    return admitted;
}

}  // namespace

TEST_CASE("an admitted sequence holds the blocks its tokens need", "[kvmanager]") {
    KVCacheManager manager(16, kBlockSize);

    const auto outcome = manager.admit(1, 40);
    REQUIRE(outcome.result == AdmissionResult::Admitted);
    REQUIRE(outcome.preempted.empty());

    REQUIRE(manager.is_admitted(1));
    REQUIRE(manager.tokens_held(1) == 40);
    // 40 tokens at 16 per block is three blocks, the last part-filled.
    REQUIRE(manager.blocks_held(1) == 3);
}

TEST_CASE("decoding into slack costs no blocks", "[kvmanager]") {
    // The common case during generation: one token at a time into a block that
    // already has room. If every token allocated, paging would be pointless.
    KVCacheManager manager(16, kBlockSize);
    REQUIRE(manager.admit(1, 1).ok());
    REQUIRE(manager.blocks_held(1) == 1);

    for (int token = 0; token < 15; ++token) {
        REQUIRE(manager.extend(1, 1).ok());
    }
    REQUIRE(manager.tokens_held(1) == 16);
    REQUIRE(manager.blocks_held(1) == 1);

    // The seventeenth token crosses the boundary.
    REQUIRE(manager.extend(1, 1).ok());
    REQUIRE(manager.blocks_held(1) == 2);
}

TEST_CASE("a full cache admits by evicting", "[kvmanager][preemption]") {
    KVCacheManager manager(4, kBlockSize);
    REQUIRE(fill(manager, 4, kBlockSize).size() == 4);
    REQUIRE(manager.free_blocks() == 0);

    const auto outcome = manager.admit(99, kBlockSize);
    REQUIRE(outcome.result == AdmissionResult::PreemptedOthers);
    REQUIRE(outcome.preempted.size() == 1);
    REQUIRE(manager.is_admitted(99));
    REQUIRE(manager.tokens_held(99) == kBlockSize);
}

TEST_CASE("newest-first protects the oldest sequence", "[kvmanager][preemption]") {
    // The property that keeps throughput from collapsing: older sequences are
    // closer to finishing, so evicting them repeatedly recomputes the work
    // nearest completion and nothing ever finishes.
    KVCacheManager manager(4, kBlockSize, PreemptionPolicy::Newest);
    fill(manager, 4, kBlockSize);

    for (int round = 0; round < 10; ++round) {
        const auto sequence = static_cast<SequenceId>(1000 + round);
        REQUIRE(manager.admit(sequence, kBlockSize).ok());
        manager.release(sequence);
    }

    // Sequence 1 was admitted first and must still hold its tokens.
    REQUIRE(manager.is_admitted(1));
    REQUIRE(manager.tokens_held(1) == kBlockSize);
}

TEST_CASE("newest-first evicts in reverse admission order", "[kvmanager][preemption]") {
    KVCacheManager manager(3, kBlockSize, PreemptionPolicy::Newest);
    fill(manager, 3, kBlockSize);

    const auto outcome = manager.admit(50, kBlockSize);
    REQUIRE(outcome.preempted == std::vector<SequenceId>{3});
}

TEST_CASE("largest-first evicts the biggest holder", "[kvmanager][preemption]") {
    // Frees the most per eviction, so fewer sequences are disturbed to satisfy
    // a given demand — at the cost of recomputing the most work.
    KVCacheManager manager(8, kBlockSize, PreemptionPolicy::Largest);
    REQUIRE(manager.admit(1, kBlockSize).ok());      // 1 block
    REQUIRE(manager.admit(2, kBlockSize * 4).ok());  // 4 blocks
    REQUIRE(manager.admit(3, kBlockSize).ok());      // 1 block

    const auto outcome = manager.admit(4, kBlockSize * 3);
    REQUIRE(outcome.result == AdmissionResult::PreemptedOthers);
    REQUIRE(outcome.preempted == std::vector<SequenceId>{2});
    // The two small sequences were left alone.
    REQUIRE(manager.is_admitted(1));
    REQUIRE(manager.is_admitted(3));
}

TEST_CASE("a sequence is never evicted to satisfy its own request", "[kvmanager][preemption]") {
    // Otherwise admission can livelock: the requester is the newest, so
    // newest-first would evict it to make room for itself, forever.
    KVCacheManager manager(4, kBlockSize, PreemptionPolicy::Newest);
    fill(manager, 3, kBlockSize);
    REQUIRE(manager.admit(4, kBlockSize).ok());  // newest, fills the cache

    const auto outcome = manager.extend(4, kBlockSize);
    REQUIRE(outcome.ok());
    REQUIRE(std::find(outcome.preempted.begin(), outcome.preempted.end(), SequenceId{4}) ==
            outcome.preempted.end());
    REQUIRE(manager.tokens_held(4) == kBlockSize * 2);
}

TEST_CASE("a preempted sequence keeps its identity and can be re-admitted",
          "[kvmanager][preemption]") {
    // Recompute, not swap: the blocks go back to the pool and the prompt is
    // re-run later, so the sequence stays known rather than being forgotten.
    KVCacheManager manager(2, kBlockSize, PreemptionPolicy::Newest);
    REQUIRE(manager.admit(1, kBlockSize).ok());
    REQUIRE(manager.admit(2, kBlockSize).ok());

    REQUIRE(manager.admit(3, kBlockSize * 2).ok());
    REQUIRE(manager.is_admitted(2));       // still known
    REQUIRE(manager.tokens_held(2) == 0);  // but holds nothing

    manager.release(3);
    REQUIRE(manager.extend(2, kBlockSize).ok());
    REQUIRE(manager.tokens_held(2) == kBlockSize);
}

TEST_CASE("a request larger than the whole cache fails without evicting",
          "[kvmanager][preemption]") {
    // Evicting everything and still failing is the worst outcome available:
    // sequences destroyed for an admission that could never have succeeded.
    KVCacheManager manager(4, kBlockSize, PreemptionPolicy::Newest);
    fill(manager, 4, kBlockSize);

    const auto outcome = manager.admit(99, kBlockSize * 100);
    REQUIRE(outcome.result == AdmissionResult::InsufficientCache);
    REQUIRE(outcome.preempted.empty());
    REQUIRE(manager.preemption_count() == 0);

    for (SequenceId sequence = 1; sequence <= 4; ++sequence) {
        REQUIRE(manager.tokens_held(sequence) == kBlockSize);
    }
}

TEST_CASE("an unsatisfiable demand evicts nothing even when it nearly fits",
          "[kvmanager][preemption]") {
    // Four blocks total, three held by others, requester holds one: evicting
    // everything eligible yields three free, one short of the four wanted.
    KVCacheManager manager(4, kBlockSize, PreemptionPolicy::Newest);
    REQUIRE(manager.admit(1, kBlockSize).ok());
    REQUIRE(manager.admit(2, kBlockSize).ok());
    REQUIRE(manager.admit(3, kBlockSize).ok());
    REQUIRE(manager.admit(4, kBlockSize).ok());

    const auto outcome = manager.extend(4, kBlockSize * 4);
    REQUIRE(outcome.result == AdmissionResult::InsufficientCache);
    REQUIRE(outcome.preempted.empty());
    REQUIRE(manager.preemption_count() == 0);
    // Every other sequence survived.
    REQUIRE(manager.active_sequences() == 4);
    for (SequenceId sequence = 1; sequence <= 3; ++sequence) {
        REQUIRE(manager.tokens_held(sequence) == kBlockSize);
    }
}

TEST_CASE("release returns blocks and forgets the sequence", "[kvmanager]") {
    KVCacheManager manager(8, kBlockSize);
    REQUIRE(manager.admit(1, kBlockSize * 4).ok());
    REQUIRE(manager.free_blocks() == 4);

    manager.release(1);
    REQUIRE_FALSE(manager.is_admitted(1));
    REQUIRE(manager.free_blocks() == 8);
    REQUIRE(manager.active_sequences() == 0);
}

TEST_CASE("releasing an unknown sequence is harmless", "[kvmanager]") {
    KVCacheManager manager(4, kBlockSize);
    REQUIRE_NOTHROW(manager.release(999));
}

TEST_CASE("explicit preemption reclaims blocks and counts the recompute",
          "[kvmanager][preemption]") {
    KVCacheManager manager(8, kBlockSize);
    REQUIRE(manager.admit(1, kBlockSize * 3).ok());

    REQUIRE(manager.preempt(1) == 3);
    REQUIRE(manager.free_blocks() == 8);
    REQUIRE(manager.preemption_count() == 1);
    // The tokens that will have to be recomputed are the cost of the policy,
    // so they are counted rather than left implicit.
    REQUIRE(manager.recomputed_tokens() == kBlockSize * 3);
}

TEST_CASE("preempting a sequence that holds nothing is a no-op", "[kvmanager]") {
    KVCacheManager manager(4, kBlockSize);
    REQUIRE(manager.admit(1, kBlockSize).ok());
    REQUIRE(manager.preempt(1) == 1);
    REQUIRE(manager.preempt(1) == 0);
    REQUIRE(manager.preemption_count() == 1);
}

TEST_CASE("extending an unadmitted sequence is rejected", "[kvmanager]") {
    KVCacheManager manager(4, kBlockSize);
    REQUIRE_THROWS_AS(manager.extend(7, 1), std::invalid_argument);
}

TEST_CASE("tokens resolve through the block table", "[kvmanager]") {
    KVCacheManager manager(8, 4);
    REQUIRE(manager.admit(1, 8).ok());

    const auto [first_block, first_offset] = manager.locate(1, 0);
    const auto [later_block, later_offset] = manager.locate(1, 5);
    REQUIRE(first_offset == 0);
    REQUIRE(later_offset == 1);
    REQUIRE(first_block != later_block);
}

TEST_CASE("locating in an unknown sequence is rejected", "[kvmanager]") {
    KVCacheManager manager(4, kBlockSize);
    REQUIRE_THROWS_AS(manager.locate(3, 0), std::invalid_argument);
}

TEST_CASE("a zero-token request changes nothing", "[kvmanager]") {
    KVCacheManager manager(4, kBlockSize);
    const auto outcome = manager.admit(1, 0);
    REQUIRE(outcome.result == AdmissionResult::Admitted);
    REQUIRE(manager.free_blocks() == 4);
}

TEST_CASE("sustained load under preemption keeps completing sequences", "[kvmanager][preemption]") {
    // The end-to-end property. With newest-first, a steady stream of arrivals
    // against a cache that cannot hold them all must still let the oldest
    // sequences run to completion — that is what distinguishes preemption from
    // thrashing.
    KVCacheManager manager(8, kBlockSize, PreemptionPolicy::Newest);

    std::vector<SequenceId> live;
    std::size_t completed = 0;

    for (SequenceId sequence = 1; sequence <= 200; ++sequence) {
        if (manager.admit(sequence, kBlockSize).ok()) {
            live.push_back(sequence);
        }

        // Advance everything still holding cache; retire whatever reaches its
        // budget, as a generation loop would.
        for (auto it = live.begin(); it != live.end();) {
            if (manager.tokens_held(*it) == 0) {
                ++it;  // preempted; a real scheduler would re-queue it
                continue;
            }
            manager.extend(*it, 1);
            if (manager.tokens_held(*it) >= kBlockSize * 2) {
                manager.release(*it);
                ++completed;
                it = live.erase(it);
            } else {
                ++it;
            }
        }
    }

    REQUIRE(completed > 0);
    REQUIRE(manager.preemption_count() > 0);
    // Progress, not thrashing: far more sequences finished than the cache could
    // ever hold at once.
    REQUIRE(completed > 8);
}

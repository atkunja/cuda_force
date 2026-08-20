#pragma once

#include <cstddef>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include "cudaforge/kv_cache.hpp"

namespace cudaforge {

/// Which sequence to evict when the block pool is exhausted.
///
/// The allocator can free a sequence's blocks; nothing in it decides *whose*.
/// That decision is the scheduling half of paged attention, and getting it
/// wrong does not show up as an error — it shows up as throughput collapsing
/// under load.
enum class PreemptionPolicy : std::uint8_t {
    /// Evict the most recently admitted sequence.
    ///
    /// The default, and the one a serving system generally wants. Older
    /// sequences are closer to finishing, so protecting them means the work
    /// already invested is not thrown away and requests keep completing.
    ///
    /// The failure mode this avoids is worth naming: evicting the *oldest*
    /// repeatedly kills the sequence nearest completion, so tokens are
    /// recomputed forever and throughput approaches zero while every individual
    /// admission still "succeeds".
    Newest,

    /// Evict whichever sequence holds the most blocks.
    ///
    /// Frees the most memory per eviction, so it preempts fewer sequences to
    /// satisfy a given demand. The cost is that the biggest sequence is usually
    /// the one with the most work invested in it, so the recompute bill is
    /// higher. Useful when admissions are large and preemption is rare.
    Largest,
};

/// What happened to a request for more cache.
enum class AdmissionResult : std::uint8_t {
    Admitted,           ///< blocks were available, or were freed by preemption
    PreemptedOthers,    ///< admitted, but one or more sequences were evicted
    InsufficientCache,  ///< even evicting everything eligible would not be enough
};

struct AdmissionOutcome {
    AdmissionResult result = AdmissionResult::Admitted;
    /// Sequences evicted to make room, in the order they were chosen.
    std::vector<SequenceId> preempted;

    [[nodiscard]] bool ok() const noexcept {
        return result != AdmissionResult::InsufficientCache;
    }
};

/// Ties the block allocator, the per-sequence tables and an eviction policy
/// together into something a scheduler can drive.
///
/// ## Recompute, not swap
///
/// A preempted sequence's blocks are returned to the pool and its table is
/// cleared, but the sequence stays *known*: it can be re-admitted and its
/// prompt re-run. The alternative — copying its blocks to host memory and
/// copying them back — is faster to resume but needs a device-to-host transfer
/// per eviction, which competes with the copies the stream scheduler is trying
/// to overlap. Recompute trades wasted compute for not touching the copy
/// engines, and is the right default when preemption is rare.
///
/// Swapping is not implemented. Saying so is more useful than implying that
/// eviction is free.
///
/// ## Thread safety
///
/// One mutex over the whole manager. Admission is a global decision — it may
/// evict any sequence — so there is nothing to shard, and it happens once per
/// batch rather than once per token.
class KVCacheManager {
public:
    KVCacheManager(std::size_t block_count, std::size_t block_size,
                   PreemptionPolicy policy = PreemptionPolicy::Newest);

    KVCacheManager(const KVCacheManager&) = delete;
    KVCacheManager& operator=(const KVCacheManager&) = delete;

    /// Reserve cache for `tokens` of a new sequence, evicting others if needed.
    ///
    /// `sequence` is never itself chosen as a victim: evicting the requester to
    /// satisfy its own request admits nothing and can livelock, admitting and
    /// immediately evicting the same sequence forever.
    AdmissionOutcome admit(SequenceId sequence, std::size_t tokens);

    /// Grow an admitted sequence by `tokens`, evicting others if needed.
    ///
    /// This is the per-step call during generation. It is separate from
    /// `admit` because the failure is different: a sequence that cannot grow
    /// has already consumed cache and produced tokens, so refusing it wastes
    /// more than refusing an admission would.
    AdmissionOutcome extend(SequenceId sequence, std::size_t tokens);

    /// Evict a specific sequence, returning its blocks to the pool. The
    /// sequence remains known and can be re-admitted.
    ///
    /// Returns the number of blocks reclaimed; zero if it held none.
    std::size_t preempt(SequenceId sequence);

    /// Forget a sequence entirely — it finished, or the client gave up.
    void release(SequenceId sequence);

    [[nodiscard]] bool is_admitted(SequenceId sequence) const;
    [[nodiscard]] std::size_t tokens_held(SequenceId sequence) const;
    [[nodiscard]] std::size_t blocks_held(SequenceId sequence) const;

    /// Physical block and offset for a token, for whatever reads the cache.
    [[nodiscard]] std::pair<BlockId, std::size_t> locate(SequenceId sequence,
                                                         std::size_t token_index) const;

    [[nodiscard]] std::size_t active_sequences() const;
    [[nodiscard]] std::size_t free_blocks() const;
    [[nodiscard]] double utilisation() const;

    /// Sequences evicted since construction. A rising count means the cache is
    /// smaller than the offered load, and the recompute is pure waste.
    [[nodiscard]] std::uint64_t preemption_count() const;
    [[nodiscard]] std::uint64_t recomputed_tokens() const;

private:
    /// Blocks a sequence would need to hold `tokens`, given what it holds now.
    [[nodiscard]] std::size_t additional_blocks_for(SequenceId sequence,
                                                    std::size_t tokens) const;

    /// Chooses victims until `needed` blocks are free, never picking `requester`.
    /// Returns the sequences evicted, or an empty vector if it cannot get there.
    std::vector<SequenceId> evict_until(std::size_t needed, SequenceId requester);

    [[nodiscard]] std::optional<SequenceId> choose_victim(SequenceId requester) const;

    AdmissionOutcome reserve(SequenceId sequence, std::size_t tokens, bool creating);

    mutable std::mutex mutex_;
    BlockAllocator allocator_;
    std::size_t block_size_;
    PreemptionPolicy policy_;

    std::unordered_map<SequenceId, SequenceBlockTable> tables_;
    /// Admission order, oldest first. A vector rather than a heap: eviction is
    /// rare and the list is short, so a linear scan is cheaper than maintaining
    /// an index that every admission would have to update.
    std::vector<SequenceId> admission_order_;

    std::uint64_t preemptions_ = 0;
    std::uint64_t recomputed_tokens_ = 0;
};

}  // namespace cudaforge

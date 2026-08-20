#include "cudaforge/kv_cache_manager.hpp"

#include <algorithm>
#include <stdexcept>

namespace cudaforge {

KVCacheManager::KVCacheManager(std::size_t block_count, std::size_t block_size,
                               PreemptionPolicy policy)
    : allocator_(block_count, block_size), block_size_(block_size), policy_(policy) {}

std::size_t KVCacheManager::additional_blocks_for(SequenceId sequence,
                                                  std::size_t tokens) const {
    const auto entry = tables_.find(sequence);
    if (entry == tables_.end()) {
        return (tokens + block_size_ - 1) / block_size_;
    }

    const SequenceBlockTable& table = entry->second;
    const std::size_t wanted = table.token_count() + tokens;
    if (wanted <= table.capacity()) {
        // Fits in the slack of the last block, which is the common case during
        // decoding: one token at a time into a block that holds `block_size`.
        return 0;
    }
    const std::size_t shortfall = wanted - table.capacity();
    return (shortfall + block_size_ - 1) / block_size_;
}

std::optional<SequenceId> KVCacheManager::choose_victim(SequenceId requester) const {
    std::optional<SequenceId> victim;

    switch (policy_) {
        case PreemptionPolicy::Newest: {
            // Walk backwards through admission order. The newest eligible
            // sequence has the least invested in it, so recomputing it costs
            // the least and older sequences keep making progress.
            for (auto it = admission_order_.rbegin(); it != admission_order_.rend(); ++it) {
                if (*it == requester) {
                    continue;
                }
                const auto entry = tables_.find(*it);
                if (entry != tables_.end() && !entry->second.blocks().empty()) {
                    victim = *it;
                    break;
                }
            }
            break;
        }
        case PreemptionPolicy::Largest: {
            std::size_t most = 0;
            for (const SequenceId candidate : admission_order_) {
                if (candidate == requester) {
                    continue;
                }
                const auto entry = tables_.find(candidate);
                if (entry == tables_.end()) {
                    continue;
                }
                const std::size_t held = entry->second.blocks().size();
                if (held > most) {
                    most = held;
                    victim = candidate;
                }
            }
            break;
        }
    }

    return victim;
}

std::vector<SequenceId> KVCacheManager::evict_until(std::size_t needed,
                                                    SequenceId requester) {
    std::vector<SequenceId> evicted;

    while (allocator_.free_blocks() < needed) {
        const auto victim = choose_victim(requester);
        if (!victim) {
            // Nothing left to take. The caller rolls back rather than leaving
            // sequences evicted for an admission that then failed anyway —
            // that would be pure waste.
            return {};
        }

        SequenceBlockTable& table = tables_.at(*victim);
        for (const BlockId block : table.blocks()) {
            allocator_.release(block);
        }
        recomputed_tokens_ += table.token_count();
        ++preemptions_;

        // The table is reset, not erased: the sequence stays known so it can be
        // re-admitted and its prompt re-run.
        table = SequenceBlockTable(*victim, block_size_);
        evicted.push_back(*victim);
    }

    return evicted;
}

AdmissionOutcome KVCacheManager::reserve(SequenceId sequence, std::size_t tokens,
                                         bool creating) {
    AdmissionOutcome outcome;
    if (tokens == 0) {
        return outcome;
    }

    if (creating && tables_.find(sequence) == tables_.end()) {
        tables_.emplace(sequence, SequenceBlockTable(sequence, block_size_));
        admission_order_.push_back(sequence);
    }

    const auto entry = tables_.find(sequence);
    if (entry == tables_.end()) {
        throw std::invalid_argument("KVCacheManager: sequence is not admitted");
    }

    const std::size_t needed = additional_blocks_for(sequence, tokens);
    if (needed > allocator_.total_blocks()) {
        // Larger than the entire cache: no amount of eviction helps, and
        // pretending otherwise would evict everything and still fail.
        outcome.result = AdmissionResult::InsufficientCache;
        return outcome;
    }

    if (allocator_.free_blocks() < needed) {
        outcome.preempted = evict_until(needed, sequence);
        if (allocator_.free_blocks() < needed) {
            outcome.result = AdmissionResult::InsufficientCache;
            return outcome;
        }
        outcome.result = AdmissionResult::PreemptedOthers;
    }

    SequenceBlockTable& table = entry->second;
    for (std::size_t i = 0; i < needed; ++i) {
        const auto block = allocator_.allocate();
        if (!block) {
            // Unreachable: the free count was checked under the same lock.
            throw std::runtime_error("KVCacheManager: allocation failed after reservation");
        }
        table.append_block(*block);
    }
    table.add_tokens(tokens);
    return outcome;
}

AdmissionOutcome KVCacheManager::admit(SequenceId sequence, std::size_t tokens) {
    std::lock_guard lock(mutex_);
    return reserve(sequence, tokens, /*creating=*/true);
}

AdmissionOutcome KVCacheManager::extend(SequenceId sequence, std::size_t tokens) {
    std::lock_guard lock(mutex_);
    return reserve(sequence, tokens, /*creating=*/false);
}

std::size_t KVCacheManager::preempt(SequenceId sequence) {
    std::lock_guard lock(mutex_);

    const auto entry = tables_.find(sequence);
    if (entry == tables_.end()) {
        return 0;
    }

    SequenceBlockTable& table = entry->second;
    const std::size_t reclaimed = table.blocks().size();
    if (reclaimed == 0) {
        return 0;
    }

    for (const BlockId block : table.blocks()) {
        allocator_.release(block);
    }
    recomputed_tokens_ += table.token_count();
    ++preemptions_;
    table = SequenceBlockTable(sequence, block_size_);
    return reclaimed;
}

void KVCacheManager::release(SequenceId sequence) {
    std::lock_guard lock(mutex_);

    const auto entry = tables_.find(sequence);
    if (entry == tables_.end()) {
        return;
    }
    for (const BlockId block : entry->second.blocks()) {
        allocator_.release(block);
    }
    tables_.erase(entry);
    admission_order_.erase(
        std::remove(admission_order_.begin(), admission_order_.end(), sequence),
        admission_order_.end());
}

bool KVCacheManager::is_admitted(SequenceId sequence) const {
    std::lock_guard lock(mutex_);
    return tables_.find(sequence) != tables_.end();
}

std::size_t KVCacheManager::tokens_held(SequenceId sequence) const {
    std::lock_guard lock(mutex_);
    const auto entry = tables_.find(sequence);
    return entry == tables_.end() ? 0 : entry->second.token_count();
}

std::size_t KVCacheManager::blocks_held(SequenceId sequence) const {
    std::lock_guard lock(mutex_);
    const auto entry = tables_.find(sequence);
    return entry == tables_.end() ? 0 : entry->second.blocks().size();
}

std::pair<BlockId, std::size_t> KVCacheManager::locate(SequenceId sequence,
                                                       std::size_t token_index) const {
    std::lock_guard lock(mutex_);
    const auto entry = tables_.find(sequence);
    if (entry == tables_.end()) {
        throw std::invalid_argument("KVCacheManager::locate on an unknown sequence");
    }
    return entry->second.locate(token_index);
}

std::size_t KVCacheManager::active_sequences() const {
    std::lock_guard lock(mutex_);
    return tables_.size();
}

std::size_t KVCacheManager::free_blocks() const {
    std::lock_guard lock(mutex_);
    return allocator_.free_blocks();
}

double KVCacheManager::utilisation() const {
    std::lock_guard lock(mutex_);
    return allocator_.utilisation();
}

std::uint64_t KVCacheManager::preemption_count() const {
    std::lock_guard lock(mutex_);
    return preemptions_;
}

std::uint64_t KVCacheManager::recomputed_tokens() const {
    std::lock_guard lock(mutex_);
    return recomputed_tokens_;
}

}  // namespace cudaforge

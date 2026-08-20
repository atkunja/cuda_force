// Runs a KV-cache scenario and reports every decision, as JSON.
//
// `python/cudaforge/kv_cache.py` reimplements this manager so the engine keeps
// working without a compiled extension. Two implementations of one policy drift
// apart quietly — each suite stays green against its own expectations while the
// admission decisions diverge — so this binary exists to let a Python test drive
// the identical script through both and compare.
//
// It is not a test. It is the C++ half of one, which is why it sits beside the
// tests without being linked into the test binary.
//
//   kv_cache_scenario <block_count> <block_size> <newest|largest> <script>
//
// The script is comma-separated operations:
//
//   a<seq>:<tokens>   admit
//   e<seq>:<tokens>   extend
//   p<seq>            preempt
//   r<seq>            release

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "cudaforge/kv_cache_manager.hpp"

using namespace cudaforge;

namespace {

const char* result_name(AdmissionResult result) {
    switch (result) {
        case AdmissionResult::Admitted:
            return "admitted";
        case AdmissionResult::PreemptedOthers:
            return "preempted_others";
        case AdmissionResult::InsufficientCache:
            return "insufficient_cache";
    }
    return "unknown";
}

void emit_outcome(const AdmissionOutcome& outcome) {
    std::cout << R"({"result":")" << result_name(outcome.result) << R"(","preempted":[)";
    for (std::size_t i = 0; i < outcome.preempted.size(); ++i) {
        std::cout << (i == 0 ? "" : ",") << outcome.preempted[i];
    }
    std::cout << "]}";
}

std::vector<std::string> split(const std::string& text, char separator) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, separator)) {
        if (!item.empty()) {
            parts.push_back(item);
        }
    }
    return parts;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: kv_cache_scenario <block_count> <block_size> "
                     "<newest|largest> <script>\n";
        return 2;
    }

    const auto block_count = static_cast<std::size_t>(std::atoll(argv[1]));
    const auto block_size = static_cast<std::size_t>(std::atoll(argv[2]));
    const std::string policy_name = argv[3];
    const PreemptionPolicy policy =
        policy_name == "largest" ? PreemptionPolicy::Largest : PreemptionPolicy::Newest;

    KVCacheManager manager(block_count, block_size, policy);

    std::vector<SequenceId> seen;
    std::cout << R"({"operations":[)";

    const std::vector<std::string> operations = split(argv[4], ',');
    for (std::size_t i = 0; i < operations.size(); ++i) {
        const std::string& operation = operations[i];
        const char kind = operation[0];
        const std::string body = operation.substr(1);
        const std::size_t colon = body.find(':');
        const auto sequence = static_cast<SequenceId>(std::atoll(body.substr(0, colon).c_str()));
        const std::size_t tokens =
            colon == std::string::npos
                ? 0
                : static_cast<std::size_t>(std::atoll(body.substr(colon + 1).c_str()));

        bool known = false;
        for (const SequenceId id : seen) {
            known = known || id == sequence;
        }
        if (!known) {
            seen.push_back(sequence);
        }

        std::cout << (i == 0 ? "" : ",");
        switch (kind) {
            case 'a':
                emit_outcome(manager.admit(sequence, tokens));
                break;
            case 'e':
                emit_outcome(manager.extend(sequence, tokens));
                break;
            case 'p':
                std::cout << R"({"result":"preempt","reclaimed":)" << manager.preempt(sequence)
                          << "}";
                break;
            case 'r':
                manager.release(sequence);
                std::cout << R"({"result":"release"})";
                break;
            default:
                std::cerr << "unknown operation: " << operation << "\n";
                return 2;
        }
    }

    std::cout << R"(],"final":{"free_blocks":)" << manager.free_blocks() << R"(,"preemptions":)"
              << manager.preemption_count() << R"(,"recomputed_tokens":)"
              << manager.recomputed_tokens() << R"(,"active_sequences":)"
              << manager.active_sequences() << R"(,"sequences":[)";
    for (std::size_t i = 0; i < seen.size(); ++i) {
        std::cout << (i == 0 ? "" : ",") << R"({"id":)" << seen[i] << R"(,"admitted":)"
                  << (manager.is_admitted(seen[i]) ? "true" : "false") << R"(,"blocks":)"
                  << manager.blocks_held(seen[i]) << R"(,"tokens":)" << manager.tokens_held(seen[i])
                  << "}";
    }
    std::cout << "]}}\n";
    return 0;
}

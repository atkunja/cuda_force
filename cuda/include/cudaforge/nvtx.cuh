#pragma once

// NVTX range annotations.
//
// An Nsight Systems timeline shows what the GPU did, but not which phase of the
// host program issued it. A gap between two kernels could be batch formation,
// tokenisation, an allocation, or the host simply being late — and the timeline
// alone cannot distinguish them. NVTX ranges label the regions so a gap can be
// attributed rather than guessed at.
//
// Compiled out entirely unless CUDAFORGE_ENABLE_NVTX is defined, so the release
// path pays nothing. Even when enabled the cost is a few hundred nanoseconds
// per range, which is negligible against the per-batch work being annotated but
// would not be against a per-kernel one — annotate phases, not launches.

#include <cstdint>
#include <string>

#ifdef CUDAFORGE_ENABLE_NVTX
#include <pthread.h>

#include <nvtx3/nvToolsExt.h>
#endif

namespace cudaforge {

/// Colours are assigned per category so a timeline is readable at a glance
/// without reading every label.
enum class NvtxCategory : unsigned {
    Ingress = 0xFF4E79A7,   // blue   — queueing and admission
    Batching = 0xFFF28E2B,  // orange — batch formation
    Transfer = 0xFF59A14F,  // green  — host/device copies
    Compute = 0xFFE15759,   // red    — kernel execution
    Sampling = 0xFFB07AA1,  // purple — token selection
};

/// RAII range. Pushing and popping by hand is exactly the kind of bookkeeping
/// an early return or an exception gets wrong, and an unbalanced range makes
/// the whole timeline unreadable rather than merely losing one label.
class NvtxRange {
public:
    explicit NvtxRange(const char* name, NvtxCategory category = NvtxCategory::Compute) {
#ifdef CUDAFORGE_ENABLE_NVTX
        nvtxEventAttributes_t attributes = {};
        attributes.version = NVTX_VERSION;
        attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        attributes.colorType = NVTX_COLOR_ARGB;
        attributes.color = static_cast<unsigned>(category);
        attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
        attributes.message.ascii = name;
        nvtxRangePushEx(&attributes);
#else
        (void)name;
        (void)category;
#endif
    }

    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
    NvtxRange(NvtxRange&&) = delete;
    NvtxRange& operator=(NvtxRange&&) = delete;

    ~NvtxRange() {
#ifdef CUDAFORGE_ENABLE_NVTX
        nvtxRangePop();
#endif
    }
};

/// Instantaneous marker rather than a range. Useful for points that have no
/// duration — a batch closing, a queue rejection — which show as ticks on the
/// timeline instead of bars.
inline void nvtx_mark(const char* name) {
#ifdef CUDAFORGE_ENABLE_NVTX
    nvtxMarkA(name);
#else
    (void)name;
#endif
}

/// Names the calling thread in the timeline. Without this, Nsight shows numeric
/// thread ids and working out which row is the batcher means guessing from the
/// activity pattern.
inline void nvtx_name_thread(const std::string& name) {
#ifdef CUDAFORGE_ENABLE_NVTX
    nvtxNameOsThreadA(static_cast<std::uint32_t>(::pthread_self()), name.c_str());
#else
    (void)name;
#endif
}

}  // namespace cudaforge

/// Annotates the enclosing scope. The `__LINE__` suffix lets two ranges coexist
/// in one scope without colliding.
#define CUDAFORGE_NVTX_CONCAT_INNER(a, b) a##b
#define CUDAFORGE_NVTX_CONCAT(a, b) CUDAFORGE_NVTX_CONCAT_INNER(a, b)
#define CUDAFORGE_NVTX_RANGE(name, category) \
    ::cudaforge::NvtxRange CUDAFORGE_NVTX_CONCAT(cudaforge_nvtx_, __LINE__)(name, category)

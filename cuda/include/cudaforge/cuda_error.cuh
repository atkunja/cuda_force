#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace cudaforge {

/// Carries the CUDA status code alongside the message so callers can branch on
/// the specific failure — `cudaErrorMemoryAllocation` is recoverable by
/// trimming a cache and retrying, while `cudaErrorIllegalAddress` has already
/// poisoned the context and is not.
class CudaError : public std::runtime_error {
public:
    CudaError(cudaError_t status, std::string message)
        : std::runtime_error(std::move(message)), status_(status) {}

    [[nodiscard]] cudaError_t status() const noexcept { return status_; }

    /// True when the CUDA context is unusable and the process should not
    /// attempt to continue issuing work.
    [[nodiscard]] bool is_sticky() const noexcept {
        switch (status_) {
            case cudaErrorIllegalAddress:
            case cudaErrorLaunchFailure:
            case cudaErrorLaunchTimeout:
            case cudaErrorHardwareStackError:
            case cudaErrorIllegalInstruction:
            case cudaErrorMisalignedAddress:
            case cudaErrorECCUncorrectable:
                return true;
            default:
                return false;
        }
    }

private:
    cudaError_t status_;
};

namespace detail {

[[noreturn]] inline void throw_cuda_error(cudaError_t status, const char* expression,
                                          const char* file, int line) {
    std::ostringstream message;
    message << "CUDA error " << static_cast<int>(status) << " (" << cudaGetErrorName(status)
            << "): " << cudaGetErrorString(status) << "\n"
            << "  at " << file << ":" << line << "\n"
            << "  while evaluating: " << expression;
    throw CudaError(status, message.str());
}

}  // namespace detail

/// Wraps any CUDA runtime call that returns a status.
///
/// Silently ignoring these is the single most common source of CUDA bugs that
/// surface thousands of lines later as an unrelated illegal access, because the
/// failure is asynchronous and the context stays poisoned. Every call site in
/// this project goes through this macro.
#define CUDAFORGE_CHECK(expr)                                                                    \
    do {                                                                                         \
        const cudaError_t cudaforge_status_ = (expr);                                            \
        if (cudaforge_status_ != cudaSuccess) {                                                  \
            ::cudaforge::detail::throw_cuda_error(cudaforge_status_, #expr, __FILE__, __LINE__); \
        }                                                                                        \
    } while (false)

/// Checks a kernel launch.
///
/// Two calls are needed and they catch different things:
///   - `cudaGetLastError()` catches launch-configuration errors (bad grid or
///     block dimensions, too much shared memory) which are reported
///     synchronously at launch.
///   - `cudaStreamSynchronize()` catches faults inside the kernel, which are
///     only reported once the work has actually run.
///
/// The synchronising half destroys the overlap the scheduler exists to create,
/// so it is compiled in only for debug builds. Release builds keep the cheap
/// synchronous check and rely on the next stream synchronisation to surface
/// execution faults.
#ifdef CUDAFORGE_DEBUG_SYNC
#define CUDAFORGE_CHECK_LAUNCH(stream)                  \
    do {                                                \
        CUDAFORGE_CHECK(cudaGetLastError());            \
        CUDAFORGE_CHECK(cudaStreamSynchronize(stream)); \
    } while (false)
#else
#define CUDAFORGE_CHECK_LAUNCH(stream)       \
    do {                                     \
        (void)(stream);                      \
        CUDAFORGE_CHECK(cudaGetLastError()); \
    } while (false)
#endif

}  // namespace cudaforge

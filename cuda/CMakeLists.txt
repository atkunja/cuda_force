# Only reached when CUDAFORGE_ENABLE_CUDA is ON. Everything here needs nvcc and
# an NVIDIA GPU to be useful; nothing in cpp/ depends on this target.

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # 80 = Ampere, 86 = Ampere consumer, 89 = Ada, 90 = Hopper.
  #
  # Turing (75) is deliberately excluded from the default. The BF16 paths need
  # compute capability 8.0: bfloat16 is emulated below it, so a Turing build
  # either fails or produces something markedly slower than the FP16 path it
  # would silently be worse than. Anyone targeting Turing should set this
  # explicitly and use the FP16 entry points.
  #
  # Wide by default so a clone builds on whatever modern card is available;
  # narrow it to the target for faster builds and smaller binaries.
  set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
endif()

if(CMAKE_CUDA_ARCHITECTURES MATCHES "7[0-5]")
  message(WARNING
    "Compute capability < 8.0 requested: the BF16 kernels require 8.0. "
    "Use the FP16 entry points on Turing and earlier.")
endif()

add_library(cudaforge_kernels
  src/activations.cu
  src/fused_norm.cu
  src/reduction.cu
  src/softmax.cu
  src/rmsnorm.cu
  src/lora_linear.cu
  src/quantization.cu
  src/gpu_scheduler.cu
  src/paged_attention.cu
  src/tensor_core_matmul.cu
)
add_library(cudaforge::kernels ALIAS cudaforge_kernels)

target_include_directories(cudaforge_kernels
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/cpp/include>
         $<INSTALL_INTERFACE:include>)

set_target_properties(cudaforge_kernels PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  # C++20, matching the portable runtime. It is not optional: the CUDA targets
  # include cpp/include headers that use concepts and std::bit_ceil, so building
  # them as C++17 fails on the shared code rather than on anything CUDA. nvcc
  # has supported C++20 since CUDA 12.0.
  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON)

target_link_libraries(cudaforge_kernels PUBLIC cudaforge::runtime)

target_compile_options(cudaforge_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  # Line info is what maps Nsight Compute's per-instruction counters back to
  # source. It costs nothing at runtime, unlike -G, which disables optimisation
  # entirely and makes any measurement meaningless.
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

option(CUDAFORGE_ENABLE_NVTX "Emit NVTX ranges for Nsight Systems" OFF)
if(CUDAFORGE_ENABLE_NVTX)
  # Header-only in CUDA 11+. Compiled out entirely when this is OFF, so the
  # release path pays nothing for the annotations.
  target_compile_definitions(cudaforge_kernels PUBLIC CUDAFORGE_ENABLE_NVTX)
  message(STATUS "  NVTX      : ON")
endif()

option(CUDAFORGE_DEBUG_SYNC "Synchronise and check after every kernel launch" OFF)
if(CUDAFORGE_DEBUG_SYNC)
  target_compile_definitions(cudaforge_kernels PUBLIC CUDAFORGE_DEBUG_SYNC)
endif()

if(CUDAFORGE_BUILD_BENCHMARKS)
  add_executable(bench_kernels ${CMAKE_SOURCE_DIR}/benchmarks/benchmark_kernels.cu)
  target_link_libraries(bench_kernels PRIVATE cudaforge::kernels)
  target_include_directories(bench_kernels PRIVATE ${CMAKE_SOURCE_DIR}/benchmarks)
  set_target_properties(bench_kernels PROPERTIES CUDA_STANDARD 20)
endif()

# Only reached when CUDAFORGE_ENABLE_CUDA is ON. Everything here needs nvcc and
# an NVIDIA GPU to be useful; nothing in cpp/ depends on this target.

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # 75 = Turing, 80 = Ampere, 86 = Ampere consumer, 89 = Ada, 90 = Hopper.
  # Wide by default so a clone builds on whatever is available; narrow this to
  # the target architecture for faster builds and smaller binaries.
  set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
endif()

add_library(cudaforge_kernels
  src/reduction.cu
  src/softmax.cu
  src/rmsnorm.cu
  src/lora_linear.cu
  src/quantization.cu
  src/gpu_scheduler.cu
)
add_library(cudaforge::kernels ALIAS cudaforge_kernels)

target_include_directories(cudaforge_kernels
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/cpp/include>
         $<INSTALL_INTERFACE:include>)

set_target_properties(cudaforge_kernels PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_STANDARD 17
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

if(CUDAFORGE_BUILD_TESTS)
  add_subdirectory(${CMAKE_SOURCE_DIR}/tests/cuda ${CMAKE_BINARY_DIR}/tests/cuda)
endif()

if(CUDAFORGE_BUILD_BENCHMARKS)
  add_executable(bench_kernels ${CMAKE_SOURCE_DIR}/benchmarks/benchmark_kernels.cu)
  target_link_libraries(bench_kernels PRIVATE cudaforge::kernels)
  target_include_directories(bench_kernels PRIVATE ${CMAKE_SOURCE_DIR}/benchmarks)
  set_target_properties(bench_kernels PROPERTIES CUDA_STANDARD 17)
endif()

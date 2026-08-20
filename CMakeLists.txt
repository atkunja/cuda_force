cmake_minimum_required(VERSION 3.24)

project(
  CudaForge
  VERSION 0.1.0
  DESCRIPTION "GPU-native LLM fine-tuning and concurrent inference runtime"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

option(CUDAFORGE_ENABLE_CUDA "Build CUDA kernels and the GPU scheduler" OFF)
option(CUDAFORGE_BUILD_TESTS "Build the C++ test suite" ON)
option(CUDAFORGE_BUILD_BENCHMARKS "Build the C++ benchmarks" ON)
option(CUDAFORGE_SANITIZER "Enable a sanitizer: address, thread, undefined, or OFF" OFF)

include(cmake/CompilerWarnings.cmake)
include(cmake/Sanitizers.cmake)

# The portable runtime never depends on CUDA. Keeping it a separate target is
# what allows the full concurrency layer to be built and tested on a host with
# no NVIDIA hardware, which is the common development case.
add_subdirectory(cpp)

if(CUDAFORGE_ENABLE_CUDA)
  enable_language(CUDA)
  add_subdirectory(cuda)
endif()

if(CUDAFORGE_BUILD_TESTS)
  enable_testing()

  # tests/cpp first: it fetches Catch2 and includes the module that defines
  # catch_discover_tests, which tests/cuda then uses. Adding the CUDA tests from
  # cuda/CMakeLists.txt instead put them before that happened, so a CUDA-enabled
  # configure failed on an undefined function.
  add_subdirectory(tests/cpp)

  if(CUDAFORGE_ENABLE_CUDA)
    add_subdirectory(tests/cuda)
  endif()
endif()

if(CUDAFORGE_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

message(STATUS "CudaForge ${PROJECT_VERSION}")
message(STATUS "  build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA       : ${CUDAFORGE_ENABLE_CUDA}")
message(STATUS "  tests      : ${CUDAFORGE_BUILD_TESTS}")
message(STATUS "  sanitizer  : ${CUDAFORGE_SANITIZER}")

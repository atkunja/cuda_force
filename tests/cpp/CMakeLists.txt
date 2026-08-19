include(FetchContent)

# Pinned to a tag rather than a branch: an unpinned test framework makes CI
# failures depend on when the job ran, which is the opposite of what a test
# suite is for.
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.7.1
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(Catch2)

add_executable(cudaforge_tests
  test_concurrent_queue.cpp
  test_thread_pool.cpp
  test_latency_histogram.cpp
  test_metrics.cpp
  test_dynamic_batcher.cpp
  test_memory_pool.cpp
  test_config.cpp
)

target_link_libraries(cudaforge_tests
  PRIVATE cudaforge::runtime cudaforge_warnings Catch2::Catch2WithMain)

cudaforge_apply_sanitizer(cudaforge_tests)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(cudaforge_tests)

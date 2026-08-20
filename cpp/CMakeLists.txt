# Portable runtime: no CUDA, no PyTorch, no third-party runtime dependencies.
# Anything added here must keep building on a host with no NVIDIA hardware.

add_library(cudaforge_runtime
  src/kv_cache.cpp
  src/kv_cache_manager.cpp
  src/metrics.cpp
  src/dynamic_batcher.cpp
)
add_library(cudaforge::runtime ALIAS cudaforge_runtime)

target_include_directories(cudaforge_runtime
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>)

target_compile_features(cudaforge_runtime PUBLIC cxx_std_20)
target_link_libraries(cudaforge_runtime PRIVATE cudaforge_warnings)

find_package(Threads REQUIRED)
target_link_libraries(cudaforge_runtime PUBLIC Threads::Threads)

cudaforge_apply_sanitizer(cudaforge_runtime)

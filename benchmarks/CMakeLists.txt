# Benchmarks are plain executables rather than a framework harness: each one
# emits JSON to stdout so the results can be captured, diffed and plotted
# without a parser that understands a framework's output format.

set(CUDAFORGE_BENCHMARKS
  bench_scheduler:benchmark_scheduler.cpp
  bench_memory:benchmark_memory.cpp
  bench_queue:benchmark_queue.cpp
  bench_histogram:benchmark_histogram.cpp)

foreach(entry IN LISTS CUDAFORGE_BENCHMARKS)
  string(REPLACE ":" ";" parts "${entry}")
  list(GET parts 0 target)
  list(GET parts 1 source)

  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE cudaforge::runtime cudaforge_warnings)

  # The runtime library is compiled with the sanitizer, so every executable
  # linking it must be too — otherwise the sanitizer's own symbols are
  # undefined at link time. Omitting this broke the ThreadSanitizer build while
  # the test target, which did apply it, kept passing.
  cudaforge_apply_sanitizer(${target})
endforeach()

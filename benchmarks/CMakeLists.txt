# Benchmarks are plain executables rather than a framework harness: each one
# emits JSON to stdout so the results can be captured, diffed and plotted
# without a parser that understands a framework's output format.

add_executable(bench_scheduler benchmark_scheduler.cpp)
target_link_libraries(bench_scheduler PRIVATE cudaforge::runtime cudaforge_warnings)

add_executable(bench_memory benchmark_memory.cpp)
target_link_libraries(bench_memory PRIVATE cudaforge::runtime cudaforge_warnings)

add_executable(bench_queue benchmark_queue.cpp)
target_link_libraries(bench_queue PRIVATE cudaforge::runtime cudaforge_warnings)

add_executable(bench_histogram benchmark_histogram.cpp)
target_link_libraries(bench_histogram PRIVATE cudaforge::runtime cudaforge_warnings)

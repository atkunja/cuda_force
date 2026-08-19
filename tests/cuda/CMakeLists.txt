# CUDA correctness tests. Only configured when CUDAFORGE_ENABLE_CUDA is ON, and
# they need an actual device at run time, not just nvcc.

add_executable(cudaforge_cuda_tests
  test_activations.cu
  test_fused_norm.cu
  test_reduced_precision.cu
  test_reduction.cu
  test_softmax.cu
  test_rmsnorm.cu
  test_lora.cu
  test_quantization.cu
  test_scheduler.cu
)

target_link_libraries(cudaforge_cuda_tests
  PRIVATE cudaforge::kernels Catch2::Catch2WithMain)

set_target_properties(cudaforge_cuda_tests PROPERTIES CUDA_STANDARD 17)

catch_discover_tests(cudaforge_cuda_tests)

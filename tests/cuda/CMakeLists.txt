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
  test_paged_attention.cu
  test_tensor_core_matmul.cu
)

target_link_libraries(cudaforge_cuda_tests
  PRIVATE cudaforge::kernels Catch2::Catch2WithMain)

# C++20 for the same reason as the kernel library: these link against the
# portable runtime's headers, which use concepts.
set_target_properties(cudaforge_cuda_tests PROPERTIES CUDA_STANDARD 20)

catch_discover_tests(cudaforge_cuda_tests)

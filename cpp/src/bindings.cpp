// PyTorch extension entry point.
//
// The operators are registered through the dispatcher (`TORCH_LIBRARY`) rather
// than exposed as plain pybind functions. That matters for three reasons:
//
//   * they compose with `torch.compile`, autograd and meta-tensor tracing,
//   * they can be called from TorchScript and from C++ without going through
//     Python,
//   * dispatch keys let the CUDA and CPU implementations be registered
//     separately, which is what makes the CPU fallback a first-class path
//     rather than an if-statement in Python.
//
// This file compiles with or without CUDA. Without it, only the CPU
// implementations are registered and the Python layer sees the operators as
// available but CPU-only — which is what makes the package importable and
// testable on a machine with no GPU.

#include <torch/extension.h>

#include <string>
#include <vector>

#ifdef CUDAFORGE_WITH_CUDA
#include <c10/cuda/CUDAStream.h>

#include "cudaforge/lora_linear.cuh"
#include "cudaforge/quantization.cuh"
#include "cudaforge/reduction.cuh"
#include "cudaforge/rmsnorm.cuh"
#include "cudaforge/softmax.cuh"
#endif

namespace cudaforge::bindings {
namespace {

void check_last_dim_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_contiguous(),
                name, " must be contiguous; call .contiguous() first. ",
                "The kernels index rows as `base + row * cols`, which is only "
                "valid for a contiguous row-major layout.");
}

void check_float32(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
                name, " must be float32, got ", tensor.scalar_type());
}

void check_2d(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.dim() == 2, name, " must be 2-D, got ", tensor.dim(), " dimensions");
}

#ifdef CUDAFORGE_WITH_CUDA

/// The current stream from PyTorch's stream pool, not a stream of our own.
///
/// Using a private stream would leave our kernels unordered with respect to
/// every other PyTorch operation on the tensors, which is a correctness bug:
/// the framework assumes ops it issued are ordered against ours.
cudaStream_t current_stream() {
    return at::cuda::getCurrentCUDAStream().stream();
}

torch::Tensor rmsnorm_cuda(const torch::Tensor& input, const torch::Tensor& weight,
                           double eps) {
    check_2d(input, "input");
    check_float32(input, "input");
    check_float32(weight, "weight");
    check_last_dim_contiguous(input, "input");
    check_last_dim_contiguous(weight, "weight");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1-D, got ", weight.dim(), " dimensions");
    TORCH_CHECK(weight.size(0) == input.size(1),
                "weight length ", weight.size(0), " does not match the input's last dimension ",
                input.size(1));

    torch::Tensor output = torch::empty_like(input);
    launch_rmsnorm(input.data_ptr<float>(), weight.data_ptr<float>(),
                   output.data_ptr<float>(), static_cast<int>(input.size(0)),
                   static_cast<int>(input.size(1)), static_cast<float>(eps),
                   RMSNormKernel::Vectorised, current_stream());
    return output;
}

torch::Tensor softmax_cuda(const torch::Tensor& input) {
    check_2d(input, "input");
    check_float32(input, "input");
    check_last_dim_contiguous(input, "input");

    torch::Tensor output = torch::empty_like(input);
    launch_softmax(input.data_ptr<float>(), output.data_ptr<float>(),
                   static_cast<int>(input.size(0)), static_cast<int>(input.size(1)),
                   SoftmaxKernel::Online, current_stream());
    return output;
}

torch::Tensor lora_linear_cuda(const torch::Tensor& x, const torch::Tensor& w,
                               const torch::Tensor& a, const torch::Tensor& b, double scale) {
    check_2d(x, "x");
    check_2d(w, "w");
    check_2d(a, "a");
    check_2d(b, "b");
    for (const auto& [tensor, name] :
         std::vector<std::pair<torch::Tensor, const char*>>{
             {x, "x"}, {w, "w"}, {a, "a"}, {b, "b"}}) {
        check_float32(tensor, name);
        check_last_dim_contiguous(tensor, name);
    }

    const auto batch = static_cast<int>(x.size(0));
    const auto in_features = static_cast<int>(x.size(1));
    const auto out_features = static_cast<int>(w.size(1));
    const auto rank = static_cast<int>(a.size(1));

    TORCH_CHECK(w.size(0) == in_features, "w must be [in_features, out_features]; got [",
                w.size(0), ", ", w.size(1), "] for in_features=", in_features);
    TORCH_CHECK(a.size(0) == in_features, "a must be [in_features, rank]; got [", a.size(0),
                ", ", a.size(1), "] for in_features=", in_features);
    TORCH_CHECK(b.size(0) == rank && b.size(1) == out_features,
                "b must be [rank, out_features] = [", rank, ", ", out_features, "]; got [",
                b.size(0), ", ", b.size(1), "]");

    torch::Tensor output = torch::empty({batch, out_features}, x.options());
    torch::Tensor workspace = torch::empty({batch, rank}, x.options());

    launch_lora_linear(x.data_ptr<float>(), w.data_ptr<float>(), a.data_ptr<float>(),
                       b.data_ptr<float>(), output.data_ptr<float>(),
                       workspace.data_ptr<float>(), batch, in_features, out_features, rank,
                       static_cast<float>(scale), LoRAKernel::Fused, current_stream());
    return output;
}

torch::Tensor sum_cuda(const torch::Tensor& input) {
    check_float32(input, "input");
    check_last_dim_contiguous(input, "input");

    torch::Tensor output = torch::zeros({1}, input.options());
    launch_reduce_sum(input.data_ptr<float>(), output.data_ptr<float>(),
                      static_cast<std::size_t>(input.numel()),
                      ReductionKernel::WarpOptimised, current_stream());
    return output.squeeze();
}

std::vector<torch::Tensor> quantize_int8_cuda(const torch::Tensor& input) {
    check_float32(input, "input");
    check_last_dim_contiguous(input, "input");

    const auto count = static_cast<int>(input.numel());
    torch::Tensor quantised =
        torch::empty({count}, input.options().dtype(torch::kInt8));
    torch::Tensor scales = torch::empty({quant_scale_count(count)}, input.options());

    launch_quantize_int8(input.data_ptr<float>(), quantised.data_ptr<std::int8_t>(),
                         scales.data_ptr<float>(), count, current_stream());
    return {quantised.view_as(input), scales};
}

torch::Tensor dequantize_int8_cuda(const torch::Tensor& quantised,
                                   const torch::Tensor& scales) {
    TORCH_CHECK(quantised.scalar_type() == torch::kInt8, "quantised must be int8");
    check_float32(scales, "scales");
    check_last_dim_contiguous(quantised, "quantised");
    check_last_dim_contiguous(scales, "scales");

    const auto count = static_cast<int>(quantised.numel());
    TORCH_CHECK(scales.numel() == quant_scale_count(count), "expected ",
                quant_scale_count(count), " block scales, got ", scales.numel());

    torch::Tensor output = torch::empty(quantised.sizes(), scales.options());
    launch_dequantize_int8(quantised.data_ptr<std::int8_t>(), scales.data_ptr<float>(),
                           output.data_ptr<float>(), count, current_stream());
    return output;
}

#endif  // CUDAFORGE_WITH_CUDA

// ---------------------------------------------------------------------------
// CPU implementations.
//
// These are not stubs. They are the reference semantics the CUDA kernels are
// tested against, and they are what runs when the package is imported on a
// machine with no GPU. Written with ATen ops rather than raw loops so they stay
// correct for any shape and dtype ATen supports.
// ---------------------------------------------------------------------------

torch::Tensor rmsnorm_cpu(const torch::Tensor& input, const torch::Tensor& weight,
                          double eps) {
    check_2d(input, "input");
    TORCH_CHECK(weight.size(0) == input.size(1),
                "weight length ", weight.size(0), " does not match the input's last dimension ",
                input.size(1));
    // Computed in float32 even for half inputs, matching the kernel's
    // accumulator rule; see rmsnorm.cuh.
    const torch::Tensor promoted = input.to(torch::kFloat32);
    const torch::Tensor variance = promoted.pow(2).mean(-1, /*keepdim=*/true);
    const torch::Tensor normalised = promoted * torch::rsqrt(variance + eps);
    return (normalised * weight.to(torch::kFloat32)).to(input.scalar_type());
}

torch::Tensor softmax_cpu(const torch::Tensor& input) {
    check_2d(input, "input");
    return torch::softmax(input, /*dim=*/-1);
}

torch::Tensor lora_linear_cpu(const torch::Tensor& x, const torch::Tensor& w,
                              const torch::Tensor& a, const torch::Tensor& b, double scale) {
    return torch::matmul(x, w) + scale * torch::matmul(torch::matmul(x, a), b);
}

torch::Tensor sum_cpu(const torch::Tensor& input) { return input.sum(); }

std::vector<torch::Tensor> quantize_int8_cpu(const torch::Tensor& input) {
    const auto count = input.numel();
    const auto block = static_cast<std::int64_t>(64);
    const auto blocks = (count + block - 1) / block;

    const torch::Tensor flat = input.reshape({-1}).to(torch::kFloat32);
    const torch::Tensor padded = torch::constant_pad_nd(flat, {0, blocks * block - count}, 0.0);
    const torch::Tensor grouped = padded.reshape({blocks, block});

    torch::Tensor scales = std::get<0>(grouped.abs().max(/*dim=*/1)) / 127.0;
    // An all-zero block would otherwise divide by zero; a scale of 1 maps it to
    // zero and back exactly, matching the kernel.
    scales = torch::where(scales > 0, scales, torch::ones_like(scales));

    const torch::Tensor quantised =
        torch::clamp(torch::round(grouped / scales.unsqueeze(1)), -127, 127)
            .reshape({-1})
            .slice(0, 0, count)
            .to(torch::kInt8);
    return {quantised.view_as(input), scales};
}

torch::Tensor dequantize_int8_cpu(const torch::Tensor& quantised,
                                  const torch::Tensor& scales) {
    const auto count = quantised.numel();
    const auto block = static_cast<std::int64_t>(64);
    const auto blocks = (count + block - 1) / block;

    const torch::Tensor flat = quantised.reshape({-1}).to(torch::kFloat32);
    const torch::Tensor padded = torch::constant_pad_nd(flat, {0, blocks * block - count}, 0.0);
    const torch::Tensor grouped = padded.reshape({blocks, block});
    return (grouped * scales.unsqueeze(1)).reshape({-1}).slice(0, 0, count).view_as(quantised);
}

bool cuda_kernels_available() {
#ifdef CUDAFORGE_WITH_CUDA
    return true;
#else
    return false;
#endif
}

}  // namespace
}  // namespace cudaforge::bindings

TORCH_LIBRARY(cudaforge, m) {
    m.def("rmsnorm(Tensor input, Tensor weight, float eps) -> Tensor");
    m.def("softmax(Tensor input) -> Tensor");
    m.def("lora_linear(Tensor x, Tensor w, Tensor a, Tensor b, float scale) -> Tensor");
    m.def("sum(Tensor input) -> Tensor");
    m.def("quantize_int8(Tensor input) -> Tensor[]");
    m.def("dequantize_int8(Tensor quantised, Tensor scales) -> Tensor");
}

TORCH_LIBRARY_IMPL(cudaforge, CPU, m) {
    m.impl("rmsnorm", &cudaforge::bindings::rmsnorm_cpu);
    m.impl("softmax", &cudaforge::bindings::softmax_cpu);
    m.impl("lora_linear", &cudaforge::bindings::lora_linear_cpu);
    m.impl("sum", &cudaforge::bindings::sum_cpu);
    m.impl("quantize_int8", &cudaforge::bindings::quantize_int8_cpu);
    m.impl("dequantize_int8", &cudaforge::bindings::dequantize_int8_cpu);
}

#ifdef CUDAFORGE_WITH_CUDA
TORCH_LIBRARY_IMPL(cudaforge, CUDA, m) {
    m.impl("rmsnorm", &cudaforge::bindings::rmsnorm_cuda);
    m.impl("softmax", &cudaforge::bindings::softmax_cuda);
    m.impl("lora_linear", &cudaforge::bindings::lora_linear_cuda);
    m.impl("sum", &cudaforge::bindings::sum_cuda);
    m.impl("quantize_int8", &cudaforge::bindings::quantize_int8_cuda);
    m.impl("dequantize_int8", &cudaforge::bindings::dequantize_int8_cuda);
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CudaForge custom operators";
    m.def("cuda_kernels_available", &cudaforge::bindings::cuda_kernels_available,
          "True when the extension was compiled with CUDA support");
}

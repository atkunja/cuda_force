"""Builds the optional PyTorch C++/CUDA extension.

Packaging metadata lives in pyproject.toml; this file exists only because the
extension has to be configured at build time based on what the machine has.
The rule it implements:

  * CUDA toolkit present  -> build a CUDAExtension with the .cu kernels
  * no CUDA toolkit       -> build a CppExtension with the CPU implementations
  * no working compiler   -> build no extension at all

The third case matters. `pip install cudaforge` on a machine without a
toolchain must still yield a working package, because `cudaforge.ops` falls
back to pure PyTorch. A hard failure here would make the package uninstallable
for anyone who only wants the reference implementations.
"""

from __future__ import annotations

import os
import sys

from setuptools import setup


def build_extensions() -> tuple[list, dict]:
    try:
        from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
    except ImportError:
        print("torch not importable; skipping the native extension", file=sys.stderr)
        return [], {}

    if os.environ.get("CUDAFORGE_SKIP_EXTENSION"):
        print("CUDAFORGE_SKIP_EXTENSION set; skipping the native extension", file=sys.stderr)
        return [], {}

    include_dirs = [
        os.path.abspath("cpp/include"),
        os.path.abspath("cuda/include"),
    ]
    sources = ["cpp/src/bindings.cpp", "cpp/src/metrics.cpp", "cpp/src/dynamic_batcher.cpp"]

    # CUDA_HOME is what torch itself uses to locate nvcc. It is the right gate
    # rather than torch.cuda.is_available(), because building the kernels needs
    # a toolkit, not a visible device — a build host and the machine that runs
    # the result are frequently not the same.
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is not None:
        sources += [
            "cuda/src/activations.cu",
            "cuda/src/reduction.cu",
            "cuda/src/softmax.cu",
            "cuda/src/rmsnorm.cu",
            "cuda/src/lora_linear.cu",
            "cuda/src/quantization.cu",
            "cuda/src/gpu_scheduler.cu",
        ]
        extension = CUDAExtension(
            name="cudaforge._C",
            sources=sources,
            include_dirs=include_dirs,
            define_macros=[("CUDAFORGE_WITH_CUDA", "1")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20"],
                "nvcc": [
                    "-O3",
                    "--expt-relaxed-constexpr",
                    # Maps profiler counters back to source lines without the
                    # optimisation loss that -G would cause.
                    "-lineinfo",
                ],
            },
        )
        print("building with CUDA support", file=sys.stderr)
    else:
        extension = CppExtension(
            name="cudaforge._C",
            sources=[source for source in sources if source.endswith((".cpp", ".cc"))],
            include_dirs=include_dirs,
            extra_compile_args=["-O3", "-std=c++20"],
        )
        print("no CUDA toolkit found; building CPU-only operators", file=sys.stderr)

    return [extension], {"build_ext": BuildExtension}


extensions, cmdclass = build_extensions()

setup(ext_modules=extensions, cmdclass=cmdclass)

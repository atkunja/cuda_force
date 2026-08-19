"""CudaForge: GPU-native LLM fine-tuning and concurrent inference runtime.

The public surface is deliberately small. Everything here works without a GPU —
operators fall back to reference PyTorch implementations, and the batching and
metrics layers are pure Python. What changes on CUDA hardware is which
implementation the dispatcher selects, not the API.
"""

from __future__ import annotations

from cudaforge.config import EngineConfig, GenerationConfig
from cudaforge.engine import EngineClosedError, InferenceEngine, Response
from cudaforge.metrics import LatencyHistogram, MetricsRegistry, MetricsSnapshot
from cudaforge.ops import (
    CUDA_KERNELS_AVAILABLE,
    KERNEL_DTYPES,
    backend_report,
    dequantize_int8,
    fused_residual_rmsnorm,
    gelu,
    kernel_supports,
    lora_linear,
    quantize_int8,
    rmsnorm,
    silu,
    softmax,
    sum_reduce,
    swiglu,
)
from cudaforge.runners import EchoRunner, GenerationResult, ModelRunner
from cudaforge.scheduler import Batch, BatchTrigger, DynamicBatcher, Request

__version__ = "0.1.0"

__all__ = [
    "CUDA_KERNELS_AVAILABLE",
    "KERNEL_DTYPES",
    "Batch",
    "BatchTrigger",
    "DynamicBatcher",
    "EchoRunner",
    "EngineClosedError",
    "EngineConfig",
    "GenerationConfig",
    "GenerationResult",
    "InferenceEngine",
    "LatencyHistogram",
    "MetricsRegistry",
    "MetricsSnapshot",
    "ModelRunner",
    "Request",
    "Response",
    "__version__",
    "backend_report",
    "dequantize_int8",
    "fused_residual_rmsnorm",
    "gelu",
    "kernel_supports",
    "lora_linear",
    "quantize_int8",
    "rmsnorm",
    "silu",
    "softmax",
    "sum_reduce",
    "swiglu",
]

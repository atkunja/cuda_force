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
    backend_report,
    dequantize_int8,
    lora_linear,
    quantize_int8,
    rmsnorm,
    softmax,
    sum_reduce,
)
from cudaforge.runners import EchoRunner, GenerationResult, ModelRunner
from cudaforge.scheduler import Batch, BatchTrigger, DynamicBatcher, Request

__version__ = "0.1.0"

__all__ = [
    "Batch",
    "BatchTrigger",
    "CUDA_KERNELS_AVAILABLE",
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
    "lora_linear",
    "quantize_int8",
    "rmsnorm",
    "softmax",
    "sum_reduce",
]

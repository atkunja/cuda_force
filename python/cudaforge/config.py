"""Configuration objects for the engine and for generation.

Validation happens in ``__post_init__`` rather than at first use. A batching
configuration that cannot work should fail when it is constructed, not twenty
minutes into a benchmark with a confusing symptom.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class GenerationConfig:
    """Per-request sampling parameters."""

    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {self.max_new_tokens}")
        if self.temperature < 0:
            raise ValueError(f"temperature must not be negative, got {self.temperature}")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.top_k < 0:
            raise ValueError(f"top_k must not be negative, got {self.top_k}")

    @property
    def greedy(self) -> bool:
        """Temperature 0 means argmax, which is what callers mean by greedy."""
        return self.temperature == 0.0


@dataclass
class EngineConfig:
    """Runtime configuration for the inference engine.

    ``max_batch_size`` and ``max_wait_us`` are the two knobs that trade
    throughput against tail latency, and they pull in opposite directions:

    * raising ``max_batch_size`` amortises weight loading over more rows, so
      throughput rises — but the batch takes longer to fill and longer to run,
      so every request in it waits longer;
    * ``max_wait_us`` caps how long an early arrival is held waiting for
      company. It is the direct upper bound on batching-induced queue delay,
      and therefore the main p99 lever.

    A reasonable starting point for ``max_wait_us`` is the single-request
    service time. Below that the batcher rarely fills a batch; above it the
    added delay costs more than the throughput it buys.
    """

    model_name: str = "sshleifer/tiny-gpt2"
    device: str = "auto"
    dtype: str = "auto"

    max_batch_size: int = 16
    max_wait_us: int = 5_000
    queue_capacity: int = 1024
    worker_threads: int = 4
    cuda_streams: int = 4

    max_prompt_chars: int = 8_192
    warmup_iterations: int = 3

    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def __post_init__(self) -> None:
        if self.max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {self.max_batch_size}")
        if self.max_wait_us < 0:
            raise ValueError(f"max_wait_us must not be negative, got {self.max_wait_us}")
        if self.queue_capacity <= 0:
            raise ValueError(f"queue_capacity must be positive, got {self.queue_capacity}")
        if self.worker_threads <= 0:
            raise ValueError(f"worker_threads must be positive, got {self.worker_threads}")
        if self.cuda_streams <= 0:
            raise ValueError(f"cuda_streams must be positive, got {self.cuda_streams}")
        # A queue smaller than a batch guarantees the batcher never reaches
        # max_batch_size, silently capping throughput at the queue depth.
        if self.queue_capacity < self.max_batch_size:
            raise ValueError(
                f"queue_capacity ({self.queue_capacity}) must be at least "
                f"max_batch_size ({self.max_batch_size})"
            )

    @property
    def max_wait_seconds(self) -> float:
        return self.max_wait_us / 1e6

    def resolve_device(self) -> torch.device:
        """Pick a device, preferring CUDA and falling back through MPS to CPU.

        MPS is included because it is what an Apple Silicon development machine
        actually has. It runs the reference operators, not the custom kernels —
        those are CUDA-only and there is no portable path for them.
        """
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def resolve_dtype(self) -> torch.dtype:
        """Pick a dtype appropriate to the resolved device.

        bfloat16 is preferred over float16 on Ampere and later: it has the same
        exponent range as float32, so it does not need loss scaling to keep
        gradients from flushing to zero. On older GPUs there is no bf16 hardware
        path, so float16 is the right choice there.
        """
        if self.dtype != "auto":
            return getattr(torch, self.dtype)

        device = self.resolve_device()
        if device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        # CPU float16 is emulated and slower than float32; MPS bfloat16 support
        # is incomplete. float32 is the honest default off CUDA.
        return torch.float32

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> EngineConfig:
        payload = dict(values)
        generation = payload.pop("generation", None)
        config = cls(**payload)
        if generation is not None:
            config.generation = GenerationConfig(**generation)
            config.__post_init__()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> EngineConfig:
        import yaml  # noqa: PLC0415  (optional dependency, imported on demand)

        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})

    def describe(self) -> str:
        device = self.resolve_device()
        return (
            f"model={self.model_name} device={device} dtype={self.resolve_dtype()} "
            f"max_batch_size={self.max_batch_size} max_wait_us={self.max_wait_us} "
            f"queue_capacity={self.queue_capacity} streams={self.cuda_streams}"
        )

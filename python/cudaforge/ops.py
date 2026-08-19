"""Custom operators, with reference implementations as the fallback path.

## Dispatch

Each function tries the compiled extension and falls back to pure PyTorch:

    rmsnorm(x, w)  ->  torch.ops.cudaforge.rmsnorm   (extension present)
                   ->  _rmsnorm_reference            (otherwise)

The fallback is not a placeholder. It is the reference semantics the CUDA
kernels are tested against, so it is exercised on every CPU test run rather
than rotting in a branch nobody takes. That is also what makes this package
usable on a machine with no GPU, which is where it was developed.

## Why the extension may be absent

Three separate things can be missing and they fail differently:

  * no NVIDIA GPU        -> extension may still import; CUDA kernels unusable
  * no CUDA toolkit      -> extension builds CPU-only
  * no compiler at all   -> no extension; everything falls back

`backend_report()` reports which of these applies, because "it's slow" and
"it silently fell back" look identical from the outside otherwise.
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass

import torch

_LOG = logging.getLogger(__name__)

# Block size for int8 quantisation. Must match kQuantBlockSize in
# cuda/include/cudaforge/quantization.cuh — the two implementations produce
# different scales otherwise, and the parity tests would fail with a confusing
# numerical error rather than a clear mismatch.
QUANT_BLOCK_SIZE = 64


def _load_extension() -> tuple[bool, bool, str]:
    """Return (extension_loaded, cuda_kernels_compiled, explanation)."""
    if importlib.util.find_spec("cudaforge._C") is None:
        return False, False, "extension not built; using reference implementations"

    try:
        from cudaforge import _C  # imported lazily: optional dependency
    except ImportError as error:
        return False, False, f"extension present but failed to load: {error}"

    cuda_compiled = bool(getattr(_C, "cuda_kernels_available", lambda: False)())
    if cuda_compiled:
        return True, True, "extension loaded with CUDA kernels"
    return True, False, "extension loaded, compiled without CUDA"


_EXTENSION_LOADED, _CUDA_COMPILED, _LOAD_MESSAGE = _load_extension()

#: True when the compiled extension exposes CUDA kernels *and* a device is
#: visible. Both halves matter: a CUDA-compiled extension on a machine with no
#: GPU still cannot run them.
CUDA_KERNELS_AVAILABLE: bool = _CUDA_COMPILED and torch.cuda.is_available()

if not CUDA_KERNELS_AVAILABLE:
    _LOG.info("cudaforge: %s", _LOAD_MESSAGE)


@dataclass(frozen=True)
class BackendReport:
    """Why a particular implementation is being used."""

    extension_loaded: bool
    cuda_compiled: bool
    cuda_device_available: bool
    device_name: str | None
    message: str

    @property
    def using_custom_kernels(self) -> bool:
        return self.extension_loaded and self.cuda_compiled and self.cuda_device_available

    def __str__(self) -> str:
        path = "custom CUDA kernels" if self.using_custom_kernels else "PyTorch reference"
        device = self.device_name or "cpu"
        return f"cudaforge: {path} on {device} ({self.message})"


def backend_report() -> BackendReport:
    """Describe the active implementation path.

    Worth calling before benchmarking anything: a silent fallback to the
    reference path will look like a very slow custom kernel.
    """
    device_available = torch.cuda.is_available()
    return BackendReport(
        extension_loaded=_EXTENSION_LOADED,
        cuda_compiled=_CUDA_COMPILED,
        cuda_device_available=device_available,
        device_name=torch.cuda.get_device_name(0) if device_available else None,
        message=_LOAD_MESSAGE,
    )


def _dispatch_available(name: str, tensor: torch.Tensor) -> bool:
    """True when the registered operator should handle this tensor."""
    if not _EXTENSION_LOADED:
        return False
    if not hasattr(torch.ops.cudaforge, name):
        return False
    # The CUDA implementation is only registered when compiled with CUDA. For a
    # CPU tensor the dispatcher's CPU implementation is registered either way.
    if tensor.is_cuda:
        return _CUDA_COMPILED
    return True


# ---------------------------------------------------------------------------
# Reference implementations.
#
# These define the semantics. When a CUDA kernel and one of these disagree, the
# kernel is wrong.
# ---------------------------------------------------------------------------


def _rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # Promoted to float32 before squaring regardless of input dtype. In float16
    # a single activation of magnitude 256 squares to infinity, and the whole
    # row becomes NaN; see rmsnorm.cuh.
    promoted = x.float()
    variance = promoted.pow(2).mean(dim=-1, keepdim=True)
    normalised = promoted * torch.rsqrt(variance + eps)
    return (normalised * weight.float()).to(x.dtype)


def _softmax_reference(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


def _lora_linear_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return x @ weight + scale * ((x @ lora_a) @ lora_b)


def _quantize_int8_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = x.reshape(-1).float()
    padding = (-flat.numel()) % QUANT_BLOCK_SIZE
    if padding:
        flat = torch.nn.functional.pad(flat, (0, padding))
    grouped = flat.view(-1, QUANT_BLOCK_SIZE)

    scales = grouped.abs().amax(dim=1) / 127.0
    # An all-zero block has absmax 0; a scale of 1 maps it to zero and back
    # exactly, and avoids a division by zero. Matches the kernel.
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))

    quantised = torch.clamp(torch.round(grouped / scales.unsqueeze(1)), -127, 127)
    trimmed = quantised.reshape(-1)[: x.numel()]
    return trimmed.to(torch.int8).view_as(x), scales


def _dequantize_int8_reference(quantised: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    flat = quantised.reshape(-1).float()
    padding = (-flat.numel()) % QUANT_BLOCK_SIZE
    original = quantised.numel()
    if padding:
        flat = torch.nn.functional.pad(flat, (0, padding))
    grouped = flat.view(-1, QUANT_BLOCK_SIZE)
    restored = (grouped * scales.unsqueeze(1)).reshape(-1)[:original]
    return restored.view_as(quantised)


# ---------------------------------------------------------------------------
# Public operators.
# ---------------------------------------------------------------------------


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Root-mean-square layer normalisation over the last dimension.

    Args:
        x: ``[..., hidden]``. Non-contiguous inputs are made contiguous, because
            the kernels index rows arithmetically.
        weight: ``[hidden]`` learned gain.
        eps: added inside the square root, matching the reference formulation.

    Raises:
        ValueError: if ``weight`` does not match ``x``'s last dimension.
    """
    if weight.ndim != 1:
        raise ValueError(f"weight must be 1-D, got shape {tuple(weight.shape)}")
    if x.shape[-1] != weight.shape[0]:
        raise ValueError(
            f"weight length {weight.shape[0]} does not match the last dimension "
            f"of x, which is {x.shape[-1]}"
        )
    if x.numel() == 0:
        return x.clone()

    x = x.contiguous()
    weight = weight.contiguous()

    if _dispatch_available("rmsnorm", x) and x.ndim == 2 and x.dtype == torch.float32:
        return torch.ops.cudaforge.rmsnorm(x, weight, eps)
    return _rmsnorm_reference(x, weight, eps)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable softmax over the last dimension."""
    if x.numel() == 0:
        return x.clone()

    x = x.contiguous()
    if _dispatch_available("softmax", x) and x.ndim == 2 and x.dtype == torch.float32:
        return torch.ops.cudaforge.softmax(x)
    return _softmax_reference(x)


def lora_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """LoRA-adapted linear layer: ``x @ weight + scale * (x @ lora_a) @ lora_b``.

    Args:
        x: ``[batch, in_features]``
        weight: ``[in_features, out_features]``, the frozen base weight.
        lora_a: ``[in_features, rank]``
        lora_b: ``[rank, out_features]``
        scale: usually ``alpha / rank``.
    """
    if x.shape[-1] != weight.shape[0]:
        raise ValueError(f"x has {x.shape[-1]} input features but weight expects {weight.shape[0]}")
    if lora_a.shape[0] != weight.shape[0]:
        raise ValueError(
            f"lora_a must be [in_features, rank] with in_features={weight.shape[0]}, "
            f"got {tuple(lora_a.shape)}"
        )
    if lora_b.shape[0] != lora_a.shape[1] or lora_b.shape[1] != weight.shape[1]:
        raise ValueError(
            f"lora_b must be [rank, out_features] = "
            f"[{lora_a.shape[1]}, {weight.shape[1]}], got {tuple(lora_b.shape)}"
        )

    tensors = [t.contiguous() for t in (x, weight, lora_a, lora_b)]
    if _dispatch_available("lora_linear", x) and x.ndim == 2 and x.dtype == torch.float32:
        return torch.ops.cudaforge.lora_linear(*tensors, scale)
    return _lora_linear_reference(*tensors, scale)


def sum_reduce(x: torch.Tensor) -> torch.Tensor:
    """Sum of every element, as a 0-D tensor."""
    if x.numel() == 0:
        return torch.zeros((), dtype=x.dtype, device=x.device)

    x = x.contiguous()
    if _dispatch_available("sum", x) and x.dtype == torch.float32:
        return torch.ops.cudaforge.sum(x)
    return x.sum()


def quantize_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-wise symmetric int8 quantisation.

    Returns:
        ``(quantised, scales)`` where ``quantised`` matches ``x``'s shape and
        ``scales`` has one entry per ``QUANT_BLOCK_SIZE`` elements.
    """
    x = x.contiguous()
    if _dispatch_available("quantize_int8", x) and x.dtype == torch.float32:
        quantised, scales = torch.ops.cudaforge.quantize_int8(x)
        return quantised, scales
    return _quantize_int8_reference(x)


def dequantize_int8(quantised: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`quantize_int8`. Lossy by construction."""
    expected = (quantised.numel() + QUANT_BLOCK_SIZE - 1) // QUANT_BLOCK_SIZE
    if scales.numel() != expected:
        raise ValueError(
            f"expected {expected} block scales for {quantised.numel()} elements, "
            f"got {scales.numel()}"
        )

    quantised = quantised.contiguous()
    scales = scales.contiguous()
    if _dispatch_available("dequantize_int8", quantised):
        return torch.ops.cudaforge.dequantize_int8(quantised, scales)
    return _dequantize_int8_reference(quantised, scales)


def quantization_error(x: torch.Tensor) -> torch.Tensor:
    """Maximum absolute round-trip error of int8 quantisation.

    Bounded by half a quantisation step per block, i.e. ``scale / 2``. Useful as
    a sanity check that a quantisation path is behaving, since exceeding that
    bound means something is wrong rather than merely lossy.
    """
    quantised, scales = quantize_int8(x)
    restored = dequantize_int8(quantised, scales)
    return (x - restored).abs().max()

"""LoRA adapter construction and a self-contained reference implementation.

Two things live here and they serve different purposes:

* :func:`attach_lora` wires up PEFT. That is what a real run should use — PEFT
  handles the module surgery, saving, merging and dispatch that a hand-rolled
  version would get subtly wrong.
* :class:`LoRALinear` is a from-scratch implementation of the same maths. It
  exists to be read and to be tested against PEFT, not to replace it. It is
  also what the custom CUDA kernel is validated against.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Literal, cast

import torch
from torch import nn

from training.config import LoRAConfig

_LOG = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank update.

        y = x W^T + b + (alpha / r) * (x A^T) B^T

    ## Initialisation

    ``lora_a`` uses Kaiming-uniform initialisation and ``lora_b`` starts at
    exactly zero. That asymmetry is deliberate and load-bearing: with B at zero
    the product BA is zero, so the adapted model is *numerically identical* to
    the base model at step 0. Training therefore starts from the pretrained
    model rather than from a perturbed one, which is what makes LoRA stable at
    high learning rates.

    Initialising both to zero would leave the product's gradient zero as well,
    and the adapter would never train.

    ## Memory

    The base weight is frozen, so no gradient and no optimiser state is
    allocated for it. For Adam that removes two fp32 tensors the size of the
    weight — usually the dominant term in fine-tuning memory, well above the
    weights themselves.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: int = 16,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if rank > min(in_features, out_features):
            raise ValueError(
                f"rank {rank} exceeds min(in_features, out_features) = "
                f"{min(in_features, out_features)}; the update would not be low-rank"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        # Dropout is applied to the adapter's input, not its output. Applying it
        # to the output would also perturb the frozen path once the update is
        # merged, changing what inference computes.
        update = self.dropout(x) @ self.lora_a.t() @ self.lora_b.t()
        return base + self.scaling * update

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """The single weight equivalent to this layer.

        Merging removes the adapter's runtime cost entirely, which is why LoRA
        adds no inference latency once training is done. The result is exact,
        not an approximation.
        """
        return self.base.weight + self.scaling * (self.lora_b @ self.lora_a)

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """Return a plain ``nn.Linear`` carrying the merged weight."""
        merged = nn.Linear(self.in_features, self.out_features, bias=self.base.bias is not None)
        merged.weight.copy_(self.merged_weight())
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)
        return merged

    @property
    def trainable_parameters(self) -> int:
        return self.lora_a.numel() + self.lora_b.numel()

    @property
    def frozen_parameters(self) -> int:
        total = self.base.weight.numel()
        if self.base.bias is not None:
            total += self.base.bias.numel()
        return total

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, scaling={self.scaling:.3f}"
        )


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return ``(trainable, total)``."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def describe_parameters(model: nn.Module) -> str:
    trainable, total = count_parameters(model)
    share = 100.0 * trainable / total if total else 0.0
    return f"trainable {trainable:,} / {total:,} parameters ({share:.3f}%)"


def attach_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """Wrap ``model`` with PEFT LoRA adapters.

    Raises:
        ImportError: if PEFT is not installed. Deliberately not caught: silently
            training a model with no adapters attached would produce a run that
            looks successful and updates nothing.
    """
    from peft import LoraConfig, TaskType, get_peft_model  # imported lazily: optional dependency

    # `LoRAConfig.__post_init__` has already checked this against the same set
    # PEFT accepts, so the narrowing is justified rather than assumed.
    bias = cast('Literal["none", "all", "lora_only"]', config.bias)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=config.target_modules,
        bias=bias,
    )
    # PEFT annotates this as `PreTrainedModel`, but it accepts any module whose
    # submodules match `target_modules` — which is what the tests exercise with
    # a plain `nn.Module`. Casting rather than narrowing the signature, so
    # callers are not forced to hold a transformers model they do not need.
    adapted = get_peft_model(cast("Any", model), peft_config)
    _LOG.info("attached LoRA: %s", describe_parameters(adapted))
    return adapted

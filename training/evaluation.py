"""Evaluation for causal language models.

Perplexity is the metric because it is what the training objective optimises:
``perplexity = exp(mean token-level cross-entropy)``. Reporting it alongside
loss is not redundant — loss is what the optimiser sees, perplexity is on a
scale a person can reason about (roughly "how many equally likely tokens is the
model choosing between").
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader


@dataclass
class EvalResult:
    loss: float
    perplexity: float
    tokens: int
    batches: int

    def __str__(self) -> str:
        return (
            f"loss {self.loss:.4f}  perplexity {self.perplexity:.2f}  "
            f"({self.tokens:,} tokens over {self.batches} batches)"
        )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> EvalResult:
    """Mean cross-entropy and perplexity over ``loader``.

    Losses are weighted by token count, not averaged over batches. An unweighted
    average would over-weight short batches — and with packed fixed-length
    blocks the batches are equal-length only until the final partial one, which
    is exactly where an unweighted mean goes wrong.
    """
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    batches = 0

    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break

        inputs = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**inputs)

        # Cross-entropy over the shifted sequence: position i predicts i+1, so
        # the last position has no target and contributes no loss.
        tokens = int(inputs["attention_mask"].sum().item()) - inputs["input_ids"].shape[0]
        tokens = max(tokens, 1)

        total_loss += float(outputs.loss.item()) * tokens
        total_tokens += tokens
        batches += 1

    if was_training:
        model.train()

    if total_tokens == 0:
        return EvalResult(loss=float("nan"), perplexity=float("nan"), tokens=0, batches=0)

    mean_loss = total_loss / total_tokens
    # exp() overflows for a badly diverged model; reporting inf is more honest
    # than crashing the eval loop at the moment something has gone wrong.
    perplexity = math.exp(mean_loss) if mean_loss < 700 else float("inf")
    return EvalResult(
        loss=mean_loss, perplexity=perplexity, tokens=total_tokens, batches=batches
    )


def perplexity_from_loss(loss: float) -> float:
    """Convert a mean cross-entropy to perplexity, guarding against overflow."""
    return math.exp(loss) if loss < 700 else float("inf")

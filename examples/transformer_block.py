#!/usr/bin/env python3
"""A transformer block built from the CudaForge operators.

    python examples/transformer_block.py

The individual kernels are easier to justify when you can see them composed the
way they would actually be used. This assembles a LLaMA-style block —
pre-normalisation, attention, SwiGLU feed-forward, residual connections — from
`cudaforge` operators, and checks it against a plain PyTorch implementation of
the same block.

It is a demonstration, not a serving path: there is no KV cache and no
attention kernel of our own. What it does show is where each custom operator
sits in a real architecture, and how often each one runs.

Runs on CPU. On a GPU the same code exercises the CUDA kernels, and the
comparison becomes a genuine parity check of the composed block rather than of
individual operators.
"""

from __future__ import annotations

import argparse
import math

import torch
from torch import nn

import cudaforge


class Attention(nn.Module):
    """Multi-head self-attention, with `cudaforge.softmax` over the scores.

    Written out rather than using `scaled_dot_product_attention` so the softmax
    is a visible, replaceable step. A production path would use FlashAttention,
    which fuses the whole thing and never materialises the score matrix — see
    the roadmap.
    """

    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError(f"hidden {hidden} is not divisible by heads {heads}")
        self.heads = heads
        self.head_dim = hidden // heads
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.out = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor, use_cudaforge: bool) -> torch.Tensor:
        batch, seq, hidden = x.shape
        qkv = self.qkv(x).view(batch, seq, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)

        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Causal mask: position i may not attend to anything after it.
        causal = torch.triu(torch.full((seq, seq), float("-inf"), device=x.device), 1)
        scores = scores + causal

        flat = scores.reshape(-1, seq)
        weights = (cudaforge.softmax(flat) if use_cudaforge else torch.softmax(flat, -1)).view_as(
            scores
        )

        attended = (weights @ value).transpose(1, 2).reshape(batch, seq, hidden)
        return self.out(attended)


class FeedForward(nn.Module):
    """SwiGLU feed-forward: ``(silu(x W_gate) * (x W_up)) W_down``.

    This is the operation `cudaforge.swiglu` exists for — the elementwise part
    is three passes over memory with framework primitives and one fused pass
    here.
    """

    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden, intermediate, bias=False)
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor, use_cudaforge: bool) -> torch.Tensor:
        gate = self.gate(x)
        up = self.up(x)
        activated = (
            cudaforge.swiglu(gate, up) if use_cudaforge else torch.nn.functional.silu(gate) * up
        )
        return self.down(activated)


class Block(nn.Module):
    """One pre-norm transformer block.

    ``fused_residual_rmsnorm`` handles the pattern that appears twice here: a
    residual add immediately followed by the next sublayer's normalisation.
    """

    def __init__(self, hidden: int, heads: int, intermediate: int) -> None:
        super().__init__()
        self.attention_norm = nn.Parameter(torch.ones(hidden))
        self.ffn_norm = nn.Parameter(torch.ones(hidden))
        self.attention = Attention(hidden, heads)
        self.feed_forward = FeedForward(hidden, intermediate)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor, use_cudaforge: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq, hidden = x.shape

        if use_cudaforge:
            normed, residual = cudaforge.fused_residual_rmsnorm(
                x.reshape(-1, hidden), residual.reshape(-1, hidden), self.attention_norm
            )
            normed = normed.view(batch, seq, hidden)
            residual = residual.view(batch, seq, hidden)
        else:
            residual = x + residual
            normed = _reference_rmsnorm(residual, self.attention_norm)

        attended = self.attention(normed, use_cudaforge)

        if use_cudaforge:
            normed, residual = cudaforge.fused_residual_rmsnorm(
                attended.reshape(-1, hidden), residual.reshape(-1, hidden), self.ffn_norm
            )
            normed = normed.view(batch, seq, hidden)
            residual = residual.view(batch, seq, hidden)
        else:
            residual = attended + residual
            normed = _reference_rmsnorm(residual, self.ffn_norm)

        return self.feed_forward(normed, use_cudaforge), residual


def _reference_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    args = parser.parse_args(argv)

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(cudaforge.backend_report())
    print(f"device: {device}\n")

    intermediate = 4 * args.hidden
    blocks = nn.ModuleList(
        Block(args.hidden, args.heads, intermediate) for _ in range(args.layers)
    ).to(device)

    x = torch.randn(args.batch, args.seq, args.hidden, device=device)

    with torch.no_grad():
        forge_hidden, forge_residual = x, torch.zeros_like(x)
        for block in blocks:
            forge_hidden, forge_residual = block(forge_hidden, forge_residual, True)

        torch_hidden, torch_residual = x, torch.zeros_like(x)
        for block in blocks:
            torch_hidden, torch_residual = block(torch_hidden, torch_residual, False)

    difference = (forge_hidden - torch_hidden).abs().max().item()

    print(f"{args.layers} blocks, hidden {args.hidden}, {args.heads} heads, seq {args.seq}")
    print("\noperator usage per forward pass:")
    print(f"  fused_residual_rmsnorm   {2 * args.layers:>4}   (twice per block)")
    print(f"  softmax                  {args.layers:>4}   (once per block, over the scores)")
    print(f"  swiglu                   {args.layers:>4}   (once per block)")
    print(f"\nmax |cudaforge - pytorch| = {difference:.3e}")

    tolerance = 1e-4
    if difference > tolerance:
        print(f"\nFAIL: exceeds tolerance {tolerance:.1e}")
        return 1
    print(f"ok: within tolerance {tolerance:.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

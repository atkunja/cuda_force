#!/usr/bin/env python3
"""Data-parallel LoRA fine-tuning with DistributedDataParallel.

    torchrun --nproc_per_node=4 examples/distributed_train.py

Requires multiple NVIDIA GPUs and NCCL. It has not been executed on the
development host, which has neither.

## What DDP does

Each rank holds a full model replica and processes a distinct shard of the
batch. After the backward pass, gradients are averaged across ranks with an
AllReduce, so every replica applies an identical update and the replicas never
diverge. The effective batch is `per_rank_batch * world_size`.

## Why AllReduce and not a parameter server

AllReduce is bandwidth-optimal: the ring algorithm moves `2(N-1)/N` times the
gradient size per rank regardless of `N`, and every link carries equal traffic.
A parameter server concentrates `N` times the gradient size on one node, which
becomes the bottleneck as soon as `N` is more than a handful.

## Why the communication is nearly free

DDP registers a hook on every parameter's gradient and starts the AllReduce for
a *bucket* of gradients as soon as that bucket is complete — during the backward
pass, not after it. Communication for the later layers therefore overlaps
computation for the earlier ones. Waiting until the backward pass finished would
serialise the two and make communication a visible cost.

## Why LoRA changes the arithmetic

Only trainable parameters participate in the AllReduce. With adapters at well
under 1% of the model, the gradient traffic per step falls by the same factor —
which is what makes multi-node LoRA practical on interconnects that would choke
on full fine-tuning.
"""

from __future__ import annotations

import argparse
import logging
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from training.config import LoRAConfig, TrainingConfig
from training.dataset import build_dataset
from training.lora import attach_lora, describe_parameters
from training.train import set_seed

_LOG = logging.getLogger("cudaforge.distributed")

CORPUS = [
    "CUDA threads are grouped into warps of 32 that execute in lockstep.",
    "AllReduce averages gradients across every rank in the process group.",
    "NCCL implements collectives over NVLink and InfiniBand.",
    "Data parallelism replicates the model and shards the batch.",
    "Gradient buckets let communication overlap the backward pass.",
    "LoRA reduces gradient traffic in proportion to the trainable parameter count.",
] * 16


def setup() -> tuple[int, int, int]:
    """Initialise the process group from torchrun's environment variables."""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # NCCL is the only backend with a fast path over NVLink and InfiniBand; gloo
    # is a CPU fallback and would make the collectives the bottleneck.
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2, help="per rank")
    parser.add_argument("--rank-dim", type=int, default=8, help="LoRA rank")
    args = parser.parse_args(argv)

    if "RANK" not in os.environ:
        raise SystemExit(
            "launch with torchrun, e.g.\n"
            "  torchrun --nproc_per_node=4 examples/distributed_train.py"
        )
    if not torch.cuda.is_available():
        raise SystemExit("this example requires NVIDIA GPUs and NCCL")

    rank, local_rank, world_size = setup()
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"[rank {rank}] %(message)s",
    )

    # Every rank uses the same seed so the replicas start identical. DDP
    # broadcasts rank 0's parameters at construction as well, but matching seeds
    # keeps dropout masks and any rank-local randomness consistent too.
    set_seed(42)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(args.model)
    config = TrainingConfig(
        model_name=args.model,
        lora=LoRAConfig(rank=args.rank_dim, alpha=args.rank_dim * 2,
                        target_modules=["c_attn"]),
    )
    model = attach_lora(base, config.lora).to(local_rank)
    _LOG.info("%s", describe_parameters(model))

    # find_unused_parameters stays False: it costs an extra graph traversal per
    # step. The frozen base parameters are excluded by requires_grad, not by
    # being unused, so DDP does not need the scan.
    model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)

    dataset = build_dataset(CORPUS, tokenizer, block_size=64)
    # The sampler gives each rank a disjoint shard. Without it every rank would
    # train on the whole dataset and the effective batch would be wrong.
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=2e-4)

    for epoch in range(args.epochs):
        # Required: without it the shuffle is identical every epoch and each
        # rank sees the same shard order forever.
        sampler.set_epoch(epoch)
        model.train()

        for step, batch in enumerate(loader):
            inputs = {key: value.to(local_rank) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(**inputs).loss

            # The AllReduce is issued here, bucket by bucket, overlapping the
            # rest of the backward pass.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if rank == 0 and step % 5 == 0:
                _LOG.info("epoch %d step %d loss %.4f", epoch, step, loss.item())

    # Only rank 0 writes. Every rank holds identical parameters after the final
    # AllReduce, so writing from all of them would race on the same paths for no
    # benefit.
    if rank == 0:
        model.module.save_pretrained("checkpoints/distributed")
        _LOG.info("saved adapters to checkpoints/distributed")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

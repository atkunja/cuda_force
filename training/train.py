"""LoRA fine-tuning for causal language models.

The loop is written out rather than delegated to ``transformers.Trainer``.
Trainer is the right tool for a production run, but the mechanics it hides —
gradient accumulation, the scaler's interaction with clipping, when the
scheduler steps — are exactly the parts worth being explicit about here.

Run it:

    python -m training.train --config training/configs/tiny.yaml

The default config needs no network access and no GPU, so the pipeline is
executable and testable anywhere.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.config import TrainingConfig
from training.dataset import build_dataset, load_texts
from training.evaluation import evaluate, perplexity_from_loss
from training.lora import attach_lora, describe_parameters

_LOG = logging.getLogger("cudaforge.train")


def set_seed(seed: int) -> None:
    """Seed every RNG the training path touches.

    Missing one of these is the usual reason two runs with identical configs
    diverge. cuDNN's autotuner is left enabled: making it deterministic costs
    real throughput, and it does not affect the sampled data or initialisation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    tokens_seen: int = 0
    best_loss: float = float("inf")


def build_model(config: TrainingConfig) -> tuple[torch.nn.Module, object]:
    """Load the base model and tokenizer, then attach adapters.

    QLoRA (``load_in_4bit``) needs bitsandbytes, which is Linux/CUDA only. It is
    requested here rather than assumed, and the failure is explicit — quietly
    training in fp16 when 4-bit was asked for would produce a run that does not
    fit the memory budget it was configured for.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, object] = {}
    if config.load_in_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "load_in_4bit requires CUDA; bitsandbytes has no CPU or MPS backend"
            )
        from transformers import BitsAndBytesConfig  # noqa: PLC0415

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            # Quantises the quantisation constants themselves; roughly another
            # 0.4 bits per parameter of saving.
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **load_kwargs)

    if config.gradient_checkpointing:
        # Trades compute for memory: activations are discarded on the forward
        # pass and recomputed during the backward pass, roughly a 30% step-time
        # cost for a large reduction in activation memory.
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    return attach_lora(model, config.lora), tokenizer


def build_loader(config: TrainingConfig, tokenizer: object) -> DataLoader:
    texts = load_texts(
        config.dataset_name,
        config.dataset_config,
        config.dataset_split,
        config.text_field,
        config.inline_texts,
    )
    dataset = build_dataset(texts, tokenizer, config.max_seq_length)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)


def train(config: TrainingConfig) -> TrainState:
    set_seed(config.seed)

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    _LOG.info("device: %s", device)

    model, tokenizer = build_model(config)
    model.to(device)
    model.train()
    _LOG.info("%s", describe_parameters(model))

    loader = build_loader(config, tokenizer)

    # Only adapter parameters reach the optimiser. Passing frozen parameters
    # would allocate Adam state for them — two fp32 tensors per parameter — and
    # give away most of LoRA's memory advantage.
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters; adapters were not attached")

    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )

    steps_per_epoch = max(len(loader) // config.gradient_accumulation_steps, 1)
    total_steps = (
        config.max_steps if config.max_steps > 0 else steps_per_epoch * config.epochs
    )
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=max(total_steps, 1),
        pct_start=max(warmup_steps / max(total_steps, 1), 0.01),
        anneal_strategy="cos",
    )

    # Loss scaling is a float16 concern only. Gradients in fp16 underflow to
    # zero below ~6e-8; scaling the loss up before the backward pass moves them
    # into range and the scaler unscales before the optimiser step. bfloat16 has
    # float32's exponent range and needs none of this, and CPU/MPS have no
    # autocast path worth using here.
    use_amp = config.mixed_precision and device.type == "cuda"
    use_fp16 = use_amp and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    autocast_dtype = torch.float16 if use_fp16 else torch.bfloat16

    state = TrainState()
    started = time.monotonic()
    output_dir = Path(config.output_dir)
    config.save(output_dir)

    accumulated = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(config.epochs):
        state.epoch = epoch
        for batch in loader:
            inputs = {key: value.to(device) for key, value in batch.items()}

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
                outputs = model(**inputs)
                # Dividing by the accumulation count keeps the effective
                # gradient magnitude independent of how the batch was split.
                loss = outputs.loss / config.gradient_accumulation_steps

            scaler.scale(loss).backward()
            accumulated += 1
            state.tokens_seen += int(inputs["input_ids"].numel())

            if accumulated < config.gradient_accumulation_steps:
                continue

            # Unscale before clipping: clipping a scaled gradient would clip at
            # the wrong threshold, by exactly the scale factor.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            # The scheduler steps per optimiser step, not per micro-batch.
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            accumulated = 0
            state.step += 1

            step_loss = float(loss.item()) * config.gradient_accumulation_steps
            state.best_loss = min(state.best_loss, step_loss)

            if config.logging_steps and state.step % config.logging_steps == 0:
                _LOG.info(
                    "step %d/%d  loss %.4f  ppl %.2f  lr %.2e  %.0f tok/s",
                    state.step,
                    total_steps,
                    step_loss,
                    perplexity_from_loss(step_loss),
                    scheduler.get_last_lr()[0],
                    state.tokens_seen / max(time.monotonic() - started, 1e-9),
                )

            if config.save_steps and state.step % config.save_steps == 0:
                save_checkpoint(model, output_dir / f"step-{state.step}", state)

            if config.max_steps > 0 and state.step >= config.max_steps:
                break

        if config.max_steps > 0 and state.step >= config.max_steps:
            break

    result = evaluate(model, loader, device, max_batches=8)
    _LOG.info("final evaluation: %s", result)

    save_checkpoint(model, output_dir / "final", state, evaluation=result.loss)
    return state


def save_checkpoint(
    model: torch.nn.Module,
    directory: Path,
    state: TrainState,
    evaluation: float | None = None,
) -> None:
    """Persist adapters only.

    A LoRA checkpoint is a few megabytes because the frozen base weights are
    not written — they are already on disk wherever the base model came from.
    Saving them again would make every checkpoint the size of the full model
    for no benefit.
    """
    directory.mkdir(parents=True, exist_ok=True)
    save_pretrained = getattr(model, "save_pretrained", None)
    if save_pretrained is not None:
        save_pretrained(directory)
    else:
        torch.save(
            {name: tensor for name, tensor in model.state_dict().items() if "lora" in name},
            directory / "adapter.pt",
        )

    metadata = {
        "step": state.step,
        "epoch": state.epoch,
        "tokens_seen": state.tokens_seen,
        "best_loss": state.best_loss,
        "eval_loss": evaluation,
    }
    (directory / "state.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _LOG.info("saved checkpoint to %s", directory)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML config; defaults are used if omitted")
    parser.add_argument("--model")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        stream=sys.stdout,
    )

    config = TrainingConfig.from_yaml(args.config) if args.config else TrainingConfig()
    for field, value in (
        ("model_name", args.model),
        ("output_dir", args.output_dir),
        ("epochs", args.epochs),
        ("max_steps", args.max_steps),
        ("batch_size", args.batch_size),
        ("learning_rate", args.learning_rate),
    ):
        if value is not None:
            setattr(config, field, value)
    if args.lora_rank is not None:
        config.lora.rank = args.lora_rank
    if args.load_in_4bit:
        config.load_in_4bit = True
    config.__post_init__()

    state = train(config)
    _LOG.info("finished after %d steps over %d tokens", state.step, state.tokens_seen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

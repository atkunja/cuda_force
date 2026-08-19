"""Model runners: the thing a batch is actually executed against.

The engine is deliberately model-agnostic. It owns queueing, batching, metrics
and lifecycle; a runner owns tokenisation and generation. Separating them means
the concurrency machinery can be tested exhaustively against a deterministic
runner with no model weights, no downloads and no GPU — which is most of what
there is to get wrong.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from cudaforge.config import EngineConfig, GenerationConfig


@dataclass
class GenerationResult:
    """One row of a batch's output."""

    text: str
    prompt_tokens: int
    generated_tokens: int


@runtime_checkable
class ModelRunner(Protocol):
    """Executes a batch of prompts.

    Implementations must accept prompts of differing lengths and generation
    settings, and must return results in the same order they were given.
    """

    def warmup(self, iterations: int) -> None:
        """Run throwaway work before serving traffic.

        Skipping this makes the first few requests dramatically slower than
        steady state, for reasons that have nothing to do with the code being
        measured: CUDA context creation, kernel autotuning, cuDNN algorithm
        selection and lazy module loading all happen on the first call. A
        benchmark that includes them is measuring initialisation.
        """

    def generate(
        self, prompts: list[str], settings: list[GenerationConfig]
    ) -> list[GenerationResult]: ...

    @property
    def description(self) -> str: ...


class EchoRunner:
    """Deterministic runner with no model dependency.

    Exists so the engine, batcher, metrics and server can be tested end to end
    without downloading weights or requiring a GPU. Output is derived from a
    hash of the prompt, so it is stable across runs and across machines — which
    is what makes engine-level tests assertable rather than merely
    smoke-checked.

    ``per_token_seconds`` simulates generation cost. It is what makes batching
    observable in a test: with zero cost, every batch completes instantly and
    the size/latency tradeoff has nothing to act on.
    """

    def __init__(self, per_token_seconds: float = 0.0, fixed_overhead: float = 0.0) -> None:
        self._per_token = per_token_seconds
        self._overhead = fixed_overhead

    def warmup(self, iterations: int) -> None:
        for _ in range(iterations):
            self.generate(["warmup"], [GenerationConfig(max_new_tokens=1)])

    def generate(
        self, prompts: list[str], settings: list[GenerationConfig]
    ) -> list[GenerationResult]:
        tokens = max((setting.max_new_tokens for setting in settings), default=1)

        # One fixed cost per batch plus a per-token cost, mirroring the shape of
        # real inference: the fixed part is what batching amortises.
        if self._overhead:
            time.sleep(self._overhead)
        if self._per_token:
            time.sleep(self._per_token * tokens)

        results = []
        for prompt, setting in zip(prompts, settings, strict=True):
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            body = digest[: max(setting.max_new_tokens, 1)]
            results.append(
                GenerationResult(
                    text=f"{prompt} -> {body}",
                    prompt_tokens=len(prompt.split()),
                    generated_tokens=setting.max_new_tokens,
                )
            )
        return results

    @property
    def description(self) -> str:
        return f"EchoRunner(per_token={self._per_token}s, overhead={self._overhead}s)"


class TransformersRunner:
    """Runs a causal language model from the transformers library.

    Batched generation needs left padding. A causal model continues from the
    final position of each row, so right padding would place pad tokens there
    and the model would be asked to continue from padding — producing garbage
    for every row shorter than the longest. Left padding keeps the real final
    token last for every row.
    """

    def __init__(self, config: EngineConfig) -> None:
        from transformers import (  # imported lazily: optional dependency
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        self._config = config
        self._device = config.resolve_device()
        self._dtype = config.resolve_dtype()

        self._tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self._tokenizer.pad_token_id is None:
            # Many causal LMs ship without a pad token. Reusing EOS is the
            # standard remedy; the attention mask keeps it from being attended
            # to, so it does not affect the output.
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        self._model = AutoModelForCausalLM.from_pretrained(config.model_name, dtype=self._dtype)
        self._model.to(self._device)
        self._model.eval()

    def warmup(self, iterations: int) -> None:
        for _ in range(max(iterations, 0)):
            self.generate(["warmup"], [GenerationConfig(max_new_tokens=1)])

    @torch.inference_mode()
    def generate(
        self, prompts: list[str], settings: list[GenerationConfig]
    ) -> list[GenerationResult]:
        encoded = self._tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(
            self._device
        )

        # The batch runs for as long as its most demanding member. Rows that
        # would have finished earlier are truncated at their own limit below
        # rather than shortening the batch.
        max_new = max(setting.max_new_tokens for setting in settings)
        first = settings[0]

        output = self._model.generate(
            **encoded,
            max_new_tokens=max_new,
            do_sample=not first.greedy,
            temperature=first.temperature if not first.greedy else None,
            top_p=first.top_p,
            top_k=first.top_k or None,
            pad_token_id=self._tokenizer.pad_token_id,
        )

        prompt_length = encoded["input_ids"].shape[1]
        results = []
        for row, setting in zip(range(output.shape[0]), settings, strict=True):
            generated = output[row, prompt_length : prompt_length + setting.max_new_tokens]
            text = self._tokenizer.decode(generated, skip_special_tokens=True)
            results.append(
                GenerationResult(
                    text=text,
                    prompt_tokens=int(encoded["attention_mask"][row].sum().item()),
                    generated_tokens=int(generated.numel()),
                )
            )
        return results

    @property
    def description(self) -> str:
        return (
            f"TransformersRunner(model={self._config.model_name}, "
            f"device={self._device}, dtype={self._dtype})"
        )

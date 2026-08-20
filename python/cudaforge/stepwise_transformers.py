"""A step-wise runner over a real causal language model.

`EchoStepwiseRunner` proves the scheduler; this makes it mean something. The
difficulty is not generating tokens — it is that continuous batching changes the
*batch composition* between steps, and the KV cache is one batch-wide tensor.

Admitting a sequence means adding a row to that tensor. Evicting one means
removing a row. Neither is something `model.generate` exposes, which is why this
drives the model a token at a time and owns the cache itself.

## Ragged lengths, and why the cache is left-padded

Rows in a batch have different amounts of history. A sequence admitted at step
100 has a 20-token prompt while its neighbours hold 120 tokens each. The cache
tensor has one sequence dimension, so the short row is **left-padded** and
masked out:

    row 0   [k k k k k k k k k k k k]   120 real
    row 1   [0 0 0 0 0 0 0 0 k k k k]   20 real, 100 masked

Left, not right: a causal model attends to everything before the current
position, so padding must sit *before* the real tokens or the model would attend
to real history through a gap. It also means every row's newest token is at the
same index, which is what makes a single decode step well-defined.

The masked positions are wasted cache. That waste is exactly what paged
attention removes, and it is why `KVCacheManager` exists — but consuming it
requires an attention kernel that reads through a block table, which this does
not have. This is the honest interim: correct, and wasteful in a way a paged
implementation would not be.

## Position ids must be explicit

With left padding, position 0 is not index 0. Positions are derived from the
mask, so a padded row is told where it actually is. Leaving them implicit makes
the model believe the padding is real history, and the output degrades in a way
that looks like a bad model rather than a bad harness.
"""

from __future__ import annotations

from typing import Any

import torch

from cudaforge.config import EngineConfig, GenerationConfig
from cudaforge.stepwise import SequenceState


def _sample(logits: torch.Tensor, settings: list[GenerationConfig]) -> torch.Tensor:
    """Pick one token per row, honouring each row's own sampling settings.

    Vectorised over the batch because rows legitimately disagree: continuous
    batching mixes requests, and looping per row would put a Python iteration on
    the per-token path.
    """
    device = logits.device
    temperature = torch.tensor(
        [max(setting.temperature, 0.0) for setting in settings], device=device
    ).unsqueeze(1)

    # Temperature 0 means argmax. Dividing by it would be a division by zero, so
    # those rows are computed greedily and merged back in at the end.
    greedy_rows = temperature.squeeze(1) == 0.0
    safe = torch.where(temperature == 0.0, torch.ones_like(temperature), temperature)
    scaled = logits / safe

    top_k = torch.tensor([setting.top_k for setting in settings], device=device)
    if bool((top_k > 0).any()):
        # Per row: keep the k highest logits, mask the rest. `kth` is the
        # threshold for that row; rows with top_k == 0 keep everything.
        limit = int(top_k.max().item())
        kth = scaled.topk(limit, dim=-1).values
        index = (top_k.clamp(min=1) - 1).clamp(max=limit - 1)
        threshold = kth.gather(1, index.unsqueeze(1))
        scaled = torch.where(
            (scaled < threshold) & (top_k > 0).unsqueeze(1),
            torch.full_like(scaled, float("-inf")),
            scaled,
        )

    top_p = torch.tensor([setting.top_p for setting in settings], device=device).unsqueeze(1)
    if bool((top_p < 1.0).any()):
        ordered, order = scaled.sort(dim=-1, descending=True)
        cumulative = ordered.softmax(dim=-1).cumsum(dim=-1)
        # Drop the tail beyond the mass threshold, but never the top token:
        # shifting keeps at least one candidate even when it alone exceeds top_p.
        drop = cumulative - ordered.softmax(dim=-1) > top_p
        drop[:, 0] = False
        ordered = ordered.masked_fill(drop, float("-inf"))
        scaled = ordered.scatter(1, order, ordered)

    sampled = torch.multinomial(scaled.softmax(dim=-1), num_samples=1).squeeze(1)
    return torch.where(greedy_rows, logits.argmax(dim=-1), sampled)


class TransformersStepwiseRunner:
    """Drives a causal LM one token at a time, owning the KV cache.

    `model` and `tokenizer` may be supplied so the runner can be tested without
    a download — the batch-composition logic is what is worth testing and it is
    model-independent.
    """

    # `Any` rather than a transformers type, matching TransformersRunner: these
    # are injection points for stubs, not a claim about the library's classes.
    _model: Any
    _tokenizer: Any

    def __init__(
        self,
        config: EngineConfig,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self._config = config
        self._device = config.resolve_device()
        self._dtype = config.resolve_dtype()

        if model is not None and tokenizer is not None:
            self._model, self._tokenizer = model, tokenizer
        else:
            from transformers import (  # imported lazily: optional dependency
                AutoModelForCausalLM,
                AutoTokenizer,
            )

            self._tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            loaded = AutoModelForCausalLM.from_pretrained(config.model_name, dtype=self._dtype)
            self._model = loaded.to(self._device)  # type: ignore[arg-type]

        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"
        self._model.eval()

        #: Sequence ids in cache-row order. The single source of truth for which
        #: row belongs to which sequence; every cache operation reindexes it.
        #: Call counts, not timings: continuous batching trades a few wide
        #: prefills for many narrow ones, and that trade is invisible in a decode
        #: step count. Counters are cheap enough to leave on.
        self.prefills = 0
        self.steps = 0

        self._rows: list[int] = []
        self._cache: Any | None = None
        self._mask: torch.Tensor | None = None
        self._last: torch.Tensor | None = None

    # -- cache surgery ------------------------------------------------------

    @staticmethod
    def _left_pad(tensor: torch.Tensor, to_length: int, dim: int) -> torch.Tensor:
        """Pad the sequence dimension on the left, so newest tokens stay aligned."""
        deficit = to_length - tensor.shape[dim]
        if deficit <= 0:
            return tensor
        shape = list(tensor.shape)
        shape[dim] = deficit
        pad = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([pad, tensor], dim=dim)

    def _merge_cache(self, incoming: Any, incoming_mask: torch.Tensor) -> None:
        """Concatenate new rows onto the running cache, aligning their lengths."""
        from transformers import DynamicCache  # imported lazily

        if self._cache is None or not self._rows:
            self._cache, self._mask = incoming, incoming_mask
            return

        assert self._mask is not None, "a populated cache always carries its mask"
        existing_length = self._cache.layers[0].keys.shape[2]
        incoming_length = incoming.layers[0].keys.shape[2]
        common = max(existing_length, incoming_length)

        merged = DynamicCache()
        for index in range(len(self._cache.layers)):
            keys = torch.cat(
                [
                    self._left_pad(self._cache.layers[index].keys, common, 2),
                    self._left_pad(incoming.layers[index].keys, common, 2),
                ],
                dim=0,
            )
            values = torch.cat(
                [
                    self._left_pad(self._cache.layers[index].values, common, 2),
                    self._left_pad(incoming.layers[index].values, common, 2),
                ],
                dim=0,
            )
            merged.update(keys, values, index)

        self._cache = merged
        self._mask = torch.cat(
            [
                self._left_pad(self._mask, common, 1),
                self._left_pad(incoming_mask, common, 1),
            ],
            dim=0,
        )

    def _keep_rows(self, keep: list[int]) -> None:
        """Reduce every batch-indexed structure to `keep`, in that order."""
        if not keep:
            self._cache, self._mask, self._last, self._rows = None, None, None, []
            return

        index = torch.tensor(keep, device=self._device)
        if self._cache is not None:
            self._cache.batch_select_indices(index)
        if self._mask is not None:
            self._mask = self._mask.index_select(0, index)
        if self._last is not None:
            self._last = self._last.index_select(0, index)
        self._rows = [self._rows[position] for position in keep]

    # -- the protocol -------------------------------------------------------

    @torch.inference_mode()
    def prefill(self, states: list[SequenceState]) -> None:
        if not states:
            return
        self.prefills += 1

        encoded = self._tokenizer(
            [state.prompt for state in states],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self._device)

        mask = encoded["attention_mask"]
        output = self._model(
            input_ids=encoded["input_ids"],
            attention_mask=mask,
            position_ids=self._positions(mask),
            use_cache=True,
        )

        self._merge_cache(output.past_key_values, mask)
        self._rows.extend(state.sequence_id for state in states)

        # The first generated token comes from the prompt's final position.
        first = _sample(output.logits[:, -1, :], [state.generation for state in states])
        self._last = (
            first.unsqueeze(1)
            if self._last is None
            else torch.cat([self._last, first.unsqueeze(1)], dim=0)
        )
        self._record(states, first, prefilled=True)

    @staticmethod
    def _positions(mask: torch.Tensor) -> torch.Tensor:
        """Positions implied by a left-padded mask.

        With left padding, index 0 is not position 0. Leaving positions implicit
        tells the model its padding is real history, and the output degrades in a
        way that looks like a bad model rather than a bad harness.
        """
        return (mask.cumsum(dim=-1) - 1).clamp(min=0)

    @torch.inference_mode()
    def decode_step(self, states: list[SequenceState]) -> None:
        active = [state for state in states if not state.finished]
        if not active or self._cache is None or self._last is None:
            return
        self.steps += 1

        # The scheduler's active set and the cache's rows must agree; it may have
        # retired a sequence since the last step.
        wanted = {state.sequence_id for state in active}
        keep = [position for position, row in enumerate(self._rows) if row in wanted]
        if len(keep) != len(self._rows):
            self._keep_rows(keep)
        if not self._rows:
            return

        assert self._mask is not None
        self._mask = torch.cat([self._mask, torch.ones_like(self._mask[:, :1])], dim=1)

        output = self._model(
            input_ids=self._last,
            attention_mask=self._mask,
            position_ids=self._positions(self._mask)[:, -1:],
            past_key_values=self._cache,
            use_cache=True,
        )
        self._cache = output.past_key_values

        by_id = {state.sequence_id: state for state in active}
        ordered = [by_id[row] for row in self._rows]
        tokens = _sample(output.logits[:, -1, :], [state.generation for state in ordered])
        self._last = tokens.unsqueeze(1)
        self._record(ordered, tokens, prefilled=False)

    def _record(self, states: list[SequenceState], tokens: torch.Tensor, prefilled: bool) -> None:
        eos = self._tokenizer.eos_token_id
        for state, token in zip(states, tokens.tolist(), strict=True):
            if eos is not None and token == eos:
                # End-of-sequence finishes the sequence without contributing a
                # token, which is distinct from exhausting the budget and is
                # exactly the row continuous batching reclaims early.
                state.stopped_early = True
                continue
            if state.finished and not prefilled:
                continue
            state.tokens.append(self._tokenizer.decode([token], skip_special_tokens=True))

    def evict(self, sequence_id: int) -> None:
        if sequence_id not in self._rows:
            return
        keep = [position for position, row in enumerate(self._rows) if row != sequence_id]
        self._keep_rows(keep)

    @property
    def active_rows(self) -> int:
        """Rows currently in the cache. Should track the scheduler's live set."""
        return len(self._rows)

    @property
    def cache_length(self) -> int:
        """Sequence length of the cache tensor, padding included."""
        if self._cache is None or not self._cache.layers:
            return 0
        return int(self._cache.layers[0].keys.shape[2])

    @property
    def description(self) -> str:
        return (
            f"TransformersStepwiseRunner(model={self._config.model_name}, "
            f"device={self._device}, dtype={self._dtype})"
        )

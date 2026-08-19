"""Dataset preparation for causal language modelling.

The packing strategy here is the standard one for continued pretraining and
instruction tuning: concatenate every example with an EOS separator, then cut
the stream into fixed-length blocks. This wastes no tokens on padding, which
matters because padding is compute the GPU performs and then discards.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

_LOG = logging.getLogger(__name__)


@dataclass
class PackedExample:
    """One training block.

    ``labels`` is a copy of ``input_ids`` because causal LM training predicts
    the next token from the sequence itself. The shift between input and target
    is applied inside the model's loss, not here — doing it here as well would
    shift twice and train the model to skip a token.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class PackedCausalDataset(Dataset[dict[str, torch.Tensor]]):
    """Fixed-length blocks cut from a concatenated token stream."""

    def __init__(self, blocks: Sequence[torch.Tensor]) -> None:
        if not blocks:
            raise ValueError(
                "no training blocks were produced; the corpus is shorter than "
                "one block, or max_seq_length is too large for it"
            )
        self._blocks = list(blocks)

    def __len__(self) -> int:
        return len(self._blocks)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        block = self._blocks[index]
        return {
            "input_ids": block,
            "attention_mask": torch.ones_like(block),
            # Cloned rather than aliased: the trainer may modify labels in place
            # to mask positions, and sharing storage would corrupt the inputs.
            "labels": block.clone(),
        }


def pack_token_stream(token_ids: Sequence[int], block_size: int) -> list[torch.Tensor]:
    """Cut a flat token stream into blocks of exactly ``block_size``.

    The trailing remainder is dropped. Keeping it would require padding, and a
    single short block contributes little while complicating every downstream
    shape assumption.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    usable = (len(token_ids) // block_size) * block_size
    if usable == 0:
        return []

    stream = torch.tensor(token_ids[:usable], dtype=torch.long)
    return list(stream.view(-1, block_size))


def build_dataset(
    texts: Iterable[str],
    tokenizer: object,
    block_size: int,
    eos_token_id: int | None = None,
) -> PackedCausalDataset:
    """Tokenise, concatenate with EOS separators, and pack.

    The EOS between documents is what teaches the model where a document ends.
    Without it, packing silently trains the model to continue from one document
    straight into an unrelated one.
    """
    encode = getattr(tokenizer, "encode", None)
    if encode is None:
        raise TypeError("tokenizer must expose an encode(text) method")

    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)

    stream: list[int] = []
    documents = 0
    for text in texts:
        if not text or not text.strip():
            continue
        stream.extend(encode(text))
        if eos_token_id is not None:
            stream.append(eos_token_id)
        documents += 1

    blocks = pack_token_stream(stream, block_size)
    _LOG.info(
        "packed %d documents (%d tokens) into %d blocks of %d",
        documents,
        len(stream),
        len(blocks),
        block_size,
    )
    return PackedCausalDataset(blocks)


def load_texts(
    dataset_name: str | None,
    dataset_config: str | None,
    split: str,
    text_field: str,
    inline_texts: Sequence[str],
) -> list[str]:
    """Return the corpus, preferring a named dataset and falling back inline.

    The inline path exists so the pipeline runs with no network access. A
    training script that cannot be executed without downloading a dataset
    cannot be tested, and an untested training script is where silent
    correctness bugs live.
    """
    if dataset_name is None:
        if not inline_texts:
            raise ValueError("set either dataset_name or inline_texts")
        return list(inline_texts)

    from datasets import load_dataset  # imported lazily: optional dependency

    dataset = load_dataset(dataset_name, dataset_config, split=split)
    if text_field not in dataset.column_names:
        raise KeyError(
            f"column {text_field!r} not in {dataset.column_names}; "
            f"set text_field to one of these"
        )
    return [row for row in dataset[text_field] if row]

from __future__ import annotations

import pytest
import torch

from training.dataset import PackedCausalDataset, build_dataset, load_texts, pack_token_stream


class StubTokenizer:
    """Deterministic whitespace tokenizer.

    A real tokenizer would make these assertions depend on a downloaded vocab
    file. The packing logic is what is under test, not the tokenisation.
    """

    eos_token_id = 999

    def encode(self, text: str) -> list[int]:
        return [len(word) for word in text.split()]


def test_packing_produces_exact_blocks():
    blocks = pack_token_stream(list(range(100)), block_size=32)
    assert len(blocks) == 3  # 96 of 100 tokens used
    assert all(block.numel() == 32 for block in blocks)


def test_the_remainder_is_dropped():
    # Keeping it would require padding, which is compute the GPU performs and
    # then discards.
    blocks = pack_token_stream(list(range(70)), block_size=32)
    assert len(blocks) == 2
    assert torch.equal(blocks[-1], torch.arange(32, 64))


def test_a_stream_shorter_than_one_block_yields_nothing():
    assert pack_token_stream(list(range(10)), block_size=32) == []


def test_a_non_positive_block_size_is_rejected():
    with pytest.raises(ValueError, match="block_size"):
        pack_token_stream([1, 2, 3], block_size=0)


def test_an_empty_dataset_is_rejected_with_an_actionable_message():
    with pytest.raises(ValueError, match="max_seq_length"):
        PackedCausalDataset([])


def test_items_carry_input_mask_and_labels():
    dataset = PackedCausalDataset([torch.arange(8)])
    item = dataset[0]
    assert set(item) == {"input_ids", "attention_mask", "labels"}
    assert torch.equal(item["labels"], item["input_ids"])
    assert item["attention_mask"].sum() == 8


def test_labels_do_not_alias_the_inputs():
    # The trainer may mask label positions in place; sharing storage would
    # corrupt the inputs and train on the mask.
    dataset = PackedCausalDataset([torch.arange(8)])
    item = dataset[0]
    item["labels"][0] = -100
    assert item["input_ids"][0] == 0


def test_documents_are_separated_by_eos():
    tokenizer = StubTokenizer()
    dataset = build_dataset(["aa bb", "cc dd"], tokenizer, block_size=3)
    # Stream is [2, 2, 999, 2, 2, 999]; the first block ends with the separator
    # that marks the end of document one.
    assert torch.equal(dataset[0]["input_ids"], torch.tensor([2, 2, 999]))


def test_blank_documents_are_skipped():
    tokenizer = StubTokenizer()
    dataset = build_dataset(["aa bb", "", "   ", "cc dd"], tokenizer, block_size=3)
    assert len(dataset) == 2


def test_build_dataset_rejects_a_tokenizer_without_encode():
    with pytest.raises(TypeError, match="encode"):
        build_dataset(["text"], object(), block_size=4)


def test_inline_texts_are_used_when_no_dataset_is_named():
    texts = load_texts(None, None, "train", "text", ["a", "b"])
    assert texts == ["a", "b"]


def test_missing_corpus_is_rejected():
    with pytest.raises(ValueError, match="inline_texts"):
        load_texts(None, None, "train", "text", [])

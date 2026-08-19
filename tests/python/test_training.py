"""Training-pipeline tests.

The full loop is exercised against a stub model rather than a downloaded one:
the mechanics under test — accumulation, seeding, checkpointing, the packing
contract — are model-independent, and a test that downloads weights is a test
that fails offline.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from training.config import LoRAConfig, TrainingConfig
from training.dataset import PackedCausalDataset
from training.evaluation import evaluate, perplexity_from_loss
from training.train import TrainState, save_checkpoint, set_seed


class StubCausalLM(nn.Module):
    """Minimal causal LM with the interface the training loop expects."""

    def __init__(self, vocab: int = 64, hidden: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, input_ids, attention_mask=None, labels=None):
        logits = self.head(self.embed(input_ids))
        loss = None
        if labels is not None:
            # Shifted by one: position i predicts i+1. The final position has no
            # target, which is why the slices differ at both ends.
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1)
            )
        return type("Output", (), {"logits": logits, "loss": loss})()


@pytest.fixture
def loader() -> DataLoader:
    blocks = [torch.randint(0, 64, (16,)) for _ in range(8)]
    return DataLoader(PackedCausalDataset(blocks), batch_size=2)


def test_training_config_defaults_are_valid():
    config = TrainingConfig()
    assert config.effective_batch_size == config.batch_size


def test_effective_batch_size_multiplies_accumulation():
    # This is the number that matters when choosing a learning rate.
    config = TrainingConfig(batch_size=4, gradient_accumulation_steps=8)
    assert config.effective_batch_size == 32


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch_size": 0}, "batch_size"),
        ({"gradient_accumulation_steps": 0}, "gradient_accumulation_steps"),
        ({"max_seq_length": 0}, "max_seq_length"),
        ({"learning_rate": 0}, "learning_rate"),
        ({"epochs": 0, "max_steps": 0}, "epochs or max_steps"),
        ({"warmup_ratio": 1.0}, "warmup_ratio"),
    ],
)
def test_invalid_training_config_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TrainingConfig(**kwargs)


def test_config_round_trips_through_a_dict():
    config = TrainingConfig.from_dict(
        {"model_name": "x", "batch_size": 2, "lora": {"rank": 16, "alpha": 32}}
    )
    assert config.lora.rank == 16
    assert config.lora.scaling == pytest.approx(2.0)


def test_config_is_written_beside_the_checkpoint(tmp_path):
    # A run must be reproducible from its output directory alone.
    config = TrainingConfig(output_dir=str(tmp_path))
    path = config.save(tmp_path)
    payload = json.loads(path.read_text())
    assert payload["model_name"] == config.model_name
    assert payload["lora"]["rank"] == config.lora.rank


def test_seeding_makes_initialisation_reproducible():
    set_seed(1234)
    first = torch.randn(16)
    set_seed(1234)
    torch.testing.assert_close(first, torch.randn(16))


def test_evaluation_reports_loss_and_perplexity(loader):
    result = evaluate(StubCausalLM(), loader, torch.device("cpu"))
    assert result.batches == 4
    assert result.tokens > 0
    assert result.perplexity == pytest.approx(perplexity_from_loss(result.loss), rel=1e-6)
    assert "perplexity" in str(result)


def test_evaluation_restores_training_mode(loader):
    model = StubCausalLM()
    model.train()
    evaluate(model, loader, torch.device("cpu"))
    assert model.training


def test_evaluation_respects_a_batch_limit(loader):
    assert evaluate(StubCausalLM(), loader, torch.device("cpu"), max_batches=2).batches == 2


def test_perplexity_saturates_instead_of_overflowing():
    # A diverged run should report inf rather than crash the eval loop at the
    # moment something has already gone wrong.
    assert perplexity_from_loss(1000.0) == float("inf")
    assert perplexity_from_loss(0.0) == pytest.approx(1.0)


def test_checkpoint_saves_only_adapter_tensors(tmp_path):
    # A LoRA checkpoint is small because the frozen base weights are not
    # written; they already exist wherever the base model came from.
    model = nn.Module()
    model.register_parameter("lora_a", nn.Parameter(torch.randn(4, 8)))
    model.register_parameter("base_weight", nn.Parameter(torch.randn(64, 64)))

    save_checkpoint(model, tmp_path / "ckpt", TrainState(step=5, tokens_seen=100))

    saved = torch.load(tmp_path / "ckpt" / "adapter.pt", weights_only=True)
    assert set(saved) == {"lora_a"}

    state = json.loads((tmp_path / "ckpt" / "state.json").read_text())
    assert state["step"] == 5
    assert state["tokens_seen"] == 100


def test_gradient_accumulation_matches_a_single_large_step():
    # Accumulating k micro-batches with the loss divided by k must produce the
    # same gradient as one batch of k times the size. If it did not, the
    # effective learning rate would depend on how the batch was split.
    torch.manual_seed(0)
    data = torch.randint(0, 64, (8, 16))

    set_seed(0)
    whole = StubCausalLM()
    whole(input_ids=data, labels=data).loss.backward()
    reference = whole.head.weight.grad.clone()

    set_seed(0)
    split = StubCausalLM()
    for chunk in data.chunk(4):
        (split(input_ids=chunk, labels=chunk).loss / 4).backward()

    torch.testing.assert_close(split.head.weight.grad, reference, rtol=1e-4, atol=1e-5)


def test_lora_target_modules_default_to_attention_projections():
    assert "c_attn" in LoRAConfig().target_modules

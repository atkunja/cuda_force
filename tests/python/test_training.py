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
from training.train import TrainState, save_checkpoint, set_seed, train


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


# --- the training loop itself ----------------------------------------------


@pytest.fixture
def stub_model() -> StubCausalLM:
    """A stub with a frozen "base" and a trainable "adapter".

    Mirrors LoRA's parameter layout so the loop's assumption — that only
    trainable parameters reach the optimiser — is actually exercised.
    """
    torch.manual_seed(0)
    model = StubCausalLM()
    model.embed.weight.requires_grad_(False)
    return model


def test_the_loop_runs_and_advances_state(tmp_path, stub_model, loader):
    config = TrainingConfig(
        output_dir=str(tmp_path),
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        mixed_precision=False,
        logging_steps=0,
    )
    state = train(config, model=stub_model, loader=loader)

    assert state.step > 0
    assert state.tokens_seen > 0
    assert state.best_loss < float("inf")


def test_the_loop_only_updates_trainable_parameters(tmp_path, stub_model, loader):
    frozen_before = stub_model.embed.weight.detach().clone()
    trainable_before = stub_model.head.weight.detach().clone()

    config = TrainingConfig(
        output_dir=str(tmp_path),
        epochs=1,
        learning_rate=1e-2,
        mixed_precision=False,
        logging_steps=0,
    )
    train(config, model=stub_model, loader=loader)

    # Compared on CPU: train() moves the model to whatever device is available,
    # which on this host is MPS.
    frozen_after = stub_model.embed.weight.detach().cpu()
    trainable_after = stub_model.head.weight.detach().cpu()

    # Passing frozen parameters to the optimiser would allocate Adam state for
    # them and give away most of LoRA's memory advantage — and would move them.
    torch.testing.assert_close(frozen_after, frozen_before)
    assert not torch.allclose(trainable_after, trainable_before)


def test_a_model_with_no_trainable_parameters_is_rejected(tmp_path, loader):
    # Silently training a model with no adapters attached produces a run that
    # looks successful and updates nothing.
    model = StubCausalLM()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    config = TrainingConfig(output_dir=str(tmp_path), epochs=1, mixed_precision=False)
    with pytest.raises(RuntimeError, match="no trainable parameters"):
        train(config, model=model, loader=loader)


def test_max_steps_caps_the_run(tmp_path, stub_model, loader):
    config = TrainingConfig(
        output_dir=str(tmp_path),
        epochs=10,
        max_steps=3,
        mixed_precision=False,
        logging_steps=0,
    )
    state = train(config, model=stub_model, loader=loader)
    assert state.step == 3


def test_accumulation_reduces_the_optimiser_step_count(tmp_path, loader):
    def run(accumulation: int) -> int:
        torch.manual_seed(0)
        model = StubCausalLM()
        config = TrainingConfig(
            output_dir=str(tmp_path),
            epochs=1,
            gradient_accumulation_steps=accumulation,
            mixed_precision=False,
            logging_steps=0,
        )
        return train(config, model=model, loader=loader).step

    # Four micro-batches per optimiser step means a quarter of the steps.
    assert run(1) == 4 * run(4)


def test_the_run_writes_a_checkpoint_and_its_config(tmp_path, stub_model, loader):
    config = TrainingConfig(
        output_dir=str(tmp_path), epochs=1, mixed_precision=False, logging_steps=0
    )
    train(config, model=stub_model, loader=loader)

    assert (tmp_path / "training_config.json").is_file()
    assert (tmp_path / "final" / "state.json").is_file()

    state = json.loads((tmp_path / "final" / "state.json").read_text())
    assert state["step"] > 0
    assert state["eval_loss"] is not None


def test_periodic_checkpoints_are_written(tmp_path, stub_model, loader):
    config = TrainingConfig(
        output_dir=str(tmp_path),
        epochs=1,
        save_steps=2,
        mixed_precision=False,
        logging_steps=0,
    )
    train(config, model=stub_model, loader=loader)
    assert (tmp_path / "step-2").is_dir()


def test_supplying_a_model_without_a_loader_is_rejected(tmp_path, stub_model):
    config = TrainingConfig(output_dir=str(tmp_path), epochs=1, mixed_precision=False)
    with pytest.raises(ValueError, match="supply a loader"):
        train(config, model=stub_model)


def test_two_seeded_runs_produce_the_same_result(tmp_path, loader):
    def run() -> float:
        torch.manual_seed(0)
        model = StubCausalLM()
        config = TrainingConfig(
            output_dir=str(tmp_path),
            epochs=1,
            seed=1234,
            mixed_precision=False,
            logging_steps=0,
        )
        return train(config, model=model, loader=loader).best_loss

    # Missing a seed is the usual reason two runs with identical configs
    # diverge, and it is invisible until someone tries to reproduce a result.
    assert run() == pytest.approx(run())


# --- the command line ------------------------------------------------------


def test_main_applies_flag_overrides_on_top_of_a_config(tmp_path, monkeypatch):
    import yaml

    from training import train as train_module

    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        yaml.safe_dump({"model_name": "from-file", "batch_size": 2, "epochs": 5}),
        encoding="utf-8",
    )

    captured: dict[str, TrainingConfig] = {}
    monkeypatch.setattr(
        train_module, "train", lambda config: captured.setdefault("config", config) or TrainState()
    )

    assert (
        train_module.main(
            ["--config", str(config_path), "--epochs", "2", "--lora-rank", "16"]
        )
        == 0
    )

    config = captured["config"]
    assert config.model_name == "from-file"  # untouched by flags
    assert config.epochs == 2  # overridden
    assert config.lora.rank == 16


def test_main_validates_the_resulting_config(tmp_path, monkeypatch):
    # An override that produces an unworkable config must fail at startup, not
    # partway through a run.
    from training import train as train_module

    monkeypatch.setattr(train_module, "train", lambda config: TrainState())
    with pytest.raises(ValueError, match="batch_size"):
        train_module.main(["--batch-size", "0"])


def test_main_defaults_need_no_config_file(monkeypatch):
    from training import train as train_module

    captured: dict[str, TrainingConfig] = {}
    monkeypatch.setattr(
        train_module, "train", lambda config: captured.setdefault("config", config) or TrainState()
    )

    assert train_module.main(["--max-steps", "1"]) == 0
    assert captured["config"].max_steps == 1


def test_the_4bit_flag_is_recorded(monkeypatch):
    from training import train as train_module

    captured: dict[str, TrainingConfig] = {}
    monkeypatch.setattr(
        train_module, "train", lambda config: captured.setdefault("config", config) or TrainState()
    )

    train_module.main(["--load-in-4bit", "--max-steps", "1"])
    assert captured["config"].load_in_4bit

"""Training configuration.

Every field that changes results is here rather than scattered through the
training script, so a run is reproducible from its config file alone. The
config is written into each checkpoint directory for the same reason.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: The values PEFT accepts for `LoraConfig.bias`. Mirrored here so an invalid
#: setting fails when the config is built rather than when the model is.
LORA_BIAS_CHOICES = frozenset({"none", "all", "lora_only"})


@dataclass
class LoRAConfig:
    """Low-rank adapter settings.

    LoRA freezes the base weights and learns ``W + (alpha/r) * B A`` where A is
    ``[in, r]`` and B is ``[r, out]``. With r far below the layer width, the
    trainable parameter count drops by two to three orders of magnitude, and —
    more importantly for memory — so does optimiser state, which for Adam is
    twice the trainable parameter count in fp32.

    ``alpha`` is a scaling numerator, not a second learning rate. The effective
    adapter scale is ``alpha / r``, so raising ``r`` without raising ``alpha``
    quietly *shrinks* the adapter's influence. Keeping ``alpha = 2 * r`` is a
    common convention that holds the scale fixed as rank changes.
    """

    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    # Attention projections are where LoRA gives the most benefit per parameter.
    # Adding the MLP projections helps on harder tasks at roughly triple the
    # adapter size.
    target_modules: list[str] = field(default_factory=lambda: ["c_attn", "c_proj"])
    #: Which bias terms PEFT should train alongside the adapters. Validated
    #: here rather than left to PEFT, which reports an invalid value from deep
    #: inside model construction, long after the config was read.
    bias: str = "none"

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {self.rank}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.bias not in LORA_BIAS_CHOICES:
            raise ValueError(f"bias must be one of {sorted(LORA_BIAS_CHOICES)}, got {self.bias!r}")
        if not self.target_modules:
            raise ValueError("target_modules must not be empty")

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank


@dataclass
class TrainingConfig:
    """A complete fine-tuning run."""

    model_name: str = "sshleifer/tiny-gpt2"
    output_dir: str = "checkpoints/run"

    dataset_name: str | None = None
    dataset_config: str | None = None
    dataset_split: str = "train"
    text_field: str = "text"
    # Used when dataset_name is None. Keeps `python -m training.train` runnable
    # with no network access, which is what makes the pipeline testable.
    inline_texts: list[str] = field(default_factory=list)

    max_seq_length: int = 256
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    epochs: int = 1
    max_steps: int = -1
    seed: int = 42

    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    load_in_4bit: bool = False

    logging_steps: int = 10
    eval_steps: int = 0
    save_steps: int = 0

    lora: LoRAConfig = field(default_factory=LoRAConfig)

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                f"gradient_accumulation_steps must be positive, "
                f"got {self.gradient_accumulation_steps}"
            )
        if self.max_seq_length <= 0:
            raise ValueError(f"max_seq_length must be positive, got {self.max_seq_length}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.epochs <= 0 and self.max_steps <= 0:
            raise ValueError("set either epochs or max_steps to a positive value")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError(f"warmup_ratio must be in [0, 1), got {self.warmup_ratio}")

    @property
    def effective_batch_size(self) -> int:
        """What the optimiser actually sees per update.

        Gradient accumulation trades steps for memory: `k` micro-batches are
        run and their gradients summed before one optimiser step, giving the
        convergence behaviour of a `k`-times larger batch at the memory cost of
        the small one. This property is the number that matters for choosing a
        learning rate.
        """
        return self.batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TrainingConfig:
        payload = dict(values)
        lora = payload.pop("lora", None)
        config = cls(**payload)
        if lora is not None:
            config.lora = LoRAConfig(**lora)
            config.__post_init__()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        import yaml  # imported lazily: optional dependency

        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})

    def save(self, directory: str | Path) -> Path:
        """Write the config beside the checkpoint so the run is reproducible."""
        import json  # imported lazily: optional dependency

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        path = target / "training_config.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

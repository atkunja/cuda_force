from __future__ import annotations

import pytest
import torch
from torch import nn
from training.config import LoRAConfig
from training.lora import LoRALinear, count_parameters, describe_parameters

from cudaforge import ops


def test_b_starts_at_zero_so_the_adapted_model_matches_the_base():
    # This is the property that makes LoRA stable: training begins from the
    # pretrained model, not from a perturbed one.
    layer = LoRALinear(32, 16, rank=4)
    x = torch.randn(8, 32)
    torch.testing.assert_close(layer(x), layer.base(x))
    assert torch.count_nonzero(layer.lora_b) == 0


def test_a_is_not_zero_so_the_adapter_can_learn():
    # If both factors started at zero the product's gradient would also be
    # zero and the adapter would never move.
    layer = LoRALinear(32, 16, rank=4)
    assert torch.count_nonzero(layer.lora_a) > 0


def test_the_base_weight_is_frozen():
    layer = LoRALinear(32, 16, rank=4)
    assert not layer.base.weight.requires_grad
    assert not layer.base.bias.requires_grad
    assert layer.lora_a.requires_grad
    assert layer.lora_b.requires_grad


def test_only_the_adapter_receives_gradients():
    layer = LoRALinear(32, 16, rank=4)
    layer(torch.randn(4, 32)).sum().backward()
    assert layer.base.weight.grad is None
    assert layer.lora_a.grad is not None
    assert layer.lora_b.grad is not None


def test_forward_matches_the_explicit_formula():
    layer = LoRALinear(64, 32, rank=8, alpha=16)
    with torch.no_grad():
        layer.lora_b.normal_()

    x = torch.randn(4, 64)
    expected = layer.base(x) + layer.scaling * (x @ layer.lora_a.t() @ layer.lora_b.t())
    torch.testing.assert_close(layer(x), expected, rtol=1e-5, atol=1e-5)


def test_forward_matches_the_cudaforge_operator():
    # The operator takes [in, rank] and [rank, out]; the module stores the
    # transposes. Confirming the two agree is what makes the kernel's parity
    # tests meaningful.
    layer = LoRALinear(64, 32, rank=8, alpha=16, bias=False)
    with torch.no_grad():
        layer.lora_b.normal_()

    x = torch.randn(4, 64)
    from_ops = ops.lora_linear(
        x,
        layer.base.weight.t().contiguous(),
        layer.lora_a.t().contiguous(),
        layer.lora_b.t().contiguous(),
        layer.scaling,
    )
    torch.testing.assert_close(layer(x), from_ops, rtol=1e-4, atol=1e-4)


def test_merging_is_exact():
    # Merging is what makes LoRA add zero inference latency; if it were an
    # approximation, deployment would silently differ from evaluation.
    layer = LoRALinear(48, 24, rank=6, alpha=12)
    with torch.no_grad():
        layer.lora_b.normal_()

    x = torch.randn(16, 48)
    merged = layer.merge()
    torch.testing.assert_close(layer(x), merged(x), rtol=1e-5, atol=1e-5)


def test_scaling_is_alpha_over_rank():
    assert LoRALinear(32, 16, rank=8, alpha=16).scaling == pytest.approx(2.0)
    assert LoRALinear(32, 16, rank=16, alpha=16).scaling == pytest.approx(1.0)


def test_raising_rank_without_alpha_shrinks_the_adapter_scale():
    # A real and easy-to-miss consequence of alpha/r scaling: doubling rank at
    # fixed alpha halves the adapter's influence.
    low = LoRALinear(32, 16, rank=4, alpha=16)
    high = LoRALinear(32, 16, rank=8, alpha=16)
    assert high.scaling < low.scaling


def test_a_rank_above_the_layer_dimensions_is_rejected():
    with pytest.raises(ValueError, match="low-rank"):
        LoRALinear(16, 8, rank=32)


def test_a_non_positive_rank_is_rejected():
    with pytest.raises(ValueError, match="rank"):
        LoRALinear(16, 8, rank=0)


def test_the_adapter_is_a_small_fraction_of_the_parameters():
    layer = LoRALinear(1024, 1024, rank=8)
    # 2 * 1024 * 8 = 16384 against 1024 * 1024 = 1048576, about 1.6%.
    assert layer.trainable_parameters == 2 * 1024 * 8
    assert layer.trainable_parameters / layer.frozen_parameters < 0.02


def test_dropout_is_disabled_in_eval_mode():
    layer = LoRALinear(32, 16, rank=4, dropout=0.5)
    with torch.no_grad():
        layer.lora_b.normal_()
    layer.eval()

    x = torch.randn(4, 32)
    torch.testing.assert_close(layer(x), layer(x))


def test_parameter_counting_helpers():
    model = nn.Sequential(LoRALinear(32, 16, rank=4), nn.ReLU())
    trainable, total = count_parameters(model)
    assert trainable == 4 * 32 + 16 * 4
    assert total > trainable
    assert "trainable" in describe_parameters(model)


def test_lora_config_validation():
    assert LoRAConfig().scaling == pytest.approx(2.0)
    for kwargs, message in [
        ({"rank": 0}, "rank"),
        ({"alpha": 0}, "alpha"),
        ({"dropout": 1.0}, "dropout"),
        ({"target_modules": []}, "target_modules"),
    ]:
        with pytest.raises(ValueError, match=message):
            LoRAConfig(**kwargs)

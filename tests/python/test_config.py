from __future__ import annotations

import pytest
import torch

from cudaforge.config import EngineConfig, GenerationConfig


def test_defaults_are_valid():
    config = EngineConfig()
    assert config.max_batch_size > 0
    assert config.queue_capacity >= config.max_batch_size


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_batch_size", 0),
        ("max_batch_size", -1),
        ("queue_capacity", 0),
        ("worker_threads", 0),
        ("cuda_streams", 0),
        ("max_wait_us", -1),
    ],
)
def test_invalid_sizing_is_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        EngineConfig(**{field: value})


def test_queue_smaller_than_batch_is_rejected():
    # Otherwise the batcher can never reach max_batch_size and throughput is
    # silently capped at the queue depth.
    with pytest.raises(ValueError, match="queue_capacity"):
        EngineConfig(max_batch_size=32, queue_capacity=16)


def test_max_wait_seconds_conversion():
    assert EngineConfig(max_wait_us=5000).max_wait_seconds == pytest.approx(0.005)


def test_explicit_device_is_honoured():
    assert EngineConfig(device="cpu").resolve_device() == torch.device("cpu")


def test_auto_device_resolves_to_something_available():
    device = EngineConfig(device="auto").resolve_device()
    assert device.type in {"cuda", "mps", "cpu"}


def test_explicit_dtype_is_honoured():
    assert EngineConfig(dtype="float16").resolve_dtype() == torch.float16


def test_auto_dtype_off_cuda_is_float32():
    # float16 on CPU is emulated and slower than float32, and MPS bfloat16
    # support is incomplete; float32 is the honest default off CUDA.
    config = EngineConfig(device="cpu", dtype="auto")
    assert config.resolve_dtype() == torch.float32


def test_generation_defaults_are_valid():
    generation = GenerationConfig()
    assert generation.max_new_tokens > 0
    assert not generation.greedy


def test_zero_temperature_means_greedy():
    assert GenerationConfig(temperature=0.0).greedy


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_new_tokens", 0),
        ("temperature", -0.5),
        ("top_p", 0.0),
        ("top_p", 1.5),
        ("top_k", -1),
    ],
)
def test_invalid_generation_params_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        GenerationConfig(**{field: value})


def test_from_dict_round_trip():
    config = EngineConfig.from_dict(
        {
            "model_name": "tiny",
            "max_batch_size": 8,
            "queue_capacity": 64,
            "generation": {"max_new_tokens": 16, "temperature": 0.5},
        }
    )
    assert config.model_name == "tiny"
    assert config.max_batch_size == 8
    assert config.generation.max_new_tokens == 16
    assert config.generation.temperature == pytest.approx(0.5)


def test_from_dict_validates_nested_generation():
    with pytest.raises(ValueError, match="max_new_tokens"):
        EngineConfig.from_dict({"generation": {"max_new_tokens": 0}})


def test_describe_mentions_the_key_knobs():
    text = EngineConfig(max_batch_size=12, max_wait_us=1234).describe()
    assert "max_batch_size=12" in text
    assert "max_wait_us=1234" in text

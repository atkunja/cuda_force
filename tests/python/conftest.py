"""Shared fixtures and CUDA-aware collection rules.

Tests marked ``cuda`` are skipped rather than failed when no GPU is present.
The distinction matters: this project is developed on a machine without one, so
a hard failure would make the suite permanently red and train everyone to
ignore it. A skip with a stated reason keeps the signal honest.
"""

from __future__ import annotations

import pytest
import torch


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if torch.cuda.is_available():
        return
    skip_cuda = pytest.mark.skip(reason="no NVIDIA GPU available on this host")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip_cuda)


@pytest.fixture(autouse=True)
def deterministic_seed() -> None:
    """Fix the RNG for every test.

    Several tests assert on tolerances between two implementations. An
    unseeded input would make a marginal failure appear intermittently, which
    is far harder to diagnose than a consistent one.
    """
    torch.manual_seed(0)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

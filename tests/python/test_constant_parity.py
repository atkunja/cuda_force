"""Cross-language constant parity.

Several constants are defined twice — once in CUDA, once in Python — because
neither language can import the other's. A silent divergence would not fail to
compile; it would produce a numerical mismatch in a parity test, which is a much
harder thing to trace back to its cause.

These tests read the C++/CUDA headers and compare. Textual, and deliberately so:
the alternative is a build-time code generator, which is more machinery than two
constants justify.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def constant_from(path: Path, name: str) -> int:
    """Read `inline constexpr <type> <name> = <value>;` out of a header."""
    source = path.read_text(encoding="utf-8")
    match = re.search(rf"inline\s+constexpr\s+\w+\s+{re.escape(name)}\s*=\s*(\d+)\s*;", source)
    assert match is not None, f"{name} not found in {path}"
    return int(match.group(1))


def test_the_quantisation_block_size_agrees_across_languages():
    # The scales are computed per block; different block sizes on the two sides
    # produce different scales and therefore different dequantised values.
    from cudaforge.ops import QUANT_BLOCK_SIZE

    cuda_value = constant_from(
        REPO_ROOT / "cuda" / "include" / "cudaforge" / "quantization.cuh",
        "kQuantBlockSize",
    )
    assert cuda_value == QUANT_BLOCK_SIZE


def test_the_quantisation_block_size_is_a_power_of_two():
    from cudaforge.ops import QUANT_BLOCK_SIZE

    assert QUANT_BLOCK_SIZE > 0
    assert QUANT_BLOCK_SIZE & (QUANT_BLOCK_SIZE - 1) == 0


def test_the_warp_size_is_thirty_two():
    # The warp primitives assume this at compile time. If it ever changes, the
    # reductions need different code, not a different constant.
    assert (
        constant_from(
            REPO_ROOT / "cpp" / "include" / "cudaforge" / "launch_config.hpp", "kWarpSize"
        )
        == 32
    )


def test_the_default_block_size_is_a_whole_number_of_warps():
    header = REPO_ROOT / "cpp" / "include" / "cudaforge" / "launch_config.hpp"
    warp = constant_from(header, "kWarpSize")
    block = constant_from(header, "kDefaultBlockSize")
    assert block % warp == 0
    assert block <= constant_from(header, "kMaxBlockSize")


@pytest.mark.parametrize(
    ("cpp_field", "python_field"),
    [
        ("requests_received", "requests_received"),
        ("requests_completed", "requests_completed"),
        ("requests_failed", "requests_failed"),
        ("requests_rejected", "requests_rejected"),
        ("requests_expired", "requests_expired"),
        ("batches_processed", "batches_processed"),
        ("batches_closed_by_size", "batches_closed_by_size"),
        ("batches_closed_by_timeout", "batches_closed_by_timeout"),
        ("tokens_generated", "tokens_generated"),
        ("queue_depth", "queue_depth"),
        ("average_batch_size", "average_batch_size"),
        ("uptime_seconds", "uptime_seconds"),
        ("requests_per_second", "requests_per_second"),
        ("tokens_per_second", "tokens_per_second"),
    ],
)
def test_metrics_field_names_match_across_languages(cpp_field, python_field):
    # A dashboard should not need to know which runtime produced a snapshot.
    from cudaforge.metrics import MetricsSnapshot

    header = (REPO_ROOT / "cpp" / "include" / "cudaforge" / "metrics.hpp").read_text(
        encoding="utf-8"
    )
    assert re.search(rf"\b{cpp_field}\b", header), f"{cpp_field} missing from the C++ snapshot"
    assert hasattr(MetricsSnapshot(), python_field)


@pytest.mark.parametrize(
    ("cpp_field", "cpp_default", "python_field"),
    [
        ("max_batch_size", 16, "max_batch_size"),
        ("queue_capacity", 1024, "queue_capacity"),
        ("worker_threads", 4, "worker_threads"),
        ("cuda_streams", 4, "cuda_streams"),
        ("max_prompt_chars", 8192, "max_prompt_chars"),
    ],
)
def test_the_two_runtimes_ship_the_same_defaults(cpp_field, cpp_default, python_field):
    # docs/concurrency.md describes one policy for both runtimes. Divergent
    # defaults would make that description wrong for whichever one a reader
    # happened to be using.
    from cudaforge.config import EngineConfig

    header = (REPO_ROOT / "cpp" / "include" / "cudaforge" / "config.hpp").read_text(
        encoding="utf-8"
    )
    match = re.search(rf"{re.escape(cpp_field)}\s*=\s*(\d+)\s*;", header)
    assert match is not None, f"{cpp_field} not found in the C++ config"

    assert int(match.group(1)) == cpp_default
    assert getattr(EngineConfig(), python_field) == cpp_default


def test_the_default_wait_agrees_across_runtimes():
    from cudaforge.config import EngineConfig

    header = (REPO_ROOT / "cpp" / "include" / "cudaforge" / "config.hpp").read_text(
        encoding="utf-8"
    )
    match = re.search(r"max_wait\{(\d+)\}", header)
    assert match is not None

    assert int(match.group(1)) == EngineConfig().max_wait_us == 5000

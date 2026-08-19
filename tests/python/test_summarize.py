"""Tests for the benchmark result summariser.

The summariser is what turns a result file into something a person pastes into
a report. Two properties matter beyond "it renders": it must not invent a
speedup where the harness measured only one side, and it must carry the
fallback caveat through, so a pasted table cannot be mistaken for a kernel
comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# The summariser lives in benchmarks/, which is not a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))

import summarize_results


def render(tmp_path: Path, payload: dict, capsys) -> str:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert summarize_results.main([str(path)]) == 0
    return capsys.readouterr().out


def test_a_table_aligns_its_columns():
    text = summarize_results.table(["a", "long header"], [["x", "y"]])
    lines = text.strip().splitlines()
    # Header, separator and one row, all the same width.
    assert len(lines) == 3
    assert len({len(line) for line in lines}) == 1


def test_an_empty_table_says_so():
    assert "_no rows_" in summarize_results.table(["a"], [])


def test_the_fallback_caveat_is_carried_through(tmp_path, capsys):
    text = render(
        tmp_path,
        {
            "benchmark": "operators",
            "backend": {"description": "reference", "using_custom_kernels": False},
            "caveat": "Custom CUDA kernels were not used.",
            "results": [
                {"name": "rmsnorm", "variant": "cudaforge", "shape": "4x8", "median_ms": 1.0},
                {"name": "rmsnorm", "variant": "torch", "shape": "4x8", "median_ms": 2.0},
            ],
        },
        capsys,
    )
    assert "Custom CUDA kernels were not used." in text
    assert "dispatch overhead" in text


def test_a_ratio_is_only_reported_when_both_sides_were_measured(tmp_path, capsys):
    text = render(
        tmp_path,
        {
            "benchmark": "operators",
            "backend": {"description": "x", "using_custom_kernels": True},
            "results": [
                # Only one side measured — the ratio must be blank, not inferred.
                {"name": "swiglu", "variant": "cudaforge", "shape": "8x8", "median_ms": 1.0},
                {"name": "rmsnorm", "variant": "cudaforge", "shape": "4x8", "median_ms": 1.0},
                {"name": "rmsnorm", "variant": "torch", "shape": "4x8", "median_ms": 2.0},
            ],
        },
        capsys,
    )
    assert "2.00x" in text  # rmsnorm, both sides present
    assert "| —" in text  # swiglu, one side missing


def test_cuda_results_report_the_share_of_peak_bandwidth(tmp_path, capsys):
    text = render(
        tmp_path,
        {
            "benchmark": "cuda_kernels",
            "device": "Test GPU",
            "compute_capability": "8.0",
            "multiprocessors": 108,
            "theoretical_bandwidth_gb_s": 1000.0,
            "results": [
                {
                    "kernel": "rmsnorm",
                    "variant": "float4",
                    "shape": "4096x1024",
                    "median_ms": 0.1,
                    "effective_bandwidth_gb_s": 800.0,
                }
            ],
        },
        capsys,
    )
    assert "80%" in text
    assert "Test GPU" in text


def test_batching_results_render_the_timeout_fraction(tmp_path, capsys):
    text = render(
        tmp_path,
        {
            "benchmark": "dynamic_batching",
            "note": "simulated",
            "cases": [
                {
                    "clients": 8,
                    "max_batch_size": 16,
                    "max_wait_us": 5000,
                    "requests_per_second": 722.0,
                    "average_batch_size": 8.0,
                    "timeout_closure_fraction": 1.0,
                    "latency_p50_ms": 7.9,
                    "latency_p99_ms": 8.1,
                }
            ],
        },
        capsys,
    )
    assert "100%" in text
    assert "722.0" in text


def test_an_unknown_benchmark_falls_back_to_a_generic_table(tmp_path, capsys):
    text = render(
        tmp_path,
        {"benchmark": "something_new", "cases": [{"threads": 4, "seconds": 1.5}]},
        capsys,
    )
    assert "something_new" in text
    assert "threads" in text


def test_missing_files_are_reported_rather_than_ignored(tmp_path, capsys):
    assert summarize_results.main([str(tmp_path / "absent")]) == 1
    assert "no result files found" in capsys.readouterr().err


def test_malformed_json_is_skipped_with_a_message(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"benchmark": "x", "cases": [{"a": 1}]}), encoding="utf-8")

    # One bad file must not prevent the others from being summarised.
    assert summarize_results.main([str(bad), str(good)]) == 0
    captured = capsys.readouterr()
    assert "skipping" in captured.err
    assert "### x" in captured.out

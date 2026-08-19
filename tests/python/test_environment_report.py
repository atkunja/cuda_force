"""Tests for the environment report.

The report is the basis for every "not measured on this host" claim in the
repository, so it has to be accurate about the one thing that matters: whether
CUDA is available. A report that silently claimed a GPU was present would make
those claims unverifiable.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import environment_report

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_the_report_collects_without_raising():
    # Every probe runs an external command; one missing tool must not take the
    # whole report down, which is the entire point of it existing.
    report = environment_report.collect()
    assert "platform" in report
    assert "tools" in report
    assert "cuda" in report


def test_a_missing_tool_is_reported_as_absent():
    tool = environment_report.probe("nonexistent", ["definitely-not-a-real-command-xyz"])
    assert not tool.available
    assert tool.version is None


def test_a_present_tool_reports_its_version():
    tool = environment_report.probe("python", [sys.executable, "--version"])
    assert tool.available
    assert "Python" in (tool.version or "")


def test_the_cuda_verdict_matches_torch():
    import torch

    report = environment_report.collect()
    assert report["cuda"]["device_visible"] == torch.cuda.is_available()


def test_capabilities_gate_cuda_work_on_a_visible_device():
    report = environment_report.collect()
    verdicts = dict(
        (label, possible) for label, possible, _ in environment_report.capabilities(report)
    )

    if not report["cuda"]["device_visible"]:
        assert not verdicts["Run the CUDA tests"]
        assert not verdicts["Benchmark the CUDA kernels"]
        assert not verdicts["Profile with Nsight"]


def test_capabilities_reflect_the_installed_packages():
    report = environment_report.collect()
    verdicts = dict(
        (label, possible) for label, possible, _ in environment_report.capabilities(report)
    )
    # torch is a hard dependency of the test suite, so this must be true here.
    assert verdicts["Run the Python test suite"]


def test_the_json_output_is_parseable():
    result = subprocess.run(
        [sys.executable, "scripts/environment_report.py", "--json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["platform"]["python"]
    assert isinstance(payload["cuda"]["device_visible"], bool)


def test_the_human_output_states_the_gpu_situation():
    result = subprocess.run(
        [sys.executable, "scripts/environment_report.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    import torch

    if not torch.cuda.is_available():
        # The claim the whole repository rests on must be stated, not implied.
        assert "No NVIDIA GPU is visible" in result.stdout
        assert "no GPU performance numbers" in result.stdout

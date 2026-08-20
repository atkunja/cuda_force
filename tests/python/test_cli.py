"""Command-line entry point tests.

The CLI is thin, but it is the first thing anyone runs, so its argument
handling and output shape are worth pinning down. Everything here uses the
deterministic runner so no model is downloaded.
"""

from __future__ import annotations

import json

import pytest

from cudaforge import cli


def test_bench_runs_and_reports_a_table(capsys):
    exit_code = cli.bench(
        ["--echo-runner", "--clients", "2", "--requests-per-client", "5", "--max-new-tokens", "4"]
    )
    assert exit_code == 0

    output = capsys.readouterr().out
    assert "throughput" in output
    assert "avg batch size" in output
    assert "latency p50/p95/p99" in output


def test_bench_json_output_is_parseable(capsys):
    exit_code = cli.bench(
        ["--echo-runner", "--clients", "2", "--requests-per-client", "5", "--json"]
    )
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["load"]["completed"] == 10
    assert payload["load"]["failed"] == 0
    assert "using_custom_cuda_kernels" in payload
    assert payload["metrics"]["requests_completed"] == 10


def test_bench_reports_the_active_backend(capsys):
    # Printed so a run that silently fell back to the reference path is visible
    # rather than inferred from a suspiciously slow number.
    cli.bench(["--echo-runner", "--clients", "1", "--requests-per-client", "2", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert "cudaforge:" in payload["backend"]


def test_bench_honours_the_batching_arguments(capsys):
    cli.bench(
        [
            "--echo-runner",
            "--clients",
            "1",
            "--requests-per-client",
            "2",
            "--max-batch-size",
            "7",
            "--max-wait-us",
            "1234",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["max_batch_size"] == 7
    assert payload["config"]["max_wait_us"] == 1234


def test_queue_capacity_is_raised_to_fit_the_batch(capsys):
    # A queue smaller than a batch would make the configured batch size
    # unreachable; the CLI raises it rather than failing validation.
    exit_code = cli.bench(
        [
            "--echo-runner",
            "--clients",
            "1",
            "--requests-per-client",
            "1",
            "--max-batch-size",
            "64",
            "--queue-capacity",
            "8",
            "--json",
        ]
    )
    assert exit_code == 0


def test_an_unknown_argument_exits_with_an_error():
    with pytest.raises(SystemExit) as excinfo:
        cli.bench(["--not-a-real-flag"])
    assert excinfo.value.code == 2


def test_help_exits_cleanly():
    with pytest.raises(SystemExit) as excinfo:
        cli.bench(["--help"])
    assert excinfo.value.code == 0


def test_version_flag_reports_the_package_version(capsys):
    import cudaforge

    with pytest.raises(SystemExit) as excinfo:
        cli.bench(["--version"])
    assert excinfo.value.code == 0
    assert cudaforge.__version__ in capsys.readouterr().out


def test_both_entry_points_accept_version():
    for entry in (cli.bench, cli.serve):
        with pytest.raises(SystemExit) as excinfo:
            entry(["--version"])
        assert excinfo.value.code == 0


def test_a_yaml_config_is_loaded(capsys):
    cli.bench(
        [
            "--config",
            "inference/configs/latency.yaml",
            "--echo-runner",
            "--clients",
            "1",
            "--requests-per-client",
            "2",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["model"] == "gpt2"
    assert payload["config"]["max_batch_size"] == 4


def test_an_explicit_flag_overrides_the_config_file(capsys):
    # argparse cannot tell "passed the default" from "passed nothing", so the
    # override is decided by comparing against the parser's defaults. This is
    # the case that would break if that comparison were wrong.
    cli.bench(
        [
            "--config",
            "inference/configs/latency.yaml",
            "--max-batch-size",
            "32",
            "--echo-runner",
            "--clients",
            "1",
            "--requests-per-client",
            "2",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["max_batch_size"] == 32
    # Unspecified fields still come from the file.
    assert payload["config"]["max_wait_us"] == 1000


def test_a_config_override_still_widens_the_queue(capsys):
    cli.bench(
        [
            "--config",
            "inference/configs/latency.yaml",
            "--max-batch-size",
            "512",
            "--echo-runner",
            "--clients",
            "1",
            "--requests-per-client",
            "1",
            "--json",
        ]
    )
    # The file's queue_capacity of 256 is below the overridden batch size, which
    # would make the batch size unreachable; the CLI widens it.
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["max_batch_size"] == 512


def test_serve_honours_the_config_file(monkeypatch):
    # serve() hands its resolved values to the server through the environment.
    # Passing the raw --model default straight through would silently ignore
    # the config file, and the server would load the wrong model.
    import os

    captured: dict[str, str] = {}

    def fake_run(*_args, **_kwargs):
        captured.update(
            {key: value for key, value in os.environ.items() if key.startswith("CUDAFORGE_")}
        )

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)
    cli.serve(["--config", "inference/configs/throughput.yaml"])

    assert captured["CUDAFORGE_MODEL"] == "gpt2"
    assert captured["CUDAFORGE_MAX_BATCH"] == "64"
    assert captured["CUDAFORGE_MAX_WAIT_US"] == "20000"


def test_serve_flags_override_the_config_file(monkeypatch):
    import os

    captured: dict[str, str] = {}

    def fake_run(*_args, **_kwargs):
        captured.update(
            {key: value for key, value in os.environ.items() if key.startswith("CUDAFORGE_")}
        )

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)
    cli.serve(["--config", "inference/configs/throughput.yaml", "--max-batch-size", "8"])

    assert captured["CUDAFORGE_MAX_BATCH"] == "8"
    assert captured["CUDAFORGE_MAX_WAIT_US"] == "20000"


# --- scheduler selection ----------------------------------------------------


def test_bench_reports_scheduling_counters_in_continuous_mode(capsys):
    """Batch-formation counters must not be printed by a scheduler that has none.

    An iteration-level scheduler never forms a batch, so `avg batch size` would
    read 0.00 and `batches` 0 — which looks like "nothing was batched" rather
    than "this scheduler does not work that way".
    """
    exit_code = cli.bench(
        [
            "--echo-runner",
            "--continuous",
            "--clients",
            "2",
            "--requests-per-client",
            "5",
            "--max-new-tokens",
            "4",
        ]
    )
    assert exit_code == 0

    output = capsys.readouterr().out
    assert "scheduler        continuous" in output
    assert "decode steps" in output
    assert "row occupancy" in output
    assert "avg batch size" not in output
    assert "batches  " not in output


def test_bench_still_reports_batch_formation_in_static_mode(capsys):
    cli.bench(["--echo-runner", "--clients", "2", "--requests-per-client", "5"])
    output = capsys.readouterr().out
    assert "scheduler        static" in output
    assert "avg batch size" in output
    assert "decode steps" not in output


def test_bench_json_names_the_scheduler(capsys):
    cli.bench(
        ["--echo-runner", "--continuous", "--clients", "2", "--requests-per-client", "5", "--json"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["config"]["scheduler"] == "continuous"
    assert payload["load"]["completed"] == 10
    assert payload["scheduling"]["completions"] == 10
    assert payload["scheduling"]["decode_steps"] > 0


def test_bench_json_omits_scheduling_for_the_static_engine(capsys):
    cli.bench(["--echo-runner", "--clients", "2", "--requests-per-client", "5", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["config"]["scheduler"] == "static"
    assert "scheduling" not in payload


def test_serve_passes_the_scheduler_choice_through_the_environment(monkeypatch):
    import os

    captured: dict[str, str] = {}

    def fake_run(*_args, **_kwargs):
        captured.update(
            {key: value for key, value in os.environ.items() if key.startswith("CUDAFORGE_")}
        )

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)
    cli.serve(["--echo-runner", "--continuous"])
    assert captured["CUDAFORGE_CONTINUOUS"] == "1"

    captured.clear()
    monkeypatch.delenv("CUDAFORGE_CONTINUOUS", raising=False)
    cli.serve(["--echo-runner"])
    assert "CUDAFORGE_CONTINUOUS" not in captured

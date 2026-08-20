"""Command-line entry points.

Two commands, both thin wrappers over library code:

    cudaforge-serve   start the HTTP server
    cudaforge-bench   drive the engine with concurrent load and report metrics

The benchmark lives here rather than in ``benchmarks/`` because it exercises
the installed package end to end, which is what someone evaluating the project
on their own hardware will reach for first.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass

from cudaforge.config import EngineConfig, GenerationConfig
from cudaforge.continuous_engine import ContinuousEngine
from cudaforge.engine import EngineClosedError, InferenceEngine, ServingEngine
from cudaforge.ops import backend_report
from cudaforge.runners import EchoRunner, ModelRunner, TransformersRunner
from cudaforge.stepwise import EchoStepwiseRunner


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    from cudaforge import __version__  # imported lazily: avoids a circular import

    parser.add_argument("--version", action="version", version=f"cudaforge {__version__}")


def _add_engine_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        help="YAML engine config; individual flags below override its values",
    )
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--max-wait-us", type=int, default=5_000)
    parser.add_argument("--queue-capacity", type=int, default=1024)
    parser.add_argument("--worker-threads", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--echo-runner",
        action="store_true",
        help="use the deterministic runner instead of loading a model",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="schedule at iteration level: refill rows as sequences finish, "
        "rather than running each batch to completion",
    )


def _config_from(args: argparse.Namespace) -> EngineConfig:
    """Build the config, letting explicit flags override a YAML file.

    argparse cannot distinguish "the user passed the default" from "the user
    passed nothing", so the overrides are applied by comparing against the
    parser's own defaults. That keeps `--config x.yaml --max-batch-size 32`
    behaving the way anyone would expect.
    """
    if args.config is None:
        return EngineConfig(
            model_name=args.model,
            device=args.device,
            max_batch_size=args.max_batch_size,
            max_wait_us=args.max_wait_us,
            queue_capacity=max(args.queue_capacity, args.max_batch_size),
            worker_threads=args.worker_threads,
        )

    config = EngineConfig.from_yaml(args.config)
    defaults = {
        "model": "sshleifer/tiny-gpt2",
        "device": "auto",
        "max_batch_size": 16,
        "max_wait_us": 5_000,
        "queue_capacity": 1024,
        "worker_threads": 4,
    }
    for flag, field in (
        ("model", "model_name"),
        ("device", "device"),
        ("max_batch_size", "max_batch_size"),
        ("max_wait_us", "max_wait_us"),
        ("queue_capacity", "queue_capacity"),
        ("worker_threads", "worker_threads"),
    ):
        value = getattr(args, flag)
        if value != defaults[flag]:
            setattr(config, field, value)

    config.queue_capacity = max(config.queue_capacity, config.max_batch_size)
    config.__post_init__()
    return config


def _build(args: argparse.Namespace) -> tuple[EngineConfig, ModelRunner]:
    config = _config_from(args)
    runner: ModelRunner = EchoRunner() if args.echo_runner else TransformersRunner(config)
    return config, runner


def _build_engine(args: argparse.Namespace) -> tuple[EngineConfig, ServingEngine]:
    """Build whichever engine the flags ask for.

    The two take different runner protocols, so the runner is chosen alongside
    the engine rather than passed into either.
    """
    config = _config_from(args)
    if not getattr(args, "continuous", False):
        runner: ModelRunner = EchoRunner() if args.echo_runner else TransformersRunner(config)
        return config, InferenceEngine(config=config, runner=runner)

    if args.echo_runner:
        return config, ContinuousEngine(config=config, runner=EchoStepwiseRunner())

    from cudaforge.stepwise_transformers import TransformersStepwiseRunner  # imported lazily

    return config, ContinuousEngine(config=config, runner=TransformersStepwiseRunner(config))


def serve(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cudaforge-serve", description="Start the HTTP server")
    _add_common_arguments(parser)
    _add_engine_arguments(parser)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    import os  # imported lazily: optional dependency

    import uvicorn  # imported lazily: optional dependency

    # Resolved through the same path as `bench`, so a --config file is honoured
    # and flags override it identically. Passing args.model straight through
    # would silently ignore the config file's model.
    config = _config_from(args)

    os.environ["CUDAFORGE_MODEL"] = config.model_name
    os.environ["CUDAFORGE_DEVICE"] = config.device
    os.environ["CUDAFORGE_MAX_BATCH"] = str(config.max_batch_size)
    os.environ["CUDAFORGE_MAX_WAIT_US"] = str(config.max_wait_us)
    os.environ["CUDAFORGE_QUEUE_CAPACITY"] = str(config.queue_capacity)
    os.environ["CUDAFORGE_WORKER_THREADS"] = str(config.worker_threads)
    if args.echo_runner:
        os.environ["CUDAFORGE_ECHO_RUNNER"] = "1"
    if args.continuous:
        os.environ["CUDAFORGE_CONTINUOUS"] = "1"

    uvicorn.run("inference.server:app", host=args.host, port=args.port, log_level="info")
    return 0


@dataclass
class LoadResult:
    completed: int
    failed: int
    rejected: int
    wall_seconds: float


def _drive(engine: ServingEngine, clients: int, per_client: int, tokens: int) -> LoadResult:
    """Run `clients` threads each issuing `per_client` requests.

    Threads submit independently rather than through ``generate_many`` so the
    arrival pattern resembles real concurrent traffic — which is what the
    batcher is being measured on.
    """
    completed = 0
    failed = 0
    rejected = 0
    lock = threading.Lock()
    generation = GenerationConfig(max_new_tokens=tokens)

    def worker(index: int) -> None:
        nonlocal completed, failed, rejected
        local_ok = local_fail = local_reject = 0
        for i in range(per_client):
            try:
                response = engine.generate(f"client-{index} request-{i}", generation)
            except EngineClosedError:
                local_reject += 1
                continue
            if response.ok:
                local_ok += 1
            else:
                local_fail += 1
        with lock:
            completed += local_ok
            failed += local_fail
            rejected += local_reject

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(clients)]
    start = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return LoadResult(completed, failed, rejected, time.monotonic() - start)


def bench(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cudaforge-bench", description="Drive the engine with concurrent load"
    )
    _add_common_arguments(parser)
    _add_engine_arguments(parser)
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--requests-per-client", type=int, default=25)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args(argv)

    config, built = _build_engine(args)
    report = backend_report()

    with built as engine:
        result = _drive(engine, args.clients, args.requests_per_client, args.max_new_tokens)
        snapshot = engine.snapshot()
        # Only the continuous engine has these; the static one reports batch
        # formation instead.
        scheduling = engine.stats() if isinstance(engine, ContinuousEngine) else None

    payload = {
        "backend": str(report),
        "using_custom_cuda_kernels": report.using_custom_kernels,
        "config": {
            "model": config.model_name,
            "device": str(config.resolve_device()),
            "max_batch_size": config.max_batch_size,
            "max_wait_us": config.max_wait_us,
            "worker_threads": config.worker_threads,
            "scheduler": "continuous" if scheduling is not None else "static",
        },
        "load": {
            "clients": args.clients,
            "requests_per_client": args.requests_per_client,
            "max_new_tokens": args.max_new_tokens,
            "completed": result.completed,
            "failed": result.failed,
            "rejected": result.rejected,
            "wall_seconds": round(result.wall_seconds, 4),
        },
        "metrics": snapshot.to_dict(),
    }

    if scheduling is not None:
        payload["scheduling"] = {
            "decode_steps": scheduling.decode_steps,
            "admissions": scheduling.admissions,
            "completions": scheduling.completions,
            "utilisation": scheduling.utilisation,
            "max_observed_batch": scheduling.max_observed_batch,
        }

    if args.json:
        json.dump(payload, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0

    print(payload["backend"])
    print(f"  model            {config.model_name}  device={config.resolve_device()}")
    print(f"  scheduler        {'continuous' if scheduling is not None else 'static'}")
    print(
        f"  batching         max_batch_size={config.max_batch_size} "
        f"max_wait_us={config.max_wait_us}"
    )
    print(f"  load             {args.clients} clients x {args.requests_per_client} requests")
    print()
    print(f"  completed        {result.completed}")
    print(f"  failed           {result.failed}")
    print(f"  rejected         {result.rejected}")
    print(f"  wall time        {result.wall_seconds:.3f} s")
    print(f"  throughput       {snapshot.requests_per_second:.1f} req/s")
    print(f"  tokens/s         {snapshot.tokens_per_second:.1f}")
    if scheduling is None:
        print(f"  avg batch size   {snapshot.average_batch_size:.2f}")
        print(
            f"  batches          {snapshot.batches_processed} "
            f"(size={snapshot.batches_closed_by_size}, "
            f"timeout={snapshot.batches_closed_by_timeout})"
        )
    else:
        # Batch-formation counters are meaningless here — an iteration-level
        # scheduler never forms a batch. Printing them would read as "nothing
        # was batched" rather than "this scheduler does not work that way".
        print(f"  decode steps     {scheduling.decode_steps}")
        print(f"  row occupancy    {scheduling.utilisation:.1%}")
        print(f"  peak batch       {scheduling.max_observed_batch}")
    print(f"  queue delay p99  {snapshot.queue_delay_p99_ms:.2f} ms")
    print(
        f"  latency p50/p95/p99  {snapshot.latency_p50_ms:.2f} / "
        f"{snapshot.latency_p95_ms:.2f} / {snapshot.latency_p99_ms:.2f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(bench())

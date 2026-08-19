#!/usr/bin/env python3
"""Report what this machine can and cannot build or run.

    python scripts/environment_report.py            # human-readable
    python scripts/environment_report.py --json     # machine-readable

`docs/environment.md` records the development host, and that record is the
basis for every "not measured on this host" claim in the repository. A
hand-written record drifts; this produces the same facts on demand, so the claim
can be checked rather than trusted.

It also answers the question anyone cloning this asks first: what will actually
run here?
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass


@dataclass
class Tool:
    name: str
    available: bool
    version: str | None
    note: str = ""


def probe(name: str, command: list[str], note: str = "") -> Tool:
    """Report a tool's presence and version without letting a failure propagate."""
    if shutil.which(command[0]) is None:
        return Tool(name, False, None, note)
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        return Tool(name, False, None, f"{note} ({error})".strip())

    output = (result.stdout or result.stderr).strip().splitlines()
    return Tool(name, True, output[0] if output else "", note)


def python_package(name: str) -> Tool:
    try:
        module = __import__(name)
    except ImportError:
        return Tool(name, False, None)
    return Tool(name, True, getattr(module, "__version__", "unknown"))


def collect() -> dict[str, object]:
    tools = [
        probe("cmake", ["cmake", "--version"]),
        probe("ninja", ["ninja", "--version"]),
        probe("clang++", ["clang++", "--version"]),
        probe("g++", ["g++", "--version"]),
        probe("clang-format", ["clang-format", "--version"]),
        probe("shellcheck", ["shellcheck", "--version"]),
        probe("docker", ["docker", "--version"]),
        probe("git", ["git", "--version"]),
        probe("nvcc", ["nvcc", "--version"], note="required to build the CUDA targets"),
        probe("nvidia-smi", ["nvidia-smi", "--version"], note="required to run them"),
        probe("nsys", ["nsys", "--version"], note="Nsight Systems"),
        probe("ncu", ["ncu", "--version"], note="Nsight Compute"),
    ]
    packages = [python_package(name) for name in ("torch", "numpy", "transformers", "peft")]

    cuda_available = False
    device = None
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or platform.machine(),
            "python": platform.python_version(),
        },
        "tools": [asdict(tool) for tool in tools],
        "python_packages": [asdict(package) for package in packages],
        "cuda": {
            "toolkit": any(tool.name == "nvcc" and tool.available for tool in tools),
            "device_visible": cuda_available,
            "device_name": device,
        },
    }


def capabilities(report: dict) -> list[tuple[str, bool, str]]:
    """What this machine can actually do, which is the question that matters."""
    tools = {tool["name"]: tool["available"] for tool in report["tools"]}
    packages = {p["name"]: p["available"] for p in report["python_packages"]}
    cuda = report["cuda"]

    return [
        ("Build the portable C++ runtime", tools.get("cmake", False), "needs cmake"),
        (
            "Run the C++ tests under sanitizers",
            tools.get("cmake", False) and tools.get("clang++", False),
            "needs cmake and clang",
        ),
        ("Run the Python test suite", packages.get("torch", False), "needs torch"),
        ("Run the reference operators", packages.get("torch", False), "needs torch"),
        ("Fine-tune with LoRA", packages.get("peft", False), "needs peft"),
        ("Compile the CUDA kernels", cuda["toolkit"], "needs the CUDA toolkit"),
        ("Run the CUDA tests", cuda["device_visible"], "needs an NVIDIA GPU"),
        ("Benchmark the CUDA kernels", cuda["device_visible"], "needs an NVIDIA GPU"),
        (
            "Profile with Nsight",
            cuda["device_visible"] and tools.get("nsys", False),
            "needs an NVIDIA GPU and Nsight",
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = collect()
    if args.json:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    host = report["platform"]
    print(f"{host['system']} {host['release']} on {host['machine']}")
    print(f"Python {host['python']}\n")

    print("tools")
    for tool in report["tools"]:
        mark = "yes" if tool["available"] else " no"
        detail = tool["version"] or tool["note"] or ""
        print(f"  [{mark}] {tool['name']:<14} {detail}")

    print("\npython packages")
    for package in report["python_packages"]:
        mark = "yes" if package["available"] else " no"
        print(f"  [{mark}] {package['name']:<14} {package['version'] or ''}")

    print("\nwhat this machine can do")
    for label, possible, requirement in capabilities(report):
        mark = "yes" if possible else " no"
        suffix = "" if possible else f"  — {requirement}"
        print(f"  [{mark}] {label}{suffix}")

    if not report["cuda"]["device_visible"]:
        print(
            "\nNo NVIDIA GPU is visible. The CUDA kernels, their tests and every GPU\n"
            "benchmark are unavailable here — which is why this repository contains\n"
            "no GPU performance numbers. See docs/environment.md."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

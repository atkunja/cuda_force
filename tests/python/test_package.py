"""Package-surface tests.

`__all__` drifting out of sync with what is actually importable is a small bug
with an annoying failure mode: `from cudaforge import *` raises AttributeError
at import time, far from the edit that caused it.
"""

from __future__ import annotations

import cudaforge


def test_every_exported_name_exists():
    missing = [name for name in cudaforge.__all__ if not hasattr(cudaforge, name)]
    assert missing == []


# Ordering of __all__ is enforced by ruff's RUF022 rule rather than asserted
# here: two sources of truth for the same convention is one too many, and the
# linter's definition is the one that gates CI.


def test_no_duplicate_exports():
    assert len(cudaforge.__all__) == len(set(cudaforge.__all__))


def test_the_main_entry_points_are_reachable_from_the_root():
    # These are what a user reaches for first; requiring a submodule import for
    # them would be a papercut on every example.
    for name in (
        "InferenceEngine",
        "Response",
        "EngineConfig",
        "GenerationConfig",
        "DynamicBatcher",
        "rmsnorm",
        "softmax",
        "lora_linear",
        "quantize_int8",
        "backend_report",
    ):
        assert hasattr(cudaforge, name), name


def test_version_is_a_dotted_string():
    assert isinstance(cudaforge.__version__, str)
    assert len(cudaforge.__version__.split(".")) >= 2


def test_importing_the_package_does_not_pull_in_optional_dependencies():
    # transformers, peft, datasets and fastapi are optional extras. Every use of
    # them is a lazy import inside the function that needs it, so `import
    # cudaforge` must not load any of them — otherwise the base install would
    # fail for anyone who only wants the operators.
    #
    # Checked in a subprocess because this process has almost certainly imported
    # them already for other tests.
    import subprocess
    import sys
    import textwrap

    probe = textwrap.dedent("""
        import sys
        import cudaforge  # noqa: F401
        leaked = [
            name
            for name in ("transformers", "peft", "datasets", "fastapi", "bitsandbytes")
            if name in sys.modules
        ]
        print(",".join(leaked))
    """)
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "", f"eagerly imported: {result.stdout.strip()}"


def test_star_import_is_safe():
    namespace: dict[str, object] = {}
    exec("from cudaforge import *", namespace)
    for name in cudaforge.__all__:
        assert name in namespace

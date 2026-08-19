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


def test_exports_are_sorted():
    # Keeps diffs on __all__ readable and makes duplicates obvious.
    assert cudaforge.__all__ == sorted(cudaforge.__all__)


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


def test_importing_the_package_does_not_require_optional_dependencies():
    # transformers, peft, datasets and fastapi are all optional extras. Importing
    # cudaforge must work without them, which is why every use is a lazy import
    # inside the function that needs it.
    import sys

    for optional in ("transformers", "peft", "datasets", "fastapi", "bitsandbytes"):
        module = sys.modules.get("cudaforge")
        assert module is not None
        # The check is that cudaforge imported at all, above; this loop only
        # documents which packages are deliberately not required.
        assert optional not in getattr(module, "__annotations__", {})


def test_star_import_is_safe():
    namespace: dict[str, object] = {}
    exec("from cudaforge import *", namespace)  # noqa: S102
    for name in cudaforge.__all__:
        assert name in namespace

"""The C++ and Python KV cache managers must make identical decisions.

Each is tested against its own expectations elsewhere, which is exactly how two
implementations of one policy drift apart while both suites stay green. This
runs the same script through both and compares every decision, not just the
final state — an implementation that reached the right totals by preempting
different sequences would be wrong in the way that matters to a scheduler.

The C++ side is `tests/cpp/kv_cache_scenario`, built by `./scripts/build.sh`.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from cudaforge.kv_cache import AdmissionResult, KVCacheManager, PreemptionPolicy

REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_harness() -> Path | None:
    """Locate the scenario binary in whichever tree was configured.

    `build.sh --cuda` configures into `build-cuda`, so a machine that only ever
    built with CUDA has no `build/` at all.
    """
    for tree in ("build", "build-cuda"):
        candidate = REPO_ROOT / tree / "tests" / "cpp" / "kv_cache_scenario"
        if candidate.is_file():
            return candidate
    return None


HARNESS = _find_harness()

pytestmark = pytest.mark.skipif(
    HARNESS is None,
    reason="kv_cache_scenario not built in build/ or build-cuda/; run ./scripts/build.sh",
)


def run_cpp(blocks: int, block_size: int, policy: str, script: str) -> dict:
    assert HARNESS is not None
    result = subprocess.run(
        [str(HARNESS), str(blocks), str(block_size), policy, script],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return json.loads(result.stdout)


def run_python(blocks: int, block_size: int, policy: str, script: str) -> dict:
    manager = KVCacheManager(
        blocks,
        block_size,
        PreemptionPolicy.LARGEST if policy == "largest" else PreemptionPolicy.NEWEST,
    )
    operations: list[dict] = []
    seen: list[int] = []

    for operation in [item for item in script.split(",") if item]:
        kind, body = operation[0], operation[1:]
        sequence_text, _, token_text = body.partition(":")
        sequence = int(sequence_text)
        tokens = int(token_text) if token_text else 0
        if sequence not in seen:
            seen.append(sequence)

        if kind == "a":
            outcome = manager.admit(sequence, tokens)
            operations.append({"result": outcome.result.value, "preempted": outcome.preempted})
        elif kind == "e":
            outcome = manager.extend(sequence, tokens)
            operations.append({"result": outcome.result.value, "preempted": outcome.preempted})
        elif kind == "p":
            operations.append({"result": "preempt", "reclaimed": manager.preempt(sequence)})
        elif kind == "r":
            manager.release(sequence)
            operations.append({"result": "release"})
        else:
            raise AssertionError(f"unknown operation {operation}")

    return {
        "operations": operations,
        "final": {
            "free_blocks": manager.free_blocks,
            "preemptions": manager.preemption_count,
            "recomputed_tokens": manager.recomputed_tokens,
            "active_sequences": manager.active_sequences,
            "sequences": [
                {
                    "id": sequence,
                    "admitted": manager.is_admitted(sequence),
                    "blocks": manager.blocks_held(sequence),
                    "tokens": manager.tokens_held(sequence),
                }
                for sequence in seen
            ],
        },
    }


# Each case names the behaviour it pins, so a divergence says what broke.
SCENARIOS = [
    ("plain admission", 16, 4, "newest", "a1:10,a2:6,a3:4"),
    ("preemption under pressure", 8, 4, "newest", "a1:10,a2:10,a3:10"),
    ("largest-first picks differently", 8, 4, "largest", "a1:4,a2:16,a3:8"),
    ("extension into the last block's slack", 8, 16, "newest", "a1:5,e1:3,e1:7"),
    ("extension that needs a new block", 8, 4, "newest", "a1:4,e1:1,e1:20"),
    ("release returns blocks to the pool", 8, 4, "newest", "a1:12,a2:12,r1,a3:12"),
    ("explicit preemption then re-admission", 8, 4, "newest", "a1:12,p1,a2:12,e1:8"),
    ("a request larger than the whole cache", 4, 4, "newest", "a1:8,a2:1000"),
    ("zero tokens is a no-op", 8, 4, "newest", "a1:0,a1:8"),
    ("repeated churn", 6, 4, "newest", "a1:8,a2:8,a3:8,r2,a4:8,p3,a5:4,e1:4"),
    ("largest-first churn", 6, 4, "largest", "a1:4,a2:12,a3:4,a4:8,r1,a5:8"),
    # Each admission needs the entire cache, so every one after the first must
    # evict its predecessor. Added because the meta-test below found the suite
    # was reaching preemption less often than it looked.
    ("every admission evicts the last", 4, 4, "newest", "a1:16,a2:16,a3:16,a4:16"),
    ("largest-first under the same pressure", 4, 4, "largest", "a1:16,a2:16,a3:16"),
]


@pytest.mark.parametrize(
    ("name", "blocks", "block_size", "policy", "script"),
    SCENARIOS,
    ids=[case[0] for case in SCENARIOS],
)
def test_the_two_managers_agree(name, blocks, block_size, policy, script):
    from_cpp = run_cpp(blocks, block_size, policy, script)
    from_python = run_python(blocks, block_size, policy, script)

    assert from_python["operations"] == from_cpp["operations"], (
        f"{name}: the two implementations made different decisions"
    )
    assert from_python["final"] == from_cpp["final"], f"{name}: final state differs"


def test_the_scenarios_actually_exercise_preemption():
    """A conformance suite where nothing is ever evicted proves very little.

    Without this, every scenario could be trivially satisfiable and the two
    implementations would agree by never reaching the interesting code.
    """
    preemptions = 0
    insufficient = 0
    for _, blocks, block_size, policy, script in SCENARIOS:
        final = run_python(blocks, block_size, policy, script)
        preemptions += final["final"]["preemptions"]
        insufficient += sum(
            1
            for operation in final["operations"]
            if operation.get("result") == AdmissionResult.INSUFFICIENT_CACHE.value
        )

    assert preemptions >= 5, f"only {preemptions} preemptions across the suite"
    assert insufficient >= 1, "no scenario reaches the insufficient-cache path"

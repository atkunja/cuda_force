"""Documentation link tests.

This repository leans hard on cross-references — the concepts index is almost
entirely links — so a broken one is a real defect, not a cosmetic one. These
tests check both directions: the checker catches what it should, and the actual
documentation passes it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# The checker lives in scripts/, which is not a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import check_docs

REPO_ROOT = Path(__file__).resolve().parents[2]


def problems_for(tmp_path: Path, body: str, name: str = "page.md") -> list[str]:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return [problem.reason for problem in check_docs.check_file(path, tmp_path)]


def test_a_missing_file_is_reported(tmp_path):
    assert "file does not exist" in problems_for(tmp_path, "See [x](nowhere.md).")


def test_an_existing_file_is_accepted(tmp_path):
    (tmp_path / "other.md").write_text("# Other\n", encoding="utf-8")
    assert problems_for(tmp_path, "See [x](other.md).") == []


def test_a_missing_anchor_is_reported(tmp_path):
    (tmp_path / "other.md").write_text("# Real Heading\n", encoding="utf-8")
    assert "no such heading in target" in problems_for(tmp_path, "See [x](other.md#absent).")


def test_a_present_anchor_is_accepted(tmp_path):
    (tmp_path / "other.md").write_text("# Real Heading\n", encoding="utf-8")
    assert problems_for(tmp_path, "See [x](other.md#real-heading).") == []


def test_a_same_page_anchor_is_resolved(tmp_path):
    assert problems_for(tmp_path, "# Top\n\nSee [x](#top).") == []


def test_a_missing_same_page_anchor_is_reported(tmp_path):
    assert "no such heading" in problems_for(tmp_path, "# Top\n\nSee [x](#bottom).")


def test_external_links_are_not_fetched(tmp_path):
    # Network checks make the result depend on connectivity and on who is
    # running them, which is the opposite of what a repository check should be.
    assert problems_for(tmp_path, "See [x](https://example.invalid/nope).") == []


def test_duplicate_headings_are_reported(tmp_path):
    # Two identical headings make any anchor to them ambiguous.
    reasons = problems_for(tmp_path, "# Same\n\ntext\n\n# Same\n")
    assert "duplicate heading makes this anchor ambiguous" in reasons


def test_links_inside_code_fences_are_ignored(tmp_path):
    body = "# Top\n\n```\nSee [x](nowhere.md).\n```\n"
    assert problems_for(tmp_path, body) == []


def test_headings_inside_code_fences_are_not_headings(tmp_path):
    # A shell comment in a fenced block would otherwise register as a heading.
    body = "# Top\n\n```bash\n# not a heading\n```\n\nSee [x](#not-a-heading).\n"
    assert "no such heading" in problems_for(tmp_path, body)


def test_each_space_becomes_one_hyphen(tmp_path):
    # GitHub anchors "Kernel A — Reduction" as "kernel-a--reduction": the em
    # dash is removed and each of the two surrounding spaces becomes a hyphen.
    assert check_docs.slugify("Kernel A — Reduction") == "kernel-a--reduction"
    assert check_docs.slugify("Simple Heading") == "simple-heading"
    assert check_docs.slugify("`code` in a heading") == "code-in-a-heading"


def test_the_repository_documentation_has_no_broken_links():
    result = subprocess.run(
        [sys.executable, "scripts/check_docs.py", "."],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout


# --- documented file paths -------------------------------------------------


def check_references_module():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import check_references

    return check_references


def test_a_named_file_that_is_missing_is_reported(tmp_path):
    checker = check_references_module()
    page = tmp_path / "page.md"
    page.write_text("See `src/nowhere.py` for details.", encoding="utf-8")
    problems = checker.check_file(page, tmp_path)
    assert len(problems) == 1
    assert "nowhere.py" in problems[0]


def test_a_named_file_that_exists_is_accepted(tmp_path):
    checker = check_references_module()
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "here.py").write_text("", encoding="utf-8")
    page = tmp_path / "page.md"
    page.write_text("See `src/here.py`.", encoding="utf-8")
    assert checker.check_file(page, tmp_path) == []


def test_prose_in_backticks_is_not_mistaken_for_a_path(tmp_path):
    # The checker must be conservative: flagging ordinary code spans would make
    # it noise, and noise gets disabled.
    checker = check_references_module()
    page = tmp_path / "page.md"
    page.write_text(
        "Use `max_batch_size`, call `pool.allocate()`, run `a/b` and `-flag/x`.",
        encoding="utf-8",
    )
    assert checker.check_file(page, tmp_path) == []


def test_urls_are_not_treated_as_paths(tmp_path):
    checker = check_references_module()
    page = tmp_path / "page.md"
    page.write_text("See `https://example.com/thing.py`.", encoding="utf-8")
    assert checker.check_file(page, tmp_path) == []


def test_globs_are_not_treated_as_paths(tmp_path):
    checker = check_references_module()
    page = tmp_path / "page.md"
    page.write_text("Run over `tests/**/*.py`.", encoding="utf-8")
    assert checker.check_file(page, tmp_path) == []


def test_paths_inside_code_fences_are_ignored(tmp_path):
    # A fenced block is a command to run, not a claim about the tree.
    checker = check_references_module()
    page = tmp_path / "page.md"
    page.write_text("```bash\ncat `src/nowhere.py`\n```\n", encoding="utf-8")
    assert checker.check_file(page, tmp_path) == []


def test_the_repository_documentation_names_only_real_paths():
    result = subprocess.run(
        [sys.executable, "scripts/check_references.py", "."],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout


# --- the API reference -----------------------------------------------------


def test_every_exported_operator_appears_in_the_api_reference():
    # The reference drifted behind the package once already: silu, gelu, swiglu
    # and fused_residual_rmsnorm were added and not documented. A reference that
    # silently omits half the surface is worse than none.
    import cudaforge

    reference = (REPO_ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    missing = [
        name for name in cudaforge.__all__ if not name.startswith("__") and name not in reference
    ]
    assert missing == [], f"undocumented exports: {missing}"


def test_the_api_reference_does_not_name_removed_exports():
    # The other direction: a reference describing something that no longer
    # exists sends a reader looking for it.
    import re

    import cudaforge

    reference = (REPO_ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    documented = set(re.findall(r"^### `(\w+)", reference, re.MULTILINE))
    exported = set(cudaforge.__all__)

    stale = documented - exported
    assert stale == set(), f"documented but not exported: {sorted(stale)}"

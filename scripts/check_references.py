#!/usr/bin/env python3
"""Check that paths named in prose actually exist.

The link checker validates Markdown links. This catches the other half: paths
mentioned in backticks — `cpp/src/dynamic_batcher.cpp`, `scripts/build.sh` —
which render fine and point at nothing once a file is renamed.

Only paths that look like repository paths are checked: they must contain a
slash, end in a known extension or name a directory that exists, and not look
like a URL, a shell command or a glob.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)

# Extensions that identify a repository file rather than prose.
TRACKED_SUFFIXES = {
    ".py",
    ".cpp",
    ".hpp",
    ".cu",
    ".cuh",
    ".md",
    ".sh",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".txt",
    ".cmake",
}


def looks_like_a_path(text: str) -> bool:
    if "/" not in text or text.startswith(("http", "//", "-", "$")):
        return False
    if any(character in text for character in " *?<>|"):
        return False
    return Path(text).suffix in TRACKED_SUFFIXES


def check_file(path: Path, root: Path) -> list[str]:
    body = CODE_FENCE_RE.sub(
        lambda match: "\n" * match.group(0).count("\n"),
        path.read_text(encoding="utf-8"),
    )

    problems: list[str] = []
    for match in CODE_SPAN_RE.finditer(body):
        candidate = match.group(1).strip()
        if not looks_like_a_path(candidate):
            continue
        if not (root / candidate).exists():
            line = body[: match.start()].count("\n") + 1
            problems.append(f"{path}:{line}: `{candidate}` does not exist")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=["."])
    args = parser.parse_args(argv)

    root = Path.cwd().resolve()
    targets: list[Path] = []
    for entry in args.paths:
        path = Path(entry)
        if path.is_dir():
            targets.extend(
                p
                for p in sorted(path.rglob("*.md"))
                if not any(part in {".venv", "node_modules"} for part in p.parts)
                and not any(part.startswith("build") for part in p.parts)
            )
        elif path.suffix == ".md":
            targets.append(path)

    problems = [problem for target in targets for problem in check_file(target, root)]
    for problem in problems:
        print(problem)

    print(f"\nchecked {len(targets)} file(s): {len(problems)} problem(s)", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

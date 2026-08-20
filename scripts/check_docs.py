#!/usr/bin/env python3
"""Validate relative links and anchors across the Markdown files.

Documentation that points at files which no longer exist is worse than
documentation that says nothing: it costs a reader time and then misleads them.
This project leans heavily on cross-references — the concepts index is almost
entirely links — so they need to be checked rather than trusted.

Checks:

  * relative file links resolve to something on disk
  * in-page anchors (`#section`) correspond to a heading in the target
  * no duplicate headings within a file, which would make an anchor ambiguous

External `http(s)` links are not fetched. Network checks make the result depend
on connectivity and on whoever is running it, which is the opposite of what a
repository check should be.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Matches [text](target); the target is captured up to a closing paren.
LINK_RE = re.compile(r"\[(?P<text>[^\]]*)\]\((?P<target>[^)\s]+)(?:\s+\"[^\"]*\")?\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.MULTILINE)
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


@dataclass
class Problem:
    source: Path
    line: int
    target: str
    reason: str

    def __str__(self) -> str:
        return f"{self.source}:{self.line}: {self.target} — {self.reason}"


def slugify(heading: str) -> str:
    """Reproduce GitHub's heading-to-anchor transformation.

    Lowercase, strip anything that is not a word character, space or hyphen,
    then replace spaces with hyphens. Inline markdown is stripped first, since
    `## The `foo` case` anchors as `the-foo-case`.
    """
    text = re.sub(r"`([^`]*)`", r"\1", heading)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_]", "", text)
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    # Each whitespace character becomes one hyphen. Collapsing runs would be
    # wrong: GitHub anchors "Kernel A — Reduction" as "kernel-a--reduction",
    # because removing the em dash leaves two spaces and each becomes a hyphen.
    return re.sub(r"\s", "-", text)


def headings_of(path: Path) -> list[str]:
    if not path.is_file() or path.suffix != ".md":
        return []
    # Fences are stripped so a `# comment` inside a shell block is not mistaken
    # for a heading.
    body = CODE_FENCE_RE.sub("", path.read_text(encoding="utf-8"))
    return [slugify(match.group(2)) for match in HEADING_RE.finditer(body)]


def check_file(path: Path, root: Path) -> list[Problem]:
    problems: list[Problem] = []
    raw = path.read_text(encoding="utf-8")
    body = CODE_FENCE_RE.sub(lambda m: "\n" * m.group(0).count("\n"), raw)

    seen = [slug for slug in headings_of(path)]
    duplicates = {slug for slug in seen if seen.count(slug) > 1}
    for slug in sorted(duplicates):
        problems.append(
            Problem(path, 1, f"#{slug}", "duplicate heading makes this anchor ambiguous")
        )

    for match in LINK_RE.finditer(body):
        target = match.group("target")
        line = body[: match.start()].count("\n") + 1

        if target.startswith(("http://", "https://", "mailto:")):
            continue

        anchor = ""
        if "#" in target:
            target, _, anchor = target.partition("#")

        if not target:
            # Same-page anchor.
            if anchor and anchor not in seen:
                problems.append(Problem(path, line, f"#{anchor}", "no such heading"))
            continue

        resolved = (path.parent / target).resolve()
        if not resolved.exists():
            problems.append(Problem(path, line, target, "file does not exist"))
            continue

        try:
            resolved.relative_to(root)
        except ValueError:
            problems.append(Problem(path, line, target, "escapes the repository"))
            continue

        if anchor:
            target_headings = headings_of(resolved)
            if target_headings and anchor not in target_headings:
                problems.append(
                    Problem(path, line, f"{target}#{anchor}", "no such heading in target")
                )

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=["."], help="files or directories")
    args = parser.parse_args(argv)

    root = Path.cwd().resolve()
    targets: list[Path] = []
    for entry in args.paths:
        path = Path(entry)
        if path.is_dir():
            targets.extend(
                p
                for p in sorted(path.rglob("*.md"))
                if not any(part in {".venv", "build", "node_modules"} for part in p.parts)
                # `validation-*` holds run artefacts, not repository
                # documentation: `validate_gpu.sh` writes a report there, and
                # scanning it made the next run of the suite fail on output the
                # previous run had produced.
                and not any(part.startswith(("build-", "validation-")) for part in p.parts)
            )
        elif path.suffix == ".md":
            targets.append(path)

    problems: list[Problem] = []
    for target in targets:
        problems.extend(check_file(target, root))

    for problem in problems:
        print(problem)

    print(f"\nchecked {len(targets)} file(s): {len(problems)} problem(s)", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

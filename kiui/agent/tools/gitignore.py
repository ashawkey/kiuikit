"""Lightweight .gitignore-aware path filtering for glob/grep/ls tools.

Collects ``.gitignore`` patterns from a directory tree and answers
"is this path ignored?" using the ``pathspec`` library (full gitignore
semantics).  The matcher is built once per search root and reused, so
per-path checks are cheap.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pathspec

# Directories we always skip regardless of .gitignore, for speed and sanity.
ALWAYS_SKIP_DIRS = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".tox", "dist", "build", ".mypy_cache", ".pytest_cache",
})

# Patterns for the always-skip dirs, fed to pathspec alongside gitignore rules.
ALWAYS_SKIP_DIRS_PATTERNS = tuple(f"{d}/" for d in ALWAYS_SKIP_DIRS)

# Cap how many .gitignore files we read to bound cost on huge trees.
_MAX_GITIGNORE_FILES = 1000


def _iter_gitignore_files(root: Path) -> Iterable[tuple[Path, list[str]]]:
    """Yield (dir_containing_gitignore, patterns) for each .gitignore under *root*.

    Pruned directories (ALWAYS_SKIP_DIRS) are not descended into.
    """
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        # prune skip dirs in place so os.walk doesn't descend into them
        dirnames[:] = [d for d in dirnames if d not in ALWAYS_SKIP_DIRS]
        if ".gitignore" in filenames:
            gi = Path(dirpath) / ".gitignore"
            try:
                lines = gi.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            patterns = [
                ln for ln in (raw.strip() for raw in lines)
                if ln and not ln.startswith("#")
            ]
            if patterns:
                yield Path(dirpath), patterns
            count += 1
            if count >= _MAX_GITIGNORE_FILES:
                return


def _reparent_pattern(pat: str, prefix: str) -> str:
    """Prefix a nested-.gitignore pattern with its directory scope.

    A pattern is scoped to the directory containing its .gitignore: a
    path-bearing pattern is relative to that directory, while a bare name
    applies at any depth below it.
    """
    if not prefix:
        return pat
    negated = pat.startswith("!")
    if negated:
        pat = pat[1:]
    anchored = pat.startswith("/")
    body = pat.lstrip("/")
    if anchored or ("/" in body.rstrip("/")):
        new = prefix + body
    else:
        new = prefix + "**/" + body
    return ("!" if negated else "") + new


class GitignoreMatcher:
    """gitignore matcher backed by the ``pathspec`` library."""

    def __init__(self, root: Path):
        self._root = root
        combined: list[str] = list(ALWAYS_SKIP_DIRS_PATTERNS)
        for base, patterns in _iter_gitignore_files(root):
            rel = base.relative_to(root)
            prefix = "" if rel == Path(".") else rel.as_posix() + "/"
            for pat in patterns:
                combined.append(_reparent_pattern(pat, prefix))
        self._spec = pathspec.PathSpec.from_lines("gitwildmatch", combined)

    def is_ignored(self, abs_path: Path, is_dir: bool) -> bool:
        try:
            rel = abs_path.relative_to(self._root).as_posix()
        except ValueError:
            return False
        if is_dir:
            rel += "/"
        return self._spec.match_file(rel)


def build_gitignore_matcher(root: Path) -> GitignoreMatcher | None:
    """Return a :class:`GitignoreMatcher` for *root*, or ``None`` when *root*
    is not a directory."""
    root = Path(root)
    if not root.is_dir():
        return None
    return GitignoreMatcher(root.resolve())

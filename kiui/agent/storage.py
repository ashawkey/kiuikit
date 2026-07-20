"""Inspect and clean project-local kia storage."""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from kiui.agent.utils import KIA_DIR_NAME

CLEANABLE_ENTRIES = ("sessions", "tool-results", "processes", "history")


@dataclass(frozen=True)
class StorageEntry:
    name: str
    path: Path
    size: int
    is_dir: bool


def kia_storage_dir(cwd: str | Path | None = None) -> Path:
    base = Path(cwd) if cwd is not None else Path.cwd()
    return base / KIA_DIR_NAME


def allocated_size(path: Path) -> int:
    """Return allocated bytes without following symbolic links."""
    stat = path.stat(follow_symlinks=False)
    size = getattr(stat, "st_blocks", 0) * 512 or stat.st_size
    if not path.is_dir() or path.is_symlink():
        return size

    with os.scandir(path) as children:
        return size + sum(allocated_size(Path(child.path)) for child in children)


def storage_entries(cwd: str | Path | None = None) -> list[StorageEntry]:
    root = kia_storage_dir(cwd)
    if not root.exists():
        return []
    return [
        StorageEntry(
            name=path.name,
            path=path,
            size=allocated_size(path),
            is_dir=path.is_dir() and not path.is_symlink(),
        )
        for path in sorted(root.iterdir(), key=lambda path: path.name)
    ]


def cleanable_entries(cwd: str | Path | None = None) -> list[StorageEntry]:
    return [entry for entry in storage_entries(cwd) if entry.name in CLEANABLE_ENTRIES]


def clean_storage(cwd: str | Path | None = None) -> int:
    """Delete generated storage and return its measured size in bytes."""
    entries = cleanable_entries(cwd)
    for entry in entries:
        if entry.is_dir:
            shutil.rmtree(entry.path)
        else:
            entry.path.unlink()
    return sum(entry.size for entry in entries)


def format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")

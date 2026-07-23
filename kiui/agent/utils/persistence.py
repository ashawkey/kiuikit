"""Durable local persistence helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_immutable(path: Path, data: bytes) -> None:
    """Durably create a content-addressed object if it is not already present."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, staging_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    staging = Path(staging_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(staging, path)
        except FileExistsError:
            pass
        if os.name == "posix":
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    finally:
        staging.unlink(missing_ok=True)


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Durably append complete JSONL records with one process-level write lock held by the caller."""
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def truncate_torn_jsonl_tail(path: Path) -> None:
    """Discard an incomplete final JSONL record before the next append."""
    if not path.exists():
        return
    with path.open("r+b") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if size == 0:
            return
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return
        f.seek(0)
        raw = f.read()
        last_newline = raw.rfind(b"\n")
        f.truncate(last_newline + 1)
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL, tolerating only a torn final record."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines()
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index == len(lines) - 1 and not raw.endswith(b"\n"):
                break
            raise ValueError(f"Corrupted JSONL record {index + 1}: {path}") from None
        if not isinstance(record, dict):
            raise ValueError(f"JSONL record {index + 1} is not an object: {path}")
        records.append(record)
    return records

"""Tool-result artifact persistence and cleanup."""

import errno
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from kiui.agent.utils import get_kia_dir


def read_tool_result_text(
    result: dict[str, Any], formatted: str, max_chars: int | None = None
) -> str:
    """Read the producer capture when available; otherwise use formatted text.

    When *max_chars* is set and the capture is larger, only a head and tail
    slice is returned. Compaction never surfaces the middle of a large capture,
    so this bounds the work (and memory) of scanning it while the full output
    stays on disk under the persisted artifact path.
    """
    producer_path = result.get("_artifact_path")
    if not producer_path:
        return formatted
    path = Path(producer_path)
    if max_chars is None or path.stat().st_size <= max_chars:
        return path.read_text(encoding="utf-8")

    head_budget = max_chars // 2
    tail_budget = max_chars - head_budget
    with path.open("rb") as f:
        head = f.read(head_budget).decode("utf-8", errors="ignore")
        f.seek(-tail_budget, os.SEEK_END)
        tail = f.read().decode("utf-8", errors="ignore")
    return f"{head}\n[... compaction input truncated; full output on disk ...]\n{tail}"


def discard_tool_result_artifact(result: dict[str, Any]) -> OSError | None:
    """Remove a producer capture, returning any recoverable cleanup error."""
    producer_path = result.get("_artifact_path")
    if producer_path:
        try:
            Path(producer_path).unlink(missing_ok=True)
        except OSError as e:
            return e
    return None


def persist_tool_result_artifact(
    tool_name: str,
    text: str,
    result: dict[str, Any],
    tool_call_id: str,
    work_dir: str | None,
    session_id: str | None,
    round_id: int,
) -> str:
    """Move a producer capture into managed storage, or save formatted text."""
    path = _artifact_path(
        tool_name, tool_call_id, work_dir, session_id, round_id
    )
    producer_path = result.get("_artifact_path")
    if not producer_path:
        _save_text(path, text)
        return _relative_path(path, work_dir)

    source = Path(producer_path)
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        try:
            os.replace(source, path)
        except OSError as e:
            if e.errno != errno.EXDEV:
                raise
            with source.open("rb") as src, staging.open("xb") as dst:
                shutil.copyfileobj(src, dst)
            staging.chmod(0o600)
            os.replace(staging, path)
            source.unlink()
        path.chmod(0o600)
        return _relative_path(path, work_dir)
    except OSError as error:
        _cleanup_after_failure(error, staging, path, source)
        raise


def _artifact_path(
    tool_name: str,
    tool_call_id: str,
    work_dir: str | None,
    session_id: str | None,
    round_id: int,
) -> Path:
    if session_id is None:
        raise RuntimeError("Tool artifact created before session initialization")

    base = Path(work_dir) if work_dir else Path.cwd()
    artifact_dir = get_kia_dir(base) / "tool-results" / session_id
    artifact_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    artifact_dir.chmod(0o700)

    call_id = re.sub(r"[^A-Za-z0-9_.-]", "_", tool_call_id)[:100]
    tool = re.sub(r"[^A-Za-z0-9_.-]", "_", tool_name)[:80]
    path = artifact_dir / f"r{round_id}-{call_id}-{tool}.txt"
    if path.exists():
        path = artifact_dir / f"r{round_id}-{uuid.uuid4().hex}-{tool}.txt"
    return path


def _save_text(path: Path, text: str) -> None:
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        fd = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(staging, path)
    except OSError as error:
        _cleanup_after_failure(error, staging, path)
        raise


def _relative_path(path: Path, work_dir: str | None) -> str:
    base = Path(work_dir) if work_dir else Path.cwd()
    return str(path.relative_to(base))


def _cleanup_after_failure(error: OSError, *paths: Path) -> None:
    cleanup_errors: list[OSError] = []
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as cleanup_error:
            cleanup_errors.append(cleanup_error)
    if cleanup_errors:
        raise OSError(
            f"{error}; artifact cleanup also failed: {cleanup_errors[0]}"
        ) from error

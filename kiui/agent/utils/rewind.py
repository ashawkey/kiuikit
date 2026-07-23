"""Branch-aware file change tracking for session rewind."""

from __future__ import annotations

import os
import shutil
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kiui.agent.session_store import SessionStore
from kiui.agent.ui import AgentConsole


@dataclass
class ChangeRecord:
    """One reversible filesystem operation."""

    round_id: int
    path: str
    op: str
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None


class ChangeTracker:
    """Record pending changes and move code state through a revision DAG."""

    def __init__(
        self,
        session_id: str,
        work_dir: str | Path,
        console: AgentConsole,
        store: SessionStore,
        code_revision_id: str | None = None,
    ):
        self.session_id = session_id
        self.work_dir = Path(work_dir).resolve()
        self.console = console
        self.store = store
        self.code_revision_id = code_revision_id
        self._pending: list[ChangeRecord] = []

    @property
    def pending_changes(self) -> list[dict]:
        return [asdict(record) for record in self._pending]

    def mark_committed(self, code_revision_id: str | None) -> None:
        self.code_revision_id = code_revision_id
        self._pending.clear()

    def flush(self) -> None:
        """Session saves own persistence; pending changes intentionally stay in memory."""

    def close(self) -> None:
        pass

    def _record_path(self, path: str | Path) -> tuple[Path, str]:
        candidate = Path(path)
        abs_path = candidate if candidate.is_absolute() else self.work_dir / candidate
        abs_path = Path(os.path.abspath(abs_path))
        try:
            stored = str(abs_path.relative_to(self.work_dir))
        except ValueError:
            stored = str(abs_path)
        return abs_path, stored

    def _absolute(self, stored: str) -> Path:
        path = Path(stored)
        return path if path.is_absolute() else self.work_dir / path

    def track_write(self, round_id: int, path: str, content: str = ""):
        abs_path, stored_path = self._record_path(path)
        before = self.store.store_path(abs_path) if abs_path.exists() or abs_path.is_symlink() else None
        mode = stat.S_IMODE(abs_path.stat().st_mode) if abs_path.exists() else 0o644
        after = self.store.store_bytes(content.encode("utf-8"), mode=mode)
        self._pending.append(ChangeRecord(round_id, stored_path, "write", before, after))

    def track_edit_result(
        self,
        round_id: int,
        path: str,
        original_content: str | None,
        new_content: str,
    ):
        abs_path, stored_path = self._record_path(path)
        mode = stat.S_IMODE(abs_path.stat().st_mode)
        before = (
            self.store.store_bytes(original_content.encode("utf-8"), mode=mode)
            if original_content is not None else None
        )
        after = self.store.store_bytes(new_content.encode("utf-8"), mode=mode)
        self._pending.append(ChangeRecord(round_id, stored_path, "edit", before, after))

    def track_remove(self, round_id: int, path: str):
        abs_path, stored_path = self._record_path(path)
        before = self.store.store_path(abs_path)
        self._pending.append(ChangeRecord(round_id, stored_path, "remove", before, None))

    def checkout_code(self, target_id: str | None) -> int:
        """Move code from the current revision to *target_id* through their LCA."""
        if self._pending:
            raise RuntimeError("Save the session before checking out a code revision")
        if target_id == self.code_revision_id:
            return 0

        current_chain = self._ancestor_chain(self.code_revision_id)
        target_chain = self._ancestor_chain(target_id)
        target_set = set(target_chain)
        lca = next((revision for revision in current_chain if revision in target_set), None)

        changed = 0
        cursor = self.code_revision_id
        while cursor != lca:
            revision = self.store.code_revisions[cursor]
            for raw in reversed(revision["changes"]):
                changed += self._apply(ChangeRecord(**raw), forward=False)
            cursor = revision["parentId"]

        forward_ids: list[str] = []
        cursor = target_id
        while cursor != lca:
            forward_ids.append(cursor)
            cursor = self.store.code_revisions[cursor]["parentId"]
        for revision_id in reversed(forward_ids):
            for raw in self.store.code_revisions[revision_id]["changes"]:
                changed += self._apply(ChangeRecord(**raw), forward=True)

        self.code_revision_id = target_id
        return changed

    def _ancestor_chain(self, revision_id: str | None) -> list[str | None]:
        chain: list[str | None] = []
        while True:
            chain.append(revision_id)
            if revision_id is None:
                return chain
            revision_id = self.store.code_revisions[revision_id]["parentId"]

    def _apply(self, record: ChangeRecord, *, forward: bool) -> int:
        path = self._absolute(record.path)
        descriptor = record.after if forward else record.before
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()
        if descriptor is not None:
            self.store.restore_object(descriptor, path)
        return 1

    def round_stats(self, round_id: int) -> tuple[int, int, int]:
        """Return file and line statistics for pending changes in a round."""
        files: set[str] = set()
        added = removed = 0
        for record in self._pending:
            if record.round_id != round_id:
                continue
            files.add(record.path)
            old_lines = self._line_count(record.before)
            new_lines = self._line_count(record.after)
            added += max(0, new_lines - old_lines)
            removed += max(0, old_lines - new_lines)
        return len(files), added, removed

    def _line_count(self, descriptor: dict[str, Any] | None) -> int:
        if descriptor is None or descriptor["kind"] != "file":
            return 0
        try:
            return len(self.store._read_object(descriptor).decode("utf-8").splitlines())
        except UnicodeDecodeError:
            return 0

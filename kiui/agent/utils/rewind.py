"""Branch-aware file change tracking for session rewind."""

from __future__ import annotations

import os
import shutil
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kiui.agent.session_store import PathDelta, SessionStore
from kiui.agent.ui import AgentConsole


@dataclass
class ChangeRecord:
    """One reversible filesystem operation."""

    round_id: int
    path: str
    op: str
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None


@dataclass
class CheckoutPlan:
    """What moving code to another revision would do, before doing it."""

    target_id: str | None
    steps: list[tuple[dict[str, Any], bool]]
    deltas: list[PathDelta]
    dirty: tuple[str, ...]
    added: int
    removed: int

    def __bool__(self) -> bool:
        return bool(self.deltas)

    @property
    def files(self) -> int:
        return len(self.deltas)


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

    def plan_checkout(self, target_id: str | None) -> CheckoutPlan:
        """Describe moving code to *target_id* without touching the filesystem."""
        if self._pending:
            raise RuntimeError("Save the session before checking out a code revision")

        steps = self.store.code_walk(self.code_revision_id, target_id)
        deltas = self.store.code_delta(self.code_revision_id, target_id)
        _, added, removed = self.store.delta_stats(deltas)
        dirty = tuple(delta.path for delta in deltas if self._is_dirty(delta))
        return CheckoutPlan(target_id, steps, deltas, dirty, added, removed)

    def apply_plan(self, plan: CheckoutPlan) -> int:
        """Execute a plan built by :meth:`plan_checkout`."""
        if self._pending:
            raise RuntimeError("Save the session before checking out a code revision")

        changed = 0
        for record, forward in plan.steps:
            changed += self._apply(ChangeRecord(**record), forward=forward)
        self.code_revision_id = plan.target_id
        return changed

    def checkout_code(self, target_id: str | None) -> int:
        """Move code from the current revision to *target_id* through their LCA."""
        return self.apply_plan(self.plan_checkout(target_id))

    def _is_dirty(self, delta: PathDelta) -> bool:
        """Whether the working tree at *delta.path* no longer matches the record.

        A dirty path means a rewind would overwrite edits made outside the agent
        (or by a tool that does not track changes), so it is worth warning about.
        Text files are compared decoded: tools write them in text mode, so the
        line endings on disk are platform-dependent while the stored bytes are not.
        """
        path = self._absolute(delta.path)
        if delta.before is None:
            # The path is expected to be absent; anything there would be clobbered.
            return path.exists() or path.is_symlink()
        if not path.exists() and not path.is_symlink():
            return True
        recorded = self.store.read_text(delta.before)
        if recorded is not None and path.is_file() and not path.is_symlink():
            try:
                return path.read_text(encoding="utf-8") != recorded
            except (UnicodeDecodeError, OSError):
                return True
        return self.store.hash_path(path) != delta.before["id"]

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

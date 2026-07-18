"""Change tracking and rollback for the kiui agent.

Tracks individual file changes made by tool calls so they can be
inverted to roll back to a previous round — without directory
snapshots or fragile ignore rules.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from kiui.agent.context import get_role, get_text
from kiui.agent.ui import AgentConsole


@dataclass
class ChangeRecord:
    """One file modification that can be undone."""
    round_id: int
    path: str          # work-dir-relative when possible, otherwise absolute
    op: str            # "write" | "edit" | "remove"
    # For write / edit: original content (None = file didn't exist)
    original_content: str | None = None
    # For write / edit: content after the change (for diff stats)
    new_content: str | None = None
    # For remove: where the backed-up file lives (relative to rewind base)
    backup_path: str | None = None
    was_dir: bool = False


class ChangeTracker:
    """Records file changes and can invert them to roll back."""

    def __init__(self, session_id: str, work_dir: str | Path, console: AgentConsole):
        self.session_id = session_id
        self.work_dir = Path(work_dir).resolve()
        self.console = console
        self._base = Path.home() / ".kia" / "rewind" / session_id
        self._base.mkdir(parents=True, exist_ok=True)
        self._log: list[ChangeRecord] = []

        # Load existing log from disk (resumed sessions)
        log_path = self._base / "change_log.json"
        if log_path.exists():
            try:
                raw = json.loads(log_path.read_text(encoding="utf-8"))
                self._log = [ChangeRecord(**r) for r in raw]
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_log(self):
        raw = [{
            "round_id": r.round_id,
            "path": r.path,
            "op": r.op,
            "original_content": r.original_content,
            "new_content": r.new_content,
            "backup_path": r.backup_path,
            "was_dir": r.was_dir,
        } for r in self._log]
        (self._base / "change_log.json").write_text(
            json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def flush(self) -> None:
        """Persist the in-memory log so on-disk backups remain reachable."""
        self._save_log()

    def close(self) -> None:
        """Finish using this tracker without deleting resumable rewind data."""
        self.flush()

    # ------------------------------------------------------------------
    # Track individual file operations  (call BEFORE the tool acts)
    # ------------------------------------------------------------------

    def _record_path(self, path: str | Path) -> tuple[Path, str]:
        candidate = Path(path)
        abs_path = candidate if candidate.is_absolute() else self.work_dir / candidate
        abs_path = Path(os.path.abspath(abs_path))
        try:
            stored = str(abs_path.relative_to(self.work_dir))
        except ValueError:
            stored = str(abs_path)
        return abs_path, stored

    def track_write(self, round_id: int, path: str, content: str = ""):
        """Capture *path* content before it is overwritten / created."""
        abs_path, stored_path = self._record_path(path)
        original = None
        if abs_path.exists() and abs_path.is_file():
            try:
                original = abs_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                original = None
        self._log.append(ChangeRecord(
            round_id=round_id, path=stored_path, op="write",
            original_content=original,
            new_content=content,
        ))

    def track_edit_result(self, round_id: int, path: str, original_content: str | None, new_content: str):
        """Record an edit with explicit before/after content, so rollback is exact.

        Callers pass the actual applied result (which may differ from a naive
        ``str.replace`` due to tolerant matching or multi-edit).
        """
        _, stored_path = self._record_path(path)
        self._log.append(ChangeRecord(
            round_id=round_id, path=stored_path, op="edit",
            original_content=original_content,
            new_content=new_content,
        ))


    def track_remove(self, round_id: int, path: str):
        """Back up *path* before it is deleted."""
        abs_path, stored_path = self._record_path(path)
        backup_path = None
        was_dir = False

        if abs_path.exists():
            backup_dir = self._base / f"backup_{len(self._log)}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_file = backup_dir / abs_path.name

            if abs_path.is_dir():
                was_dir = True
                try:
                    shutil.copytree(abs_path, str(backup_file))
                    backup_path = str(backup_file.relative_to(self._base))
                except OSError:
                    return
            elif abs_path.is_file():
                try:
                    shutil.copy2(abs_path, backup_file)
                    backup_path = str(backup_file.relative_to(self._base))
                except OSError:
                    return

        self._log.append(ChangeRecord(
            round_id=round_id, path=stored_path, op="remove",
            backup_path=backup_path, was_dir=was_dir,
        ))

    # ------------------------------------------------------------------
    # Rollback – code
    # ------------------------------------------------------------------

    def rollback_code(self, target_round: int) -> int:
        """Undo all file changes from rounds > *target_round*.

        Applies inverse operations in reverse chronological order.
        Returns the number of files restored/changed.
        """
        undone = 0

        for rec in reversed(self._log):
            if rec.round_id <= target_round:
                continue

            if rec.op == "write":
                abs_path = self.work_dir / rec.path
                if rec.original_content is None:
                    # File was created → delete it
                    try:
                        if abs_path.exists():
                            abs_path.unlink()
                            undone += 1
                    except OSError:
                        pass
                else:
                    # File was overwritten → restore original
                    try:
                        abs_path.parent.mkdir(parents=True, exist_ok=True)
                        abs_path.write_text(rec.original_content, encoding="utf-8")
                        undone += 1
                    except OSError:
                        pass

            elif rec.op == "edit":
                abs_path = self.work_dir / rec.path
                if rec.original_content is not None:
                    try:
                        abs_path.parent.mkdir(parents=True, exist_ok=True)
                        abs_path.write_text(rec.original_content, encoding="utf-8")
                        undone += 1
                    except OSError:
                        pass

            elif rec.op == "remove":
                abs_path = self.work_dir / rec.path
                if rec.backup_path:
                    backup = self._base / rec.backup_path
                    if backup.exists():
                        try:
                            if rec.was_dir:
                                if abs_path.exists():
                                    shutil.rmtree(abs_path, ignore_errors=True)
                                shutil.copytree(backup, abs_path)
                            else:
                                abs_path.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(backup, abs_path)
                            undone += 1
                        except OSError:
                            pass
                else:
                    # No backup → file didn't exist → delete it
                    try:
                        if abs_path.exists():
                            if abs_path.is_dir():
                                shutil.rmtree(abs_path, ignore_errors=True)
                            else:
                                abs_path.unlink()
                            undone += 1
                    except OSError:
                        pass

        # Remove rolled-back entries from log
        self._log = [r for r in self._log if r.round_id <= target_round]
        self._save_log()
        return undone

    # ------------------------------------------------------------------
    # Conversation rollback
    # ------------------------------------------------------------------

    def rollback_conversation(self, messages: list, target_round: int) -> tuple[list, int]:
        """Truncate *messages* to only keep rounds ≤ *target_round*."""
        round_num = 0
        cut_after = len(messages)

        for i, msg in enumerate(messages):
            if get_role(msg) == "user":
                round_num += 1
                if round_num == target_round + 1:
                    cut_after = i
                    break

        return messages[:cut_after], max(0, min(target_round, round_num))

    # ------------------------------------------------------------------
    # Picker helpers
    # ------------------------------------------------------------------

    def _compute_round_stats(self, round_id: int) -> tuple[int, int, int]:
        """Return (files_changed, lines_added, lines_removed) for a round."""
        files: set[str] = set()
        added = 0
        removed = 0

        for rec in self._log:
            if rec.round_id != round_id:
                continue

            if rec.op in ("write", "edit"):
                if rec.path:
                    files.add(rec.path)
                old_lines = rec.original_content.count("\n") if rec.original_content else 0
                new_lines = rec.new_content.count("\n") if rec.new_content else 0
                if rec.original_content and not rec.original_content.endswith("\n"):
                    old_lines += 1
                if rec.new_content and not rec.new_content.endswith("\n"):
                    new_lines += 1
                if rec.op == "write" and rec.original_content is None:
                    # New file — all lines are added
                    added += new_lines
                else:
                    added += max(0, new_lines - old_lines)
                    removed += max(0, old_lines - new_lines)

            elif rec.op == "remove":
                if rec.path:
                    files.add(rec.path)
                old_lines = rec.original_content.count("\n") if rec.original_content else 0
                if rec.original_content and not rec.original_content.endswith("\n"):
                    old_lines += 1
                removed += old_lines

        return len(files), added, removed

    def build_round_list(self, messages: list) -> list[dict]:
        """Return [{round: int, preview: str, files: int, added: int, removed: int}] for each round."""
        rounds = []
        round_num = 0
        for msg in messages:
            if get_role(msg) == "user":
                round_num += 1
                text = get_text(msg).replace("\n", " ").strip()
                preview = text[:80] + ("..." if len(text) > 80 else "")
                files, added, removed = self._compute_round_stats(round_num)
                rounds.append({
                    "round": round_num,
                    "preview": preview,
                    "files": files,
                    "added": added,
                    "removed": removed,
                })
        return rounds

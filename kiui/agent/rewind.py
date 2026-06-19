"""Change tracking and rollback for the kiui agent.

Tracks individual file changes made by tool calls so they can be
inverted to roll back to a previous round — without directory
snapshots or fragile ignore rules.
"""

from __future__ import annotations

import hashlib
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
    path: str          # relative to work_dir
    op: str            # "write" | "edit" | "remove" | "exec_batch"
    # For write / edit: original content (None = file didn't exist)
    original_content: str | None = None
    # For remove: where the backed-up file lives (relative to rewind base)
    backup_path: str | None = None
    was_dir: bool = False
    # For exec_batch: path to dir containing originals of changed files
    exec_batch_dir: str | None = None


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
            "backup_path": r.backup_path,
            "was_dir": r.was_dir,
            "exec_batch_dir": r.exec_batch_dir,
        } for r in self._log]
        (self._base / "change_log.json").write_text(
            json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Track individual file operations  (call BEFORE the tool acts)
    # ------------------------------------------------------------------

    def track_write(self, round_id: int, path: str):
        """Capture *path* content before it is overwritten / created."""
        abs_path = self.work_dir / path
        original = None
        if abs_path.exists() and abs_path.is_file():
            try:
                original = abs_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                original = None
        self._log.append(ChangeRecord(
            round_id=round_id, path=path, op="write",
            original_content=original,
        ))

    def track_edit(self, round_id: int, path: str):
        """Capture *path* content before an edit is applied."""
        abs_path = self.work_dir / path
        original = None
        try:
            original = abs_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            original = None
        self._log.append(ChangeRecord(
            round_id=round_id, path=path, op="edit",
            original_content=original,
        ))

    def track_remove(self, round_id: int, path: str):
        """Back up *path* before it is deleted."""
        abs_path = self.work_dir / path
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
            round_id=round_id, path=path, op="remove",
            backup_path=backup_path, was_dir=was_dir,
        ))

    # ------------------------------------------------------------------
    # Track exec_command  (pre / post pair)
    # ------------------------------------------------------------------

    def pre_exec_backup(self) -> str:
        """Walk work_dir (skip only .kia), copy every file to a temp dir.

        Returns the batch directory path relative to ``self._base``.
        Call ``post_exec_compare(batch_rel, round_id)`` after the command.
        """
        batch_dir = self._base / f"exec_pre_{len(self._log)}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for root, dirs, files in os.walk(self.work_dir, topdown=True):
            dirs[:] = [d for d in dirs if d != ".kia"]
            for fname in files:
                fpath = Path(root) / fname
                try:
                    rel = fpath.relative_to(self.work_dir)
                except ValueError:
                    continue
                dest = batch_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(fpath, dest)
                    count += 1
                except OSError:
                    pass

        batch_rel = batch_dir.relative_to(self._base).as_posix()
        self.console.system(f"Pre-exec backup: {count} files saved")
        return batch_rel

    def post_exec_compare(self, batch_rel: str, round_id: int):
        """Compare current work_dir against pre-exec backup.

        Keeps originals only for files that changed or were deleted.
        Stores a ChangeRecord if anything changed.
        """
        batch_dir = self._base / batch_rel
        if not batch_dir.exists():
            return

        has_changes = False

        # 1. Find files in backup that changed or were deleted
        for backup_root, dirs, files in os.walk(batch_dir, topdown=True):
            for fname in files:
                backup_path = Path(backup_root) / fname
                try:
                    rel = backup_path.relative_to(batch_dir)
                except ValueError:
                    continue
                current_path = self.work_dir / rel

                if not current_path.exists():
                    # File deleted — keep backup
                    has_changes = True
                elif current_path.is_file():
                    try:
                        current_h = hashlib.sha256(current_path.read_bytes()).hexdigest()
                        backup_h = hashlib.sha256(backup_path.read_bytes()).hexdigest()
                        if current_h == backup_h:
                            backup_path.unlink()  # unchanged → discard
                        else:
                            has_changes = True
                    except OSError:
                        has_changes = True  # keep on error
                else:
                    backup_path.unlink()  # now a dir → discard

            # Remove empty dirs in backup
            for d in dirs:
                dpath = Path(backup_root) / d
                try:
                    if not any(dpath.iterdir()):
                        dpath.rmdir()
                except OSError:
                    pass

        # 2. Find files that were created (exist now, not in backup)
        #    → add empty marker file to backup
        for root, dirs, files in os.walk(self.work_dir, topdown=True):
            dirs[:] = [d for d in dirs if d != ".kia"]
            for fname in files:
                fpath = Path(root) / fname
                try:
                    rel = fpath.relative_to(self.work_dir)
                except ValueError:
                    continue
                marker = batch_dir / rel
                if not marker.exists():
                    marker.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        marker.write_bytes(b"")  # empty = "didn't exist before"
                        has_changes = True
                    except OSError:
                        pass

        if has_changes:
            self._log.append(ChangeRecord(
                round_id=round_id, path="", op="exec_batch",
                exec_batch_dir=batch_rel,
            ))
            self._save_log()
        else:
            shutil.rmtree(batch_dir, ignore_errors=True)

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

            elif rec.op == "exec_batch":
                if rec.exec_batch_dir:
                    batch = self._base / rec.exec_batch_dir
                    if batch.exists():
                        undone += self._invert_exec_batch(batch)

        # Remove rolled-back entries from log
        self._log = [r for r in self._log if r.round_id <= target_round]
        self._save_log()
        return undone

    def _invert_exec_batch(self, batch_dir: Path) -> int:
        """Restore files from an exec_batch backup."""
        restored = 0
        for backup_root, dirs, files in os.walk(batch_dir, topdown=True):
            for fname in files:
                backup_path = Path(backup_root) / fname
                try:
                    rel = backup_path.relative_to(batch_dir)
                except ValueError:
                    continue
                current_path = self.work_dir / rel
                try:
                    content = backup_path.read_bytes()
                except OSError:
                    continue

                if content == b"":
                    # Empty marker → file was created by exec_command → delete
                    try:
                        if current_path.exists():
                            current_path.unlink()
                            restored += 1
                    except OSError:
                        pass
                else:
                    # Restore original content
                    try:
                        current_path.parent.mkdir(parents=True, exist_ok=True)
                        current_path.write_bytes(content)
                        restored += 1
                    except OSError:
                        pass

        shutil.rmtree(batch_dir, ignore_errors=True)
        return restored

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

    def build_round_list(self, messages: list) -> list[dict]:
        """Return [{round: int, preview: str}] for each round."""
        rounds = []
        round_num = 0
        for msg in messages:
            if get_role(msg) == "user":
                round_num += 1
                text = get_text(msg).replace("\n", " ").strip()
                preview = text[:80] + ("..." if len(text) > 80 else "")
                rounds.append({"round": round_num, "preview": preview})
        return rounds

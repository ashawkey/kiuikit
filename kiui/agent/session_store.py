"""Append-only session history with branchable conversation and code revisions."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import stat
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from filelock import FileLock

from kiui.agent.context import get_role, get_text
from kiui.agent.utils.persistence import (
    append_jsonl,
    read_jsonl,
    truncate_torn_jsonl_tail,
    write_immutable,
)


def _message_id(message: dict[str, Any]) -> str:
    raw = json.dumps(message, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _object_id(kind: str, data: bytes) -> str:
    return hashlib.sha256(kind.encode("ascii") + b"\0" + data).hexdigest()


def _object_key(descriptor: dict[str, Any] | None) -> tuple[str, int] | None:
    """Identity of a stored object: content plus permissions, or ``None`` if absent."""
    if descriptor is None:
        return None
    return descriptor["id"], descriptor["mode"]


# Above this many lines, line-by-line matching costs more than the stat is worth.
DIFF_LINE_LIMIT = 20000


@dataclass(frozen=True)
class PathDelta:
    """Net effect on one path of moving between two code revisions."""

    path: str
    before: dict[str, Any] | None
    after: dict[str, Any] | None

    @property
    def op(self) -> str:
        if self.before is None:
            return "create"
        if self.after is None:
            return "delete"
        return "modify"


class SessionStore:
    """Materialize one session from an append-only revision DAG."""

    def __init__(self, sessions_dir: Path, session_id: str):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", session_id):
            raise ValueError(f"Invalid session ID: {session_id!r}")
        self.session_id = session_id
        self.path = sessions_dir / session_id
        self.history_path = self.path / "history.jsonl"
        self.objects_path = self.path / "objects"
        self.lock = FileLock(str(self.path / ".lock"))
        self.messages: dict[str, dict[str, Any]] = {}
        self.revisions: dict[str, dict[str, Any]] = {}
        self.code_revisions: dict[str, dict[str, Any]] = {}
        self.revision_order: list[str] = []
        self.head_id: str | None = None
        self._text_stats: dict[tuple[tuple[str, int] | None, ...], tuple[int, int]] = {}
        self._load()

    def _load(self) -> None:
        for record in read_jsonl(self.history_path):
            kind = record.get("type")
            if kind == "message":
                self.messages[record["id"]] = record["message"]
            elif kind == "code_revision":
                self.code_revisions[record["id"]] = record
            elif kind == "revision":
                self.revisions[record["id"]] = record
                self.revision_order.append(record["id"])
            elif kind == "head":
                target = record["revisionId"]
                if target not in self.revisions:
                    raise ValueError(f"Session head references unknown revision: {target}")
                self.head_id = target
            else:
                raise ValueError(f"Unknown session history record type: {kind!r}")

    @property
    def exists(self) -> bool:
        return self.head_id is not None

    def store_bytes(self, data: bytes, *, mode: int = 0o644) -> dict[str, Any]:
        return self._store_object("file", data, mode)

    def store_path(self, path: Path) -> dict[str, Any]:
        """Store a file tree as immutable content-addressed objects."""
        info = path.lstat()
        mode = stat.S_IMODE(info.st_mode)
        if path.is_symlink():
            return self._store_object("symlink", os.readlink(path).encode("utf-8"), mode)
        if path.is_file():
            return self._store_object("file", path.read_bytes(), mode)
        if path.is_dir():
            entries = [
                {"name": child.name, "object": self.store_path(child)}
                for child in sorted(path.iterdir(), key=lambda child: child.name)
            ]
            manifest = json.dumps(entries, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            return self._store_object("tree", manifest, mode)
        raise ValueError(f"Cannot store special file: {path}")

    def restore_object(self, descriptor: dict[str, Any], path: Path) -> None:
        """Restore and verify one content-addressed object tree."""
        kind = descriptor["kind"]
        data = self._read_object(descriptor)
        mode = descriptor["mode"]
        if kind == "file":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            path.chmod(mode)
        elif kind == "symlink":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(data.decode("utf-8"))
        elif kind == "tree":
            path.mkdir(parents=True)
            entries = json.loads(data)
            for entry in entries:
                name = entry["name"]
                if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
                    raise ValueError(f"Invalid name in session tree object: {name!r}")
                self.restore_object(entry["object"], path / name)
            path.chmod(mode)
        else:
            raise ValueError(f"Unknown object kind: {kind!r}")

    def _store_object(self, kind: str, data: bytes, mode: int) -> dict[str, Any]:
        object_id = _object_id(kind, data)
        write_immutable(self._object_path(object_id), data)
        return {"id": object_id, "kind": kind, "mode": mode}

    def _read_object(self, descriptor: dict[str, Any]) -> bytes:
        object_id = descriptor["id"]
        kind = descriptor["kind"]
        data = self._object_path(object_id).read_bytes()
        if _object_id(kind, data) != object_id:
            raise ValueError(f"Corrupted session object: {object_id}")
        return data

    def read_text(self, descriptor: dict[str, Any] | None) -> str | None:
        """Return a stored file's text, or ``None`` for a missing or binary object.

        Line endings are normalized: tools write files in text mode, so the same
        content is stored with CRLF when captured from disk and LF when captured
        from a tool argument. Restoring uses the raw bytes and is unaffected.
        """
        if descriptor is None or descriptor["kind"] != "file":
            return None
        try:
            return self._read_object(descriptor).decode("utf-8").replace("\r\n", "\n")
        except UnicodeDecodeError:
            return None

    def hash_path(self, path: Path) -> str | None:
        """Content-addressed ID of *path* as it is on disk, without storing it.

        Returns ``None`` for a missing path or one that cannot be stored, so
        callers comparing against a recorded descriptor treat it as a mismatch.
        """
        if not path.exists() and not path.is_symlink():
            return None
        if path.is_symlink():
            return _object_id("symlink", os.readlink(path).encode("utf-8"))
        if path.is_file():
            return _object_id("file", path.read_bytes())
        if path.is_dir():
            entries = []
            for child in sorted(path.iterdir(), key=lambda child: child.name):
                child_id = self.hash_path(child)
                if child_id is None:
                    return None
                info = child.lstat()
                kind = (
                    "symlink" if child.is_symlink()
                    else "tree" if child.is_dir()
                    else "file"
                )
                entries.append({
                    "name": child.name,
                    "object": {"id": child_id, "kind": kind, "mode": stat.S_IMODE(info.st_mode)},
                })
            manifest = json.dumps(entries, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            return _object_id("tree", manifest)
        return None

    def _object_path(self, object_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", object_id):
            raise ValueError(f"Invalid session object ID: {object_id!r}")
        return self.objects_path / object_id[:2] / object_id[2:]

    def summary(self) -> dict[str, Any]:
        """Materialize only the metadata needed by session pickers."""
        data = self.materialize()
        messages = data["messages"]
        return {
            "messages": messages,
            "message_count": len(messages),
            "round_id": data.get("round_id", 0),
            "model": data.get("model", "?"),
        }

    def materialize(self, revision_id: str | None = None) -> dict[str, Any]:
        revision_id = revision_id or self.head_id
        if revision_id is None or revision_id not in self.revisions:
            raise ValueError(f"Unknown session revision: {revision_id}")
        revision = self.revisions[revision_id]
        data = dict(revision["state"])
        try:
            data["messages"] = [self.messages[mid] for mid in revision["messageIds"]]
        except KeyError as e:
            raise ValueError(f"Session revision references unknown message: {e.args[0]}") from None
        data["revision_id"] = revision_id
        data["code_revision_id"] = revision.get("codeRevisionId")
        return data

    def commit(
        self,
        data: dict[str, Any],
        *,
        parent_id: str | None,
        code_parent_id: str | None,
        changes: list[dict[str, Any]],
        reason: str,
    ) -> tuple[str, str | None, bool]:
        """Append a conversation revision and optional code revision."""
        messages = data["messages"]
        state = {key: value for key, value in data.items() if key != "messages"}
        message_ids = [_message_id(message) for message in messages]

        self.path.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self._load_fresh()
            truncate_torn_jsonl_tail(self.history_path)
            records: list[dict[str, Any]] = []
            for mid, message in zip(message_ids, messages):
                if mid not in self.messages:
                    record = {"type": "message", "id": mid, "message": message}
                    records.append(record)
                    self.messages[mid] = message

            code_revision_id = code_parent_id
            if changes:
                code_revision_id = uuid.uuid4().hex
                code_record = {
                    "type": "code_revision",
                    "id": code_revision_id,
                    "parentId": code_parent_id,
                    "changes": changes,
                    "createdAt": time.time(),
                }
                records.append(code_record)
                self.code_revisions[code_revision_id] = code_record

            if parent_id is not None:
                current = self.revisions[parent_id]
                unchanged = (
                    current["messageIds"] == message_ids
                    and current["state"] == state
                    and current.get("codeRevisionId") == code_revision_id
                )
                if unchanged:
                    self.head_id = parent_id
                    return parent_id, code_revision_id, False

            revision_id = uuid.uuid4().hex
            revision = {
                "type": "revision",
                "id": revision_id,
                "parentId": parent_id,
                "codeRevisionId": code_revision_id,
                "messageIds": message_ids,
                "state": state,
                "reason": reason,
                "createdAt": time.time(),
            }
            head = {
                "type": "head",
                "revisionId": revision_id,
                "previousId": self.head_id,
                "reason": reason,
                "createdAt": time.time(),
            }
            records.extend((revision, head))
            append_jsonl(self.history_path, records)
            self.revisions[revision_id] = revision
            self.revision_order.append(revision_id)
            self.head_id = revision_id
            return revision_id, code_revision_id, True

    def checkout(self, revision_id: str, *, reason: str = "rewind") -> dict[str, Any]:
        """Move the durable head to an existing revision without deleting descendants."""
        self.path.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self._load_fresh()
            truncate_torn_jsonl_tail(self.history_path)
            if revision_id not in self.revisions:
                raise ValueError(f"Unknown session revision: {revision_id}")
            head = {
                "type": "head",
                "revisionId": revision_id,
                "previousId": self.head_id,
                "reason": reason,
                "createdAt": time.time(),
            }
            append_jsonl(self.history_path, [head])
            self.head_id = revision_id
            return self.materialize(revision_id)

    def resolve_revision(self, value: str) -> str:
        """Resolve a full/prefix revision ID, or the newest revision for a round number."""
        if value.isdigit():
            round_id = int(value)
            matches = [
                rid for rid in self.revision_order
                if self.revisions[rid]["state"].get("round_id") == round_id
            ]
        else:
            matches = [rid for rid in self.revision_order if rid.startswith(value)]
        if not matches:
            raise ValueError(f"No revision matches {value!r}")
        if not value.isdigit() and len(matches) > 1:
            raise ValueError(f"Revision prefix {value!r} is ambiguous")
        return matches[-1]

    def code_walk(self, from_id: str | None, to_id: str | None) -> list[tuple[dict[str, Any], bool]]:
        """Ordered ``(change record, forward)`` steps moving code from *from_id* to *to_id*.

        The walk first undoes the current branch down to the lowest common
        ancestor, then replays the target branch forward. Applying and previewing
        a checkout both consume this sequence, so they cannot disagree.
        """
        if from_id == to_id:
            return []

        current_chain = self._code_ancestors(from_id)
        target_chain = self._code_ancestors(to_id)
        target_set = set(target_chain)
        lca = next((revision for revision in current_chain if revision in target_set), None)

        steps: list[tuple[dict[str, Any], bool]] = []
        cursor = from_id
        while cursor != lca:
            revision = self.code_revisions[cursor]
            steps.extend((raw, False) for raw in reversed(revision["changes"]))
            cursor = revision["parentId"]

        forward_ids: list[str] = []
        cursor = to_id
        while cursor != lca:
            forward_ids.append(cursor)
            cursor = self.code_revisions[cursor]["parentId"]
        for revision_id in reversed(forward_ids):
            steps.extend((raw, True) for raw in self.code_revisions[revision_id]["changes"])
        return steps

    def _code_ancestors(self, revision_id: str | None) -> list[str | None]:
        chain: list[str | None] = []
        while True:
            chain.append(revision_id)
            if revision_id is None:
                return chain
            revision_id = self.code_revisions[revision_id]["parentId"]

    def code_delta(self, from_id: str | None, to_id: str | None) -> list[PathDelta]:
        """Net per-path effect of moving code from *from_id* to *to_id*.

        Paths that a walk touches but ends up leaving byte-identical are dropped,
        so an empty result means the move really would not touch the filesystem.
        """
        source: dict[str, dict[str, Any] | None] = {}
        target: dict[str, dict[str, Any] | None] = {}
        for record, forward in self.code_walk(from_id, to_id):
            path = record["path"]
            before, after = record["before"], record["after"]
            step_source, step_target = (before, after) if forward else (after, before)
            source.setdefault(path, step_source)
            target[path] = step_target
        return [
            PathDelta(path, source[path], target[path])
            for path in sorted(target)
            if _object_key(source[path]) != _object_key(target[path])
        ]

    def revision_changes(self, revision_id: str) -> list[PathDelta]:
        """Net file changes a revision introduced relative to its parent revision."""
        revision = self.revisions[revision_id]
        parent_id = revision["parentId"]
        parent_code = self.revisions[parent_id].get("codeRevisionId") if parent_id else None
        return self.code_delta(parent_code, revision.get("codeRevisionId"))

    def delta_stats(self, deltas: list[PathDelta]) -> tuple[int, int, int]:
        """Return ``(files, lines added, lines removed)`` for *deltas*."""
        added = removed = 0
        for delta in deltas:
            delta_added, delta_removed = self.text_stats(delta.before, delta.after)
            added += delta_added
            removed += delta_removed
        return len(deltas), added, removed

    def text_stats(
        self, before: dict[str, Any] | None, after: dict[str, Any] | None
    ) -> tuple[int, int]:
        """Return ``(added, removed)`` line counts between two stored objects.

        Binary files and directory trees have no line diff and count as zero.
        """
        key = (_object_key(before), _object_key(after))
        cached = self._text_stats.get(key)
        if cached is not None:
            return cached
        old_lines = (self.read_text(before) or "").splitlines()
        new_lines = (self.read_text(after) or "").splitlines()
        if max(len(old_lines), len(new_lines)) > DIFF_LINE_LIMIT:
            # Line-by-line matching is quadratic; fall back to net size for huge files.
            stats = (
                max(0, len(new_lines) - len(old_lines)),
                max(0, len(old_lines) - len(new_lines)),
            )
        else:
            added = removed = 0
            matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag in ("replace", "delete"):
                    removed += i2 - i1
                if tag in ("replace", "insert"):
                    added += j2 - j1
            stats = (added, removed)
        self._text_stats[key] = stats
        return stats

    def text_hunks(
        self, before: dict[str, Any] | None, after: dict[str, Any] | None
    ) -> list[tuple[int, str, str]]:
        """Return ``(start line, removed text, added text)`` for each changed hunk.

        Empty when either side is binary or a directory tree.
        """
        old_text = self.read_text(before)
        new_text = self.read_text(after)
        if (old_text is None and before is not None) or (new_text is None and after is not None):
            return []
        old_lines = (old_text or "").splitlines()
        new_lines = (new_text or "").splitlines()
        if max(len(old_lines), len(new_lines)) > DIFF_LINE_LIMIT:
            # Too large to align cheaply; hand back one whole-file hunk instead.
            return [(1, old_text or "", new_text or "")]
        matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
        return [
            (i1 + 1, "\n".join(old_lines[i1:i2]), "\n".join(new_lines[j1:j2]))
            for tag, i1, i2, j1, j2 in matcher.get_opcodes()
            if tag != "equal"
        ]

    def revision_prompt(self, revision_id: str) -> str:
        """Last user message in a revision, collapsed to a single line."""
        for message_id in reversed(self.revisions[revision_id]["messageIds"]):
            message = self.messages.get(message_id)
            if message is None or get_role(message) != "user":
                continue
            return " ".join(get_text(message).split())
        return ""

    def revision_ancestors(self, revision_id: str) -> list[str]:
        """Revision IDs from *revision_id* back to the root, newest first."""
        chain: list[str] = []
        cursor: str | None = revision_id
        while cursor is not None:
            chain.append(cursor)
            cursor = self.revisions[cursor]["parentId"]
        return chain

    def candidates(self) -> list[dict[str, Any]]:
        """Return all revisions newest-first for branch-aware rewind selection.

        Carries the conversation and file-change context the picker needs. Line
        counts are deliberately left out: they read stored objects, while
        everything here comes from records already in memory.
        """
        result = []
        for rid in reversed(self.revision_order):
            revision = self.revisions[rid]
            state = revision["state"]
            parent_id = revision["parentId"]
            parent = self.revisions.get(parent_id) if parent_id else None
            result.append({
                "id": rid,
                "parent_id": parent_id,
                "round_id": state.get("round_id", 0),
                "reason": revision["reason"],
                "created_at": revision["createdAt"],
                "messages": len(revision["messageIds"]),
                "new_messages": len(revision["messageIds"]) - (len(parent["messageIds"]) if parent else 0),
                "prompt": self.revision_prompt(rid),
                "files": len(self.revision_changes(rid)),
                "code_revision_id": revision.get("codeRevisionId"),
                "current": rid == self.head_id,
            })
        return result

    def _load_fresh(self) -> None:
        self.messages.clear()
        self.revisions.clear()
        self.code_revisions.clear()
        self.revision_order.clear()
        self.head_id = None
        self._load()

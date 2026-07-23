"""Append-only session history with branchable conversation and code revisions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
import uuid
from pathlib import Path
from typing import Any

from filelock import FileLock

from kiui.agent.utils.persistence import (
    append_jsonl,
    read_jsonl,
    truncate_torn_jsonl_tail,
    write_immutable,
)


def _message_id(message: dict[str, Any]) -> str:
    raw = json.dumps(message, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


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
        object_id = hashlib.sha256(kind.encode("ascii") + b"\0" + data).hexdigest()
        write_immutable(self._object_path(object_id), data)
        return {"id": object_id, "kind": kind, "mode": mode}

    def _read_object(self, descriptor: dict[str, Any]) -> bytes:
        object_id = descriptor["id"]
        kind = descriptor["kind"]
        data = self._object_path(object_id).read_bytes()
        actual = hashlib.sha256(kind.encode("ascii") + b"\0" + data).hexdigest()
        if actual != object_id:
            raise ValueError(f"Corrupted session object: {object_id}")
        return data

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

    def candidates(self) -> list[dict[str, Any]]:
        """Return all revisions newest-first for branch-aware rewind selection."""
        result = []
        for rid in reversed(self.revision_order):
            revision = self.revisions[rid]
            state = revision["state"]
            result.append({
                "id": rid,
                "parent_id": revision["parentId"],
                "round_id": state.get("round_id", 0),
                "reason": revision["reason"],
                "created_at": revision["createdAt"],
                "messages": len(revision["messageIds"]),
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

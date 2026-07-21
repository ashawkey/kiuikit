"""File reading, writing, editing, listing, and removal tools."""

import base64
import os
import shutil
from pathlib import Path
from typing import Any

from .constants import MAX_READ_BYTES, MAX_READ_LINES, SKIP_DIRS as _SKIP_DIRS


_IMAGE_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def _image_mime_type(data: bytes) -> str | None:
    for signature, mime_type in _IMAGE_SIGNATURES:
        if data.startswith(signature):
            return mime_type
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _human_size(n: int) -> str:
    """Format a byte count compactly (e.g. 1.2K, 3.4M)."""
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024 or unit == "T":
            if unit == "B":
                return f"{n}{unit}"
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}T"


def _normalize_ws(text: str) -> str:
    """Collapse each line's trailing whitespace and unify line endings.

    Used only to *locate* a match when the exact bytes differ by trailing
    whitespace or CRLF/LF — the actual replacement always operates on the
    real file bytes so nothing else is disturbed.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines)


def find_match(content: str, old_text: str) -> list[tuple[int, int]]:
    """Locate *old_text* within LF-normalized *content*, tolerating whitespace.

    Returns a list of ``(start, end)`` character offsets for every match.
    An empty list means no match. *content* must already be LF-normalized;
    *old_text* is normalized internally.

    Strategy (first that finds any match wins):
      1. Exact substring match.
      2. Whitespace-tolerant, line-aligned match (ignore per-line trailing
         whitespace), mapped back to exact offsets.
    """
    if not old_text:
        return []

    old_norm_lf = old_text.replace("\r\n", "\n").replace("\r", "\n")

    # Strategy 1: exact
    spans: list[tuple[int, int]] = []
    step = len(old_norm_lf)
    idx = content.find(old_norm_lf)
    while idx != -1:
        spans.append((idx, idx + step))
        idx = content.find(old_norm_lf, idx + step)
    if spans:
        return spans

    # Strategy 2: whitespace-tolerant, line-aligned
    norm_old = _normalize_ws(old_text)
    if not norm_old.strip():
        return []

    lines = content.split("\n")
    offsets: list[int] = []
    pos = 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1  # +1 for the '\n'
    norm_lines = [ln.rstrip() for ln in lines]
    old_lines = norm_old.split("\n")
    n = len(old_lines)

    i = 0
    while i <= len(norm_lines) - n:
        if norm_lines[i:i + n] == old_lines:
            last = i + n - 1
            spans.append((offsets[i], offsets[last] + len(lines[last])))
            i += n  # non-overlapping
        else:
            i += 1
    return spans


def apply_edit(content: str, old_text: str, new_text: str, replace_all: bool) -> tuple[str, int, int, str | None]:
    """Apply a single edit to *content*, returning LF-normalized content.

    Returns ``(new_content, count, first_line_num, error)``:
      - error is None on success; otherwise a human-readable message.
      - count is how many occurrences were replaced.
      - first_line_num is the 1-based line number of the first replacement.
    """
    work = content.replace("\r\n", "\n").replace("\r", "\n")
    new_lf = new_text.replace("\r\n", "\n").replace("\r", "\n")

    spans = find_match(work, old_text)
    count = len(spans)
    if count == 0:
        return content, 0, 0, f"Text not found in file: {old_text[:100]}"

    if count > 1 and not replace_all:
        return (
            content, count, 0,
            f"matches {count} locations. Include more surrounding context to make "
            "it unique, or set replace_all=true to replace every occurrence.",
        )

    # Splice replacements at exact offsets (correct even when tolerant matches
    # have differing byte content).
    pieces: list[str] = []
    prev = 0
    for start, end in spans:
        pieces.append(work[prev:start])
        pieces.append(new_lf)
        prev = end
        if not replace_all:
            break
    pieces.append(work[prev:])
    new_content = "".join(pieces)

    line_num = work[:spans[0][0]].count("\n") + 1
    return new_content, (count if replace_all else 1), line_num, None


class FileToolsMixin:
    def _resolve_path(self, path: str | os.PathLike[str]) -> Path:
        """Resolve a caller path lexically against the executor's work directory."""
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = Path(self._work_dir) / candidate
        return Path(os.path.abspath(candidate))

    def _read_file(self, file: str, offset: int | None = None, limit: int | None = None) -> dict[str, Any]:
        """Read file contents with optional offset and limit."""
        self.console.tool(f"read_file {file} (offset={offset}, limit={limit})")

        file_path = self._resolve_path(file)
        if not file_path.exists():
            return {"error": f"File not found: {file}", "success": False}
        if not file_path.is_file():
            return {"error": f"Path is not a file: {file}", "success": False}

        try:
            with open(file_path, encoding="utf-8") as f:
                all_lines = f.readlines()
        except UnicodeDecodeError:
            return {"error": f"Cannot read binary file: {file}", "success": False}

        total_lines = len(all_lines)
        lines = all_lines

        if offset is not None:
            lines = lines[max(0, offset - 1):]

        effective_limit = limit if limit is not None else MAX_READ_LINES
        truncated_by_lines = len(lines) > effective_limit
        if truncated_by_lines:
            lines = lines[:effective_limit]

        content = "".join(lines)

        truncated_by_bytes = False
        if len(content.encode("utf-8")) > MAX_READ_BYTES:
            truncated_by_bytes = True
            encoded = content.encode("utf-8")[:MAX_READ_BYTES]
            content = encoded.decode("utf-8", errors="ignore")
            lines = content.splitlines(keepends=True)

        if truncated_by_lines or truncated_by_bytes:
            content += f"\n[output truncated: {len(lines)} of {total_lines} lines shown. Use offset/limit for more.]"

        return {"content": content, "lines_read": len(lines), "success": True}

    def _read_image(self, file: str) -> dict[str, Any]:
        """Read an image into an OpenAI-compatible data URL."""
        self.console.tool(f"read_image {file}")

        file_path = self._resolve_path(file)
        if not file_path.exists():
            return {"error": f"File not found: {file}", "success": False}
        if not file_path.is_file():
            return {"error": f"Path is not a file: {file}", "success": False}

        data = file_path.read_bytes()
        mime_type = _image_mime_type(data)
        if mime_type is None:
            return {
                "error": f"Unsupported image format: {file}. Use PNG, JPEG, GIF, or WebP.",
                "success": False,
            }

        encoded = base64.b64encode(data).decode("ascii")
        return {
            "message": f"Image loaded: {file} ({mime_type}, {_human_size(len(data))})",
            "image_url": f"data:{mime_type};base64,{encoded}",
            "mime_type": mime_type,
            "size": len(data),
            "success": True,
        }

    def _write_file(self, file: str, content: str) -> dict[str, Any]:
        """Write content to file, creating parent directories."""
        self.console.tool(f"write_file {file}")

        file_path = self._resolve_path(file)
        if self._change_tracker and self._get_round_id:
            self._change_tracker.track_write(self._get_round_id(), str(file_path), content)

        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        return {
            "message": f"Successfully wrote {len(content)} characters to {file}",
            "success": True,
            "diff": {
                "path": file,
                "old_text": "",        # write_file has no "old" — show all as additions
                "new_text": content,
                "line_num": None,
                "count": 1,
            },
        }

    def _edit_file(self, file: str, old_text: str, new_text: str, replace_all: bool = False) -> dict[str, Any]:
        """Make a surgical edit to a file (whitespace-tolerant matching)."""
        self.console.tool(f"edit_file {file}")

        file_path = self._resolve_path(file)
        if not file_path.exists():
            return {"error": f"File not found: {file}", "success": False}
        if not file_path.is_file():
            return {"error": f"Path is not a file: {file}", "success": False}

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return {"error": f"Cannot edit binary file: {file}", "success": False}

        new_content, count, line_num, error = apply_edit(content, old_text, new_text, replace_all)
        if error is not None:
            return {"error": f"{error} (in {file})", "success": False}

        original = content.replace("\r\n", "\n").replace("\r", "\n")
        if self._change_tracker and self._get_round_id:
            self._change_tracker.track_edit_result(self._get_round_id(), str(file_path), original, new_content)

        file_path.write_text(new_content, encoding="utf-8")

        diff_msg = (
            f"Replaced {count} occurrence(s):\n- {old_text}\n+ {new_text}"
            if replace_all
            else f"Line {line_num}:\n- {old_text}\n+ {new_text}"
        )
        return {
            "message": f"Successfully edited {file}\n{diff_msg}",
            "success": True,
            "diff": {
                "path": file,
                "old_text": old_text,
                "new_text": new_text,
                "line_num": None if replace_all else line_num,
                "count": count,
            },
        }

    def _multi_edit(self, file: str, edits: list | None = None) -> dict[str, Any]:
        """Apply an ordered sequence of edits to one file, all-or-nothing."""
        if not edits:
            return {"error": "No edits provided.", "success": False}

        self.console.tool(f"multi_edit {file} ({len(edits)} edits)")

        file_path = self._resolve_path(file)
        if not file_path.exists():
            return {"error": f"File not found: {file}", "success": False}
        if not file_path.is_file():
            return {"error": f"Path is not a file: {file}", "success": False}

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return {"error": f"Cannot edit binary file: {file}", "success": False}

        original = content.replace("\r\n", "\n").replace("\r", "\n")
        working = original
        total_replacements = 0
        diffs: list[dict[str, Any]] = []

        for i, edit in enumerate(edits):
            if not isinstance(edit, dict):
                return {
                    "error": (
                        f"Edit #{i + 1} must be an object with 'old_text'/'new_text' "
                        f"keys, got {type(edit).__name__}. Pass 'edits' as a list of "
                        "objects, e.g. [{\"old_text\": \"a\", \"new_text\": \"b\"}]."
                    ),
                    "success": False,
                }
            old_text = edit.get("old_text", "")
            new_text = edit.get("new_text", "")
            replace_all = bool(edit.get("replace_all", False))
            if not old_text:
                return {"error": f"Edit #{i + 1}: old_text is required.", "success": False}

            working, count, line_num, error = apply_edit(working, old_text, new_text, replace_all)
            if error is not None:
                # All-or-nothing: nothing has been written to disk yet.
                return {"error": f"Edit #{i + 1} failed: {error} (in {file})", "success": False}
            total_replacements += count
            diffs.append({
                "path": file,
                "old_text": old_text,
                "new_text": new_text,
                "line_num": None if replace_all else line_num,
                "count": count,
            })

        if self._change_tracker and self._get_round_id:
            self._change_tracker.track_edit_result(self._get_round_id(), str(file_path), original, working)

        file_path.write_text(working, encoding="utf-8")

        return {
            "message": (
                f"Successfully applied {len(edits)} edit(s) to {file} "
                f"({total_replacements} replacement(s))."
            ),
            "success": True,
            "diffs": diffs,
        }

    def _ls(self, path: str | None = None, all: bool = False) -> dict[str, Any]:
        """List a directory's immediate contents (gitignore-aware)."""
        self.console.tool(f"ls {path or '.'}" + (" (all)" if all else ""))

        base = self._resolve_path(path or ".")
        if not base.exists():
            return {"error": f"Path not found: {path}", "success": False}
        if not base.is_dir():
            return {"error": f"Not a directory: {path}", "success": False}

        matcher = None
        if not all:
            from kiui.agent.tools.gitignore import build_gitignore_matcher
            matcher = build_gitignore_matcher(base)

        try:
            entries = sorted(base.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError as e:
            return {"error": f"Cannot list directory: {e}", "success": False}

        dirs: list[str] = []
        files: list[str] = []
        skipped = 0
        for entry in entries:
            is_dir = entry.is_dir()
            if not all:
                if entry.name.startswith("."):
                    skipped += 1
                    continue
                if is_dir and entry.name in _SKIP_DIRS:
                    skipped += 1
                    continue
                if matcher is not None and matcher.is_ignored(entry.resolve(), is_dir):
                    skipped += 1
                    continue
            if is_dir:
                dirs.append(f"{entry.name}/")
            else:
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = 0
                files.append(f"{entry.name}  ({_human_size(size)})")

        lines = dirs + files
        content = "\n".join(lines) if lines else "(empty)"
        if skipped and not all:
            content += f"\n[{skipped} hidden/ignored entr{'y' if skipped == 1 else 'ies'} omitted; use all=true to show]"

        return {
            "content": content,
            "count": len(lines),
            "success": True,
        }

    def _remove_file(self, file: str) -> dict[str, Any]:
        """Remove a file or directory."""
        self.console.tool(f"remove_file {file}")

        target = self._resolve_path(file)
        if not target.exists():
            return {"error": f"Path not found: {file}", "success": False}

        if self._change_tracker and self._get_round_id:
            self._change_tracker.track_remove(self._get_round_id(), str(target))

        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()

        return {"message": f"Removed {file}", "success": True}

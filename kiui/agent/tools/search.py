"""Gitignore-aware glob and grep tools."""

import json
import locale
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pathspec

from .constants import (
    GLOB_TIMEOUT_SECONDS,
    MAX_GLOB_RESULTS,
    MAX_GREP_MATCHES,
    MAX_TOOL_OUTPUT_CHARS,
    SKIP_DIRS as _SKIP_DIRS,
)
from .process_util import _terminate_process


def _decode_bytes(b: bytes | None) -> str:
    """Decode subprocess output bytes using the preferred locale encoding."""
    if not b:
        return ""
    encoding = locale.getpreferredencoding()
    try:
        return b.decode(encoding)
    except UnicodeDecodeError:
        return b.decode("utf-8", errors="replace")


def _build_search_result(matches: list, truncated: bool, guidance: str) -> dict[str, Any]:
    """Build a search result, dropping trailing matches until it fits."""
    kept = matches.copy()
    reason = "item cap" if truncated else None

    while True:
        result: dict[str, Any] = {
            "matches": kept,
            "count": len(kept),
            "truncated": truncated,
            "success": True,
        }
        if truncated:
            result["truncation_reason"] = reason
            result["guidance"] = guidance
        if len(json.dumps(result, indent=2)) <= MAX_TOOL_OUTPUT_CHARS:
            return result
        kept.pop()
        truncated = True
        reason = "character cap"


class SearchToolsMixin:
    def _glob_files(self, pattern: str, base_dir: str | None = None, recursive: bool = True, include_ignored: bool = False) -> dict[str, Any]:
        """Find files matching a glob pattern (gitignore-aware)."""
        self.console.tool(f"glob_files {pattern} (recursive={recursive})")

        base = self._resolve_path(base_dir or ".")
        if not base.is_dir():
            return {"error": f"Not a directory: {base}", "success": False}
        if not recursive:
            if "**" in pattern:
                return {
                    "error": "recursive=False but pattern contains '**'. "
                    "Use recursive=True for recursive globbing, or remove '**' for a flat search.",
                    "success": False,
                }
            return self._glob_flat(pattern, base, include_ignored)
        if not shutil.which("rg"):
            return {
                "error": "glob_files requires ripgrep (rg). Install ripgrep and retry.",
                "success": False,
            }
        return self._glob_ripgrep(pattern, base, include_ignored)

    def _glob_spec(self, pattern: str) -> pathspec.PathSpec:
        if pattern.startswith("!"):
            pattern = "\\" + pattern
        return pathspec.PathSpec.from_lines("gitwildmatch", [pattern])

    def _glob_result(self, matches: list[str], truncated: bool) -> dict[str, Any]:
        matches.sort()
        return _build_search_result(
            matches,
            truncated,
            "Use a narrower glob pattern or base_dir.",
        )

    def _glob_flat(self, pattern: str, base: Path, include_ignored: bool) -> dict[str, Any]:
        ignore_spec = None
        if not include_ignored:
            try:
                lines = (base / ".gitignore").read_text(
                    encoding="utf-8", errors="ignore"
                ).splitlines()
            except OSError:
                lines = []
            ignore_spec = pathspec.PathSpec.from_lines("gitwildmatch", lines)

        spec = self._glob_spec(pattern)
        matches = []
        truncated = False
        deadline = time.monotonic() + GLOB_TIMEOUT_SECONDS
        try:
            entries = os.scandir(base)
        except OSError as e:
            return {"error": f"Cannot scan directory: {e}", "success": False}
        with entries:
            for entry in entries:
                if self.cancellation is not None and self.cancellation.cancelled:
                    return {"error": "Glob search interrupted.", "success": False, "interrupted": True}
                if time.monotonic() >= deadline:
                    return {
                        "error": f"Glob search timed out after {GLOB_TIMEOUT_SECONDS}s. Use a narrower pattern or base_dir.",
                        "success": False,
                    }
                if entry.name in _SKIP_DIRS:
                    continue
                try:
                    if not entry.is_file(follow_symlinks=True):
                        continue
                except OSError:
                    continue
                if ignore_spec is not None and ignore_spec.match_file(entry.name):
                    continue
                if not spec.match_file(entry.name):
                    continue
                if len(matches) >= MAX_GLOB_RESULTS:
                    truncated = True
                    break
                matches.append(entry.name)
        return self._glob_result(matches, truncated)

    def _glob_ripgrep(self, pattern: str, base: Path, include_ignored: bool) -> dict[str, Any]:
        cmd = ["rg", "--files", "--null", "--hidden", "--no-ignore-parent"]
        if include_ignored:
            cmd.append("--no-ignore")
        for directory in _SKIP_DIRS:
            cmd.extend(["--glob", f"!**/{directory}/**"])
        match_spec = self._glob_spec(pattern)

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=base,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=os.name != "nt",
                creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
            )
        except FileNotFoundError:
            return {
                "error": "glob_files requires ripgrep (rg). Install ripgrep and retry.",
                "success": False,
            }

        output: queue.Queue[str | None] = queue.Queue(maxsize=256)
        stop_reader = threading.Event()

        def emit(value: str | None) -> bool:
            while not stop_reader.is_set():
                try:
                    output.put(value, timeout=0.05)
                    return True
                except queue.Full:
                    pass
            return False

        def read_stdout() -> None:
            pending = b""
            while not stop_reader.is_set():
                chunk = proc.stdout.read(8192)
                if not chunk:
                    break
                fields = (pending + chunk).split(b"\0")
                pending = fields.pop()
                for field in fields:
                    if not emit(os.fsdecode(field)):
                        return
            if pending:
                emit(os.fsdecode(pending))
            emit(None)

        stderr_parts: list[bytes] = []
        stderr_size = 0

        def read_stderr() -> None:
            nonlocal stderr_size
            while chunk := proc.stderr.read(8192):
                if stderr_size < MAX_TOOL_OUTPUT_CHARS:
                    stderr_parts.append(chunk)
                    stderr_size += len(chunk)

        reader = threading.Thread(target=read_stdout, daemon=True)
        error_reader = threading.Thread(target=read_stderr, daemon=True)
        reader.start()
        error_reader.start()
        matches = []
        truncated = False
        error = None
        deadline = time.monotonic() + GLOB_TIMEOUT_SECONDS
        try:
            while True:
                if self.cancellation is not None and self.cancellation.cancelled:
                    error = {
                        "error": "Glob search interrupted.",
                        "success": False,
                        "interrupted": True,
                    }
                    break
                if time.monotonic() >= deadline:
                    error = {
                        "error": f"Glob search timed out after {GLOB_TIMEOUT_SECONDS}s. Use a narrower pattern or base_dir.",
                        "success": False,
                    }
                    break
                try:
                    line = output.get(timeout=0.05)
                except queue.Empty:
                    continue
                if line is None:
                    break
                if not match_spec.match_file(line):
                    continue
                if len(matches) >= MAX_GLOB_RESULTS:
                    truncated = True
                    break
                matches.append(line)
        finally:
            stop_reader.set()
            if proc.poll() is None:
                _terminate_process(proc)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            reader.join(timeout=1)
            error_reader.join(timeout=1)
            stderr = _decode_bytes(b"".join(stderr_parts))

        if error is not None:
            return error
        if not truncated and proc.returncode not in (0, 1):
            return {"error": f"ripgrep error: {stderr.strip()}", "success": False}
        return self._glob_result(matches, truncated)

    def _grep_files(
        self,
        pattern: str,
        path: str | None = None,
        file_glob: str | None = None,
        case_insensitive: bool = False,
    ) -> dict[str, Any]:
        """Search file contents using a regex pattern."""
        # Build an informative log line with all search parameters
        parts = [f"grep_files {pattern}"]
        if path:
            parts.append(f"path={path}")
        if file_glob:
            parts.append(f"glob={file_glob}")
        if case_insensitive:
            parts.append("(case-insensitive)")
        self.console.tool(" ".join(parts))

        base = self._resolve_path(path or ".")

        if not shutil.which("rg"):
            return {
                "error": "grep_files requires ripgrep (rg). Install ripgrep and retry.",
                "success": False,
            }
        return self._grep_ripgrep(pattern, base, file_glob, case_insensitive)

    def _grep_ripgrep(
        self,
        pattern: str,
        base: Path,
        file_glob: str | None,
        case_insensitive: bool,
    ) -> dict[str, Any]:
        """Search using ripgrep with structured JSON output.

        JSON mode avoids fragile ``file:line:text`` delimiter parsing: paths,
        line numbers and match text are explicit fields, so filenames containing
        ``:`` and matched text containing ``:`` are handled correctly, as is the
        single-file case (where rg's text output omits the filename prefix).
        """
        cmd = [
            "rg", "--json",
            "--path-separator", "/",
        ]
        if case_insensitive:
            cmd.append("--ignore-case")
        if file_glob:
            cmd.extend(["--glob", file_glob])
        cmd.extend(["--", pattern, str(base)])

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return {"error": "ripgrep timed out after 30s", "success": False}
        except FileNotFoundError:
            return {
                "error": "grep_files requires ripgrep (rg). Install ripgrep and retry.",
                "success": False,
            }

        if result.returncode not in (0, 1):  # 1 = no matches
            stderr = _decode_bytes(result.stderr).strip()
            return {"error": f"ripgrep error: {stderr}", "success": False}

        matches = []
        truncated = False
        for raw_line in _decode_bytes(result.stdout).splitlines():
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "match":
                continue
            data = event["data"]
            # ripgrep emits {"text": ...} for valid UTF-8 or {"bytes": <base64>}
            # for non-UTF-8 paths/lines; skip the latter rather than guess.
            path_obj = data["path"]
            if "text" not in path_obj:
                continue
            # Stop once a match beyond the cap is seen, so an exact-cap result
            # (with no further matches) is not falsely flagged as truncated.
            if len(matches) >= MAX_GREP_MATCHES:
                truncated = True
                break
            file_str = path_obj["text"]
            lines_obj = data["lines"]
            text = lines_obj.get("text", "").rstrip("\n")

            # make path relative to base when base is a directory; when base is
            # a single file, rg reports that file's own path, so keep it as-is.
            if base.is_dir():
                try:
                    rel = str(Path(file_str).relative_to(base))
                except ValueError:
                    rel = file_str
            else:
                rel = file_str
            matches.append({"file": rel, "line": data["line_number"], "text": text[:200]})

        return _build_search_result(
            matches,
            truncated,
            "Use a narrower regex, path, or file_glob.",
        )

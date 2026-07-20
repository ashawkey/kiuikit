"""Gitignore-aware glob and grep tools."""

import json
import locale
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .constants import MAX_GLOB_RESULTS, MAX_GREP_MATCHES, SKIP_DIRS as _SKIP_DIRS


def _decode_bytes(b: bytes | None) -> str:
    """Decode subprocess output bytes using the preferred locale encoding."""
    if not b:
        return ""
    encoding = locale.getpreferredencoding()
    try:
        return b.decode(encoding)
    except UnicodeDecodeError:
        return b.decode("utf-8", errors="replace")


class SearchToolsMixin:
    def _glob_files(self, pattern: str, base_dir: str | None = None, recursive: bool = True, include_ignored: bool = False) -> dict[str, Any]:
        """Find files matching a glob pattern (gitignore-aware)."""
        self.console.tool(f"glob_files {pattern} (recursive={recursive})")

        base = self._resolve_path(base_dir or ".")
        if not base.is_dir():
            return {"error": f"Not a directory: {base}", "success": False}

        matcher = None
        if not include_ignored:
            from kiui.agent.tools.gitignore import build_gitignore_matcher
            matcher = build_gitignore_matcher(base)
        base_resolved = base.resolve()

        if not recursive:
            if "**" in pattern:
                return {
                    "error": "recursive=False but pattern contains '**'. "
                    "Use recursive=True for recursive globbing, or remove '**' for a flat search.",
                    "success": False,
                }
            iterator = base.glob(pattern)
        elif "**" in pattern:
            iterator = base.glob(pattern)
        else:
            iterator = base.rglob(pattern)

        matches = []
        for p in iterator:
            if any(part in _SKIP_DIRS for part in p.parts):
                continue
            if matcher is not None:
                try:
                    is_dir = p.is_dir()
                except OSError:
                    is_dir = False
                if matcher.is_ignored((base_resolved / p.relative_to(base)), is_dir):
                    continue
            matches.append(str(p.relative_to(base)))
            if len(matches) >= MAX_GLOB_RESULTS:
                break

        matches.sort()
        return {
            "matches": matches,
            "count": len(matches),
            "truncated": len(matches) == MAX_GLOB_RESULTS,
            "success": True,
        }

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

        if shutil.which("rg"):
            return self._grep_ripgrep(pattern, base, file_glob, case_insensitive)
        return self._grep_python(pattern, base, file_glob, case_insensitive)

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
            return self._grep_python(pattern, base, file_glob, case_insensitive)

        if result.returncode not in (0, 1):  # 1 = no matches
            stderr = _decode_bytes(result.stderr).strip()
            return {"error": f"ripgrep error: {stderr}", "success": False}

        matches = []
        for raw_line in _decode_bytes(result.stdout).splitlines():
            if len(matches) >= MAX_GREP_MATCHES:
                break
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

        return {
            "matches": matches,
            "count": len(matches),
            "truncated": len(matches) == MAX_GREP_MATCHES,
            "success": True,
        }

    def _grep_python(
        self,
        pattern: str,
        base: Path,
        file_glob: str | None,
        case_insensitive: bool,
    ) -> dict[str, Any]:
        """Pure-Python fallback when ripgrep is not available."""
        flags = re.IGNORECASE if case_insensitive else 0
        try:
            compiled = re.compile(pattern, flags)
        except re.error as e:
            return {"error": f"Invalid regex: {e}", "success": False}

        matcher = None
        base_resolved = base
        if base.is_dir():
            from kiui.agent.tools.gitignore import build_gitignore_matcher
            matcher = build_gitignore_matcher(base)
            base_resolved = base.resolve()

        def _candidate_files():
            if base.is_file():
                yield base
            else:
                glob_pat = file_glob or "*"
                for p in base.rglob(glob_pat):
                    if not p.is_file() or any(part in _SKIP_DIRS for part in p.parts):
                        continue
                    if matcher is not None and matcher.is_ignored((base_resolved / p.relative_to(base)), False):
                        continue
                    yield p

        matches = []
        for file_path in _candidate_files():
            if len(matches) >= MAX_GREP_MATCHES:
                break
            try:
                text = file_path.read_text(encoding="utf-8", errors="strict")
            except (UnicodeDecodeError, OSError):
                continue
            rel = str(file_path.relative_to(base)) if not base.is_file() else str(file_path)
            for lineno, line in enumerate(text.splitlines(), 1):
                if compiled.search(line):
                    matches.append({"file": rel, "line": lineno, "text": line[:200]})
                    if len(matches) >= MAX_GREP_MATCHES:
                        break

        return {
            "matches": matches,
            "count": len(matches),
            "truncated": len(matches) == MAX_GREP_MATCHES,
            "success": True,
        }

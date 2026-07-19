"""Built-in tool definitions and executor for kiui agent."""

import codecs
import ipaddress
import json
import locale
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import socket
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from kiui.agent.ui import AgentConsole
from kiui.agent.interrupt import CancelWatcher
from kiui.agent.io import CancellationToken


def _terminate_process(proc: subprocess.Popen) -> None:
    """Kill *proc* and its child process tree (best-effort)."""
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        else:
            # start_new_session=True makes the child PID its process-group ID;
            # that group can outlive the leader when a command backgrounds work.
            os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass

MAX_READ_LINES = 1000
MAX_READ_BYTES = 24_000
MAX_EXEC_OUTPUT_CHARS = 24_000
MAX_STREAMING_BUFFER_CHARS = 1_000_000
MAX_EXEC_ARTIFACT_BYTES = 100 * 1024 * 1024
EXEC_READER_JOIN_TIMEOUT = 5
MAX_WEB_FETCH_CHARS = 20_000
MAX_WEB_FETCH_BYTES = 2 * 1024 * 1024
MAX_WEB_REDIRECTS = 5
MAX_GLOB_RESULTS = 500
MAX_GREP_MATCHES = 200

_SKIP_DIRS = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".tox", "dist", "build", ".mypy_cache", ".pytest_cache",
})
_IPV6_TRANSITION_NETWORKS = (
    ipaddress.ip_network("64:ff9b::/96"),
    ipaddress.ip_network("64:ff9b:1::/48"),
    ipaddress.ip_network("2001::/32"),  # Teredo
    ipaddress.ip_network("2002::/16"),  # 6to4
)


def _resolve_public_addresses(host: str, port: int) -> tuple[str, ...]:
    """Resolve *host* and reject any result that is not globally routable."""
    addresses: list[str] = []
    for family, _, _, _, sockaddr in socket.getaddrinfo(
        host, port, type=socket.SOCK_STREAM
    ):
        if family not in (socket.AF_INET, socket.AF_INET6):
            continue
        address = ipaddress.ip_address(sockaddr[0].split("%", 1)[0])
        if (
            not address.is_global
            or address.is_multicast
            or (
                isinstance(address, ipaddress.IPv6Address)
                and any(address in network for network in _IPV6_TRANSITION_NETWORKS)
            )
        ):
            raise ValueError(f"destination resolves to non-public address {address}")
        value = str(address)
        if value not in addresses:
            addresses.append(value)
    if not addresses:
        raise ValueError("destination has no public IP address")
    return tuple(addresses)


def _decode_bytes(b: bytes | None) -> str:
    """Decode subprocess output bytes using the preferred locale encoding."""
    if not b:
        return ""
    encoding = locale.getpreferredencoding()
    try:
        return b.decode(encoding)
    except UnicodeDecodeError:
        return b.decode("utf-8", errors="replace")


def _human_size(n: int) -> str:
    """Format a byte count compactly (e.g. 1.2K, 3.4M)."""
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024 or unit == "T":
            if unit == "B":
                return f"{n}{unit}"
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}T"


# ── Tolerant text matching for edits ─────────────────────────────────────

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



def get_tool_definitions(
    include_subagent: bool = True,
    allowed: set[str] | frozenset | None = None,
) -> list[dict[str, Any]]:
    """Return OpenAI-format tool definitions for the built-in tools.

    Set *include_subagent* to False for sub-agents, which must not be able to
    spawn further sub-agents (keeps spawning a single, sequential level deep).
    Set *allowed* to a persona's tool whitelist to advertise only those tools;
    None advertises all tools.
    """
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file. Defaults to the first 1000 lines. Large results are compacted; use grep_files first and offset/limit for focused reads.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string", "description": "Path to the file to read"},
                        "offset": {"type": "integer", "description": "Line number to start reading from (1-indexed)"},
                        "limit": {"type": "integer", "description": "Maximum number of lines to read"},
                    },
                    "required": ["file"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Create or overwrite a file with content. Creates parent directories automatically.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string", "description": "Path to the file to write"},
                        "content": {"type": "string", "description": "Content to write to the file"},
                    },
                    "required": ["file", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Make a surgical edit to a file by replacing exact text. old_text should match the file content; minor differences in trailing whitespace / line endings are tolerated. By default old_text must resolve to exactly one location; set replace_all=true to replace every occurrence.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string", "description": "Path to the file to edit"},
                        "old_text": {"type": "string", "description": "Exact text to find and replace"},
                        "new_text": {"type": "string", "description": "New text to replace the old text with"},
                        "replace_all": {"type": "boolean", "description": "Replace all occurrences instead of requiring exactly one (default: false)"},
                    },
                    "required": ["file", "old_text", "new_text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "multi_edit",
                "description": (
                    "Apply a sequence of edits to a SINGLE file in one atomic operation. "
                    "Edits are applied in order, each to the result of the previous one. "
                    "If any edit fails to match, NO changes are written (all-or-nothing). "
                    "Prefer this over multiple edit_file calls when changing several places in one file. "
                    "Same tolerant matching as edit_file."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string", "description": "Path to the file to edit"},
                        "edits": {
                            "type": "array",
                            "description": "Ordered list of edits to apply",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "old_text": {"type": "string", "description": "Exact text to find and replace"},
                                    "new_text": {"type": "string", "description": "Replacement text"},
                                    "replace_all": {"type": "boolean", "description": "Replace every occurrence (default: false)"},
                                },
                                "required": ["old_text", "new_text"],
                            },
                        },
                    },
                    "required": ["file", "edits"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ls",
                "description": (
                    "List the contents of a directory (non-recursive). Shows entry names, "
                    "type (file/dir) and size. Respects .gitignore and skips noise dirs by default. "
                    "Large listings are compacted; use glob_files with a narrow pattern when possible. "
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory to list (default: working directory)"},
                        "all": {"type": "boolean", "description": "Include hidden and gitignored entries (default: false)"},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "exec_command",
                "description": (
                    "Run a shell command and stream its output. Returns stdout, stderr, and exit code; "
                    "large output is compacted with full capture available in an artifact."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to execute"},
                        "cwd": {"type": "string", "description": "Working directory (optional)"},
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "glob_files",
                "description": (
                    "Find files matching a glob pattern. Preferred over exec_command with find. "
                    "Searches recursively by default. Respects .gitignore and skips noise dirs. "
                    "Set recursive=false to match only in the immediate directory. Max 500 results."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py', 'src/**/*.ts', or '*.py'"},
                        "base_dir": {"type": "string", "description": "Directory to search in (default: current directory)"},
                        "recursive": {"type": "boolean", "description": "Search subdirectories recursively (default: true)"},
                        "include_ignored": {"type": "boolean", "description": "Include .gitignored files (default: false)"},
                    },
                    "required": ["pattern"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "grep_files",
                "description": "Search file contents using a regex pattern. Returns up to 200 matching lines with file path and line number.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Regex pattern to search for"},
                        "path": {"type": "string", "description": "File or directory to search (default: current directory)"},
                        "file_glob": {"type": "string", "description": "Filename filter, e.g. '*.py' or '*.ts'"},
                        "case_insensitive": {"type": "boolean", "description": "Case-insensitive search (default: false)"},
                    },
                    "required": ["pattern"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for real-time information. Returns summarized results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_fetch",
                "description": "Fetch URL content and convert to readable text. Content capped at 20K chars.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch"},
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "remove_file",
                "description": "Remove a file or directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string", "description": "Path to the file or directory to remove"},
                    },
                    "required": ["file"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "spawn_subagent",
                "description": (
                    "Spawn a sub-agent to run a task. Blocks until the sub-agent completes and returns the result."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Natural language task for the sub-agent",
                        },
                    },
                    "required": ["task"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "load_skill",
                "description": "Load the full prompt instructions for a skill by name. Use this when a task matches a skill's domain so you can follow its specialized guidance.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name of the skill to load"},
                    },
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "report_goal",
                "description": (
                    "Report whether the current standing goal (set by the user via /goal) is met. "
                    "Call this exactly once at the end of a goal-check turn. "
                    "When met=true the automatic goal iteration stops; when met=false the agent is "
                    "prompted again to keep working toward the goal."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "met": {"type": "boolean", "description": "True if the goal is now fully satisfied."},
                        "reason": {"type": "string", "description": "Brief explanation of the current status or what still remains."},
                    },
                    "required": ["met"],
                },
            },
        },
    ]

    if not include_subagent:
        definitions = [
            d for d in definitions
            if d.get("function", {}).get("name") != "spawn_subagent"
        ]
    if allowed is not None:
        definitions = [
            d for d in definitions
            if d.get("function", {}).get("name") in allowed
        ]
    return definitions


class ToolExecutor:
    """Execute built-in tool calls and return structured results."""

    _DISPATCH_MAP: dict[str, str] = {
        "read_file": "_read_file",
        "write_file": "_write_file",
        "edit_file": "_edit_file",
        "multi_edit": "_multi_edit",
        "ls": "_ls",
        "exec_command": "_exec_command",
        "glob_files": "_glob_files",
        "grep_files": "_grep_files",
        "web_search": "_web_search",
        "web_fetch": "_web_fetch",
        "remove_file": "_remove_file",
        "spawn_subagent": "_spawn_subagent",
        "load_skill": "_load_skill",
        "report_goal": "_report_goal",
    }

    def __init__(
        self,
        console: AgentConsole | None = None,
        subagent_manager=None,
        work_dir: str | None = None,
        change_tracker=None,
        get_round_id=None,
        skills: dict | None = None,
        cancellation: CancellationToken | None = None,
    ):
        self.console = console or AgentConsole()
        self.cancellation = cancellation
        self.subagent_manager = subagent_manager
        self._work_dir = str(Path(work_dir).absolute()) if work_dir else str(Path.cwd())
        self._change_tracker = change_tracker
        self._get_round_id = get_round_id  # callable → int
        self._skills = skills or {}
        self._loaded_skills: set[str] = set()
        # Per-session usage counter: skill name → number of load_skill invocations
        # (including redundant "already loaded" calls). Persisted with the session
        # for telemetry and surfaced in /usage and the final summary.
        self._skill_loads: dict[str, int] = {}
        # Last report_goal() call result: None, or {"met": bool, "reason": str}.
        # Consumed by LLMAgent after each goal-check round.
        self.goal_report: dict | None = None

    def _resolve_path(self, path: str | os.PathLike[str]) -> Path:
        """Resolve a caller path lexically against the executor's work directory."""
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = Path(self._work_dir) / candidate
        return Path(os.path.abspath(candidate))

    def execute(self, function_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch and execute a tool call. Returns dict with success key."""
        method_name = self._DISPATCH_MAP.get(function_name)
        if not method_name:
            return {"error": f"Unknown tool: {function_name}", "success": False}
        if self.cancellation is not None and self.cancellation.cancelled:
            return {
                "error": "Tool call skipped: the user interrupted the turn.",
                "success": False,
                "interrupted": True,
            }
        try:
            return getattr(self, method_name)(**arguments)
        except Exception as e:
            return {"error": f"Tool execution failed: {e}", "success": False}

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
            from kiui.agent.gitignore import build_gitignore_matcher
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


    def _exec_command(self, command: str, cwd: str | None = None) -> dict[str, Any]:
        """Execute a shell command, streaming output in real-time.

        Uses subprocess.Popen with reader threads so output is displayed as it
        arrives. Rolling character buffers bound memory use for long-running
        processes; the returned result keeps trailing output from both streams,
        reserving up to half its character budget for stderr.
        """
        cwd = str(self._resolve_path(cwd or "."))
        self.console.tool(f"exec_command: {command} (cwd={cwd})")

        artifact_file = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix="kia-exec-", suffix=".txt", delete=False
        )
        artifact_path = artifact_file.name

        if sys.platform == "win32":
            # Use PowerShell (with user profile) as the modern default on Windows.
            # -NoLogo suppresses the copyright banner; profile is loaded by default.
            shell_cmd = ["powershell", "-NoLogo", "-Command", command]
            try:
                proc = subprocess.Popen(
                    shell_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    cwd=cwd or None,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                )
            except Exception:
                artifact_file.close()
                Path(artifact_path).unlink(missing_ok=True)
                raise
        else:
            shell_cmd = ["/bin/bash", "-lc", command]
            try:
                proc = subprocess.Popen(
                    shell_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    cwd=cwd or None, start_new_session=True,
                )
            except Exception:
                artifact_file.close()
                Path(artifact_path).unlink(missing_ok=True)
                raise

        stdout_lines: deque[str] = deque()
        stderr_lines: deque[str] = deque()
        stdout_size = [0]
        stderr_size = [0]
        artifact_lock = threading.Lock()
        artifact_size_bytes = [0]
        total_output_chars = [0]
        artifact_truncated = [False]
        artifact_write_error: list[str] = []
        capture_stopped = threading.Event()

        def _drain(stream, lines_buf, size_ref, prefix=""):
            decoder = codecs.getincrementaldecoder(locale.getpreferredencoding())(errors="replace")

            def consume(text: str) -> None:
                if not text or capture_stopped.is_set():
                    return
                lines_buf.append(text)
                size_ref[0] += len(text)
                while size_ref[0] > MAX_STREAMING_BUFFER_CHARS and len(lines_buf) > 1:
                    size_ref[0] -= len(lines_buf.popleft())
                captured = f"{prefix}{text}"
                encoded = captured.encode("utf-8")
                with artifact_lock:
                    if capture_stopped.is_set():
                        return
                    total_output_chars[0] += len(captured)
                    remaining = MAX_EXEC_ARTIFACT_BYTES - artifact_size_bytes[0]
                    if len(encoded) > remaining:
                        artifact_truncated[0] = True
                    if remaining > 0 and not artifact_write_error:
                        chunk = encoded[:remaining].decode("utf-8", errors="ignore")
                        try:
                            artifact_file.write(chunk)
                        except OSError as e:
                            artifact_write_error.append(str(e))
                            artifact_truncated[0] = True
                        else:
                            artifact_size_bytes[0] += len(chunk.encode("utf-8"))
                for display in re.split(r"[\r\n]+", text):
                    if display:
                        self.console.print(f"  {prefix}{display}", style="dim")

            pending = ""
            try:
                while raw := stream.read1(4096):
                    pending += decoder.decode(raw)
                    start = 0
                    for match in re.finditer(r"\r\n|\r|\n", pending):
                        consume(pending[start:match.end()])
                        start = match.end()
                    pending = pending[start:]
                pending += decoder.decode(b"", final=True)
                consume(pending)
            finally:
                stream.close()

        t_out = threading.Thread(
            target=_drain, args=(proc.stdout, stdout_lines, stdout_size), daemon=True,
        )
        t_err = threading.Thread(
            target=_drain, args=(proc.stderr, stderr_lines, stderr_size, "[stderr] "), daemon=True,
        )
        t_out.start()
        t_err.start()

        # Wait for the process, watching the keyboard so ESC / Ctrl+C aborts it.
        interrupted = False
        try:
            with CancelWatcher(self.cancellation) as watcher:
                while proc.poll() is None:
                    if watcher.is_cancelled:
                        interrupted = True
                        break
                    time.sleep(0.1)
        except KeyboardInterrupt:
            interrupted = True

        if interrupted:
            self.console.warn("Interrupting command...")
            _terminate_process(proc)

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _terminate_process(proc)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        # A background descendant can retain the pipes after the shell exits.
        # Stop capture before closing the shared artifact, then let any lingering
        # daemon drainers discard bytes rather than touching closed state.
        t_out.join(timeout=EXEC_READER_JOIN_TIMEOUT)
        t_err.join(timeout=EXEC_READER_JOIN_TIMEOUT)
        readers_incomplete = t_out.is_alive() or t_err.is_alive()
        if readers_incomplete:
            with artifact_lock:
                capture_stopped.set()
                artifact_truncated[0] = True
            self.console.warn("Output readers did not finish; terminating remaining process tree.")
            _terminate_process(proc)
            t_out.join(timeout=EXEC_READER_JOIN_TIMEOUT)
            t_err.join(timeout=EXEC_READER_JOIN_TIMEOUT)

        with artifact_lock:
            try:
                artifact_file.flush()
            except OSError as e:
                if not artifact_write_error:
                    artifact_write_error.append(str(e))
                artifact_truncated[0] = True
            finally:
                artifact_file.close()

        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)

        total_len = len(stdout) + len(stderr)
        truncated = total_len > MAX_EXEC_OUTPUT_CHARS
        if truncated:
            stderr_budget = min(len(stderr), MAX_EXEC_OUTPUT_CHARS // 2)
            stdout = stdout[-(MAX_EXEC_OUTPUT_CHARS - stderr_budget):]
            stderr = stderr[-(MAX_EXEC_OUTPUT_CHARS - len(stdout)):]

        res: dict[str, Any] = {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": proc.returncode if proc.returncode is not None else -1,
            "success": not interrupted and proc.returncode == 0,
            "streamed": True,
            "_artifact_path": artifact_path,
            "original_output_chars": total_output_chars[0],
            "artifact_size_bytes": artifact_size_bytes[0],
            "artifact_truncated": artifact_truncated[0],
        }
        if truncated:
            res["truncation_notice"] = (
                f"[output truncated: showing {len(stdout) + len(stderr)} of {total_len} characters]"
            )
        if artifact_write_error:
            res["artifact_capture_error"] = artifact_write_error[0]
        if readers_incomplete:
            res["artifact_capture_incomplete"] = True
        if interrupted:
            res["interrupted"] = True
            res["error"] = "Command was interrupted by user."

        return res

    def _glob_files(self, pattern: str, base_dir: str | None = None, recursive: bool = True, include_ignored: bool = False) -> dict[str, Any]:
        """Find files matching a glob pattern (gitignore-aware)."""
        self.console.tool(f"glob_files {pattern} (recursive={recursive})")

        base = self._resolve_path(base_dir or ".")
        if not base.is_dir():
            return {"error": f"Not a directory: {base}", "success": False}

        matcher = None
        if not include_ignored:
            from kiui.agent.gitignore import build_gitignore_matcher
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
            from kiui.agent.gitignore import build_gitignore_matcher
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

    def _web_search(self, query: str) -> dict[str, Any]:
        """Web search using DuckDuckGo."""
        self.console.tool(f"web_search: {query}")
        try:
            from ddgs import DDGS
        except ImportError:
            return {"error": "web_search requires ddgs: pip install ddgs", "success": False}

        try:
            results = DDGS().text(query, max_results=5)
            if not results:
                return {"content": "No results found.", "success": True}
            
            formatted_results = []
            for res in results:
                title = res.get("title", "No title")
                href = res.get("href", "No URL")
                body = res.get("body", "No description")
                formatted_results.append(f"Title: {title}\nURL: {href}\nSnippet: {body}\n")
            
            return {"content": "\n".join(formatted_results), "success": True}
        except Exception as e:
            return {"error": f"Search failed: {e}", "success": False}

    def _web_fetch(self, url: str) -> dict[str, Any]:
        """Fetch public HTTP(S) content with bounded redirects and bytes."""
        self.console.tool(f"web_fetch: {url}")
        try:
            import httpcore
            import httpx
            from bs4 import BeautifulSoup
        except ImportError:
            return {"error": "web_fetch requires httpx and beautifulsoup4: pip install httpx beautifulsoup4", "success": False}

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        current = httpx.URL(url)
        final_url = current
        body = b""
        try:
            for redirect_count in range(MAX_WEB_REDIRECTS + 1):
                if current.scheme not in ("http", "https") or not current.host:
                    raise ValueError("only absolute HTTP(S) URLs are allowed")
                if current.userinfo:
                    raise ValueError("URL credentials are not allowed")

                port = current.port or (443 if current.scheme == "https" else 80)
                addresses = _resolve_public_addresses(current.host, port)

                class PinnedBackend(httpcore.SyncBackend):
                    def connect_tcp(self, host, port, **kwargs):
                        last_error = None
                        for address in addresses:
                            try:
                                return super().connect_tcp(address, port, **kwargs)
                            except Exception as exc:
                                last_error = exc
                        assert last_error is not None
                        raise last_error

                pool = httpcore.ConnectionPool(
                    ssl_context=httpx.create_ssl_context(trust_env=False),
                    network_backend=PinnedBackend(),
                )
                transport = httpx.HTTPTransport()
                transport._pool = pool
                try:
                    with httpx.Client(transport=transport, trust_env=False) as client:
                        with client.stream(
                            "GET", current, headers=headers, timeout=30.0
                        ) as response:
                            if response.status_code in (301, 302, 303, 307, 308):
                                location = response.headers.get("location")
                                if not location:
                                    raise ValueError("redirect response has no Location header")
                                if redirect_count == MAX_WEB_REDIRECTS:
                                    raise ValueError("too many redirects")
                                current = current.join(location)
                                continue
                            response.raise_for_status()
                            chunks = []
                            size = 0
                            for chunk in response.iter_bytes():
                                size += len(chunk)
                                if size > MAX_WEB_FETCH_BYTES:
                                    raise ValueError(
                                        f"response exceeds {MAX_WEB_FETCH_BYTES} bytes"
                                    )
                                chunks.append(chunk)
                            body = b"".join(chunks)
                            final_url = current
                            encoding = response.encoding or "utf-8"
                            break
                finally:
                    transport.close()
            else:  # pragma: no cover - loop always breaks or raises
                raise ValueError("too many redirects")
        except Exception as e:
            return {"error": f"Failed to fetch URL: {e}", "success": False}

        text_body = body.decode(encoding, errors="replace")
        soup = BeautifulSoup(text_body, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)

        title = soup.find("title")
        if title:
            text = f"# {title.get_text().strip()}\n\n{text}"

        truncated = len(text) > MAX_WEB_FETCH_CHARS
        content = text[:MAX_WEB_FETCH_CHARS]
        if truncated:
            content += f"\n[output truncated: showing first {MAX_WEB_FETCH_CHARS} of {len(text)} chars]"

        return {"content": content, "url": str(final_url), "success": True}

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

    # ── Sub-agent tool ───────────────────────────────────────

    def _spawn_subagent(self, task: str = "") -> dict[str, Any]:
        """Spawn a sub-agent and wait for it to complete."""
        self.console.tool(f"spawn_subagent: {task[:60]}")

        if self.subagent_manager is None:
            return {"error": "Sub-agent spawning is not available.", "success": False}
        if not task:
            return {"error": "task is required.", "success": False}

        return self.subagent_manager.spawn(task=task, cwd=self._work_dir)

    # ── Skill tool ────────────────────────────────────────────

    def _load_skill(self, name: str) -> dict[str, Any]:
        """Load a skill's full prompt instructions into the conversation context."""
        self.console.tool(f"load_skill {name}")

        if not self._skills:
            return {
                "error": "No skills installed. Create a folder under .kia/skills/<name>/ with a SKILL.md file.",
                "success": False,
            }

        if name not in self._skills:
            available = ", ".join(sorted(self._skills.keys()))
            return {
                "error": f"Skill '{name}' not found. Available: {available}",
                "success": False,
            }

        if name in self._loaded_skills:
            self._skill_loads[name] = self._skill_loads.get(name, 0) + 1
            return {"message": f"Skill '{name}' is already loaded.", "success": True}

        self._loaded_skills.add(name)
        self._skill_loads[name] = self._skill_loads.get(name, 0) + 1
        skill = self._skills[name]
        body = skill["body"]
        skill_dir = skill.get("dir")
        resources = [
            directory
            for directory in ("references", "scripts", "assets")
            if skill_dir and (Path(skill_dir) / directory).is_dir()
        ]
        if resources:
            resource_list = ", ".join(f"{directory}/…" for directory in resources)
            header = (
                f"[Skill '{name}' loaded. Its directory is {skill_dir} — resolve relative "
                f"files in {resource_list} against that path using read_file / exec_command "
                f"as the instructions require.]\n\n"
            )
        else:
            header = f"[Skill '{name}' loaded.]\n\n"
        body = header + body
        return {"content": body, "success": True}

    # ── Goal tool ────────────────────────────────────────────

    def _report_goal(self, met: bool = False, reason: str = "") -> dict[str, Any]:
        """Record whether the current standing goal is met.

        The result is stashed on ``self.goal_report`` for the agent loop to
        read after the round; it decides whether to keep auto-iterating.
        """
        met = bool(met)
        reason = reason or ""
        self.console.tool(f"report_goal(met={met})")
        self.goal_report = {"met": met, "reason": reason}
        status = "goal met" if met else "goal not yet met"
        return {
            "message": f"Recorded: {status}." + (f" {reason}" if reason else ""),
            "success": True,
        }


TOOL_SUMMARY_MAX_LINES = 4
TOOL_SUMMARY_MAX_CHARS = 300


def format_tool_summary(result_text: str, max_lines: int = TOOL_SUMMARY_MAX_LINES, max_chars: int = TOOL_SUMMARY_MAX_CHARS) -> str:
    """Truncate a formatted tool result into a brief summary for user display."""
    lines = result_text.splitlines()
    total_lines = len(lines)

    if total_lines <= max_lines and len(result_text) <= max_chars:
        return result_text

    shown = lines[:max_lines]
    summary = "\n".join(shown)

    if len(summary) > max_chars:
        summary = summary[:max_chars].rstrip()

    remaining = total_lines - max_lines
    if remaining > 0:
        summary += f"\n... ({total_lines} lines total)"
    elif len(result_text) > max_chars:
        summary += "..."

    return summary


def format_tool_result(result: dict[str, Any]) -> str:
    """Format a tool result dict into a string for the conversation."""
    if not result.get("success", False):
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        error = result.get("error", "")
        if stdout or stderr:
            parts = [stdout.rstrip("\n")]
            if stderr:
                parts.append(f"[stderr]: {stderr.rstrip()}")
            if error:
                parts.append(f"Error: {error}")
            return "\n".join(part for part in parts if part)
        return f"Error: {error or 'Unknown error'}"

    if "content" in result:
        return result["content"]
    elif "message" in result:
        return result["message"]
    elif "stdout" in result:
        stdout = result["stdout"]
        stderr = result.get("stderr", "")
        if stdout and stderr:
            text = f"{stdout}\n[stderr]: {stderr}"
        elif stderr:
            text = f"[stderr]: {stderr}"
        else:
            text = stdout
        if result.get("truncation_notice"):
            text += f"\n{result['truncation_notice']}"
        return text
    else:
        return json.dumps(result, indent=2)

"""Built-in tool definitions and executor for kiui agent."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from kiui.agent.ui import AgentConsole

MAX_READ_LINES = 2000
MAX_READ_BYTES = 50_000
MAX_EXEC_OUTPUT_BYTES = 50_000
MAX_WEB_FETCH_CHARS = 20_000
MAX_GLOB_RESULTS = 500
MAX_GREP_MATCHES = 200

_SKIP_DIRS = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".tox", "dist", "build", ".mypy_cache", ".pytest_cache",
})


def get_tool_definitions() -> list[dict[str, Any]]:
    """Return OpenAI-format tool definitions for all built-in tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file. Defaults to first 2000 lines / 50KB. Use offset/limit for larger files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to read"},
                        "offset": {"type": "integer", "description": "Line number to start reading from (1-indexed)"},
                        "limit": {"type": "integer", "description": "Maximum number of lines to read"},
                    },
                    "required": ["path"],
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
                        "path": {"type": "string", "description": "Path to the file to write"},
                        "content": {"type": "string", "description": "Content to write to the file"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Make a surgical edit to a file by replacing exact text. old_text must match exactly (including whitespace) and must appear exactly once in the file. If it matches multiple locations, include more surrounding context to disambiguate.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to edit"},
                        "old_text": {"type": "string", "description": "Exact text to find and replace"},
                        "new_text": {"type": "string", "description": "New text to replace the old text with"},
                    },
                    "required": ["path", "old_text", "new_text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "exec_command",
                "description": "Execute a shell command and return stdout, stderr, and exit code. Output capped at 50KB.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to execute"},
                        "cwd": {"type": "string", "description": "Working directory (optional)"},
                        "timeout": {"type": "integer", "description": "Timeout in seconds (default: 120)"},
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
                    "Find files matching a glob pattern. Searches recursively by default. "
                    "Set recursive=false to match only in the immediate directory. Max 500 results."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py', 'src/**/*.ts', or '*.py'"},
                        "base_dir": {"type": "string", "description": "Directory to search in (default: current directory)"},
                        "recursive": {"type": "boolean", "description": "Search subdirectories recursively (default: true)"},
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
                        "path": {"type": "string", "description": "Path to the file or directory to remove"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "spawn_subagent",
                "description": (
                    "Spawn a sub-agent. mode='run' (default) runs a one-shot task in the background. "
                    "mode='session' starts a persistent session you can send follow-up messages to."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Natural language task for the sub-agent (required for mode=run, ignored for mode=session)",
                        },
                        "label": {
                            "type": "string",
                            "description": "Short label for display/addressing (required for mode=session)",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["run", "session"],
                            "description": "run=one-shot background task, session=persistent (default: run)",
                        },
                        "timeout_seconds": {
                            "type": "integer",
                            "description": "Kill sub-agent after this many seconds, 0=no timeout (default: 0, only for mode=run)",
                        },
                    },
                    "required": ["task"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_subagents",
                "description": "List all active and completed sub-agents with their status.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "kill_subagent",
                "description": "Kill a running sub-agent by its run ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string", "description": "The run ID of the sub-agent to kill"},
                    },
                    "required": ["run_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "send_to_subagent",
                "description": "Send a follow-up message to a persistent sub-agent session and get its response.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "Label or run_id of the persistent session",
                        },
                        "message": {
                            "type": "string",
                            "description": "Message to send to the sub-agent",
                        },
                    },
                    "required": ["target", "message"],
                },
            },
        },
    ]


class ToolExecutor:
    """Execute built-in tool calls and return structured results."""

    _DISPATCH_MAP: dict[str, str] = {
        "read_file": "_read_file",
        "write_file": "_write_file",
        "edit_file": "_edit_file",
        "exec_command": "_exec_command",
        "glob_files": "_glob_files",
        "grep_files": "_grep_files",
        "web_search": "_web_search",
        "web_fetch": "_web_fetch",
        "remove_file": "_remove_file",
        "spawn_subagent": "_spawn_subagent",
        "list_subagents": "_list_subagents",
        "kill_subagent": "_kill_subagent",
        "send_to_subagent": "_send_to_subagent",
    }

    def __init__(self, console: AgentConsole | None = None, subagent_manager=None):
        self.console = console or AgentConsole()
        self.subagent_manager = subagent_manager

    def execute(self, function_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch and execute a tool call. Returns dict with success key."""
        method_name = self._DISPATCH_MAP.get(function_name)
        if not method_name:
            return {"error": f"Unknown tool: {function_name}", "success": False}
        try:
            return getattr(self, method_name)(**arguments)
        except Exception as e:
            return {"error": f"Tool execution failed: {e}", "success": False}

    def _read_file(self, path: str, offset: int | None = None, limit: int | None = None) -> dict[str, Any]:
        """Read file contents with optional offset and limit."""
        self.console.tool(f"read_file {path}")

        file_path = Path(path)
        if not file_path.exists():
            return {"error": f"File not found: {path}", "success": False}
        if not file_path.is_file():
            return {"error": f"Path is not a file: {path}", "success": False}

        try:
            with open(file_path, encoding="utf-8") as f:
                all_lines = f.readlines()
        except UnicodeDecodeError:
            return {"error": f"Cannot read binary file: {path}", "success": False}

        total_lines = len(all_lines)
        lines = all_lines

        if offset:
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

    def _write_file(self, path: str, content: str) -> dict[str, Any]:
        """Write content to file, creating parent directories."""
        self.console.tool(f"write_file {path}")

        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        return {"message": f"Successfully wrote {len(content)} characters to {path}", "success": True}

    def _edit_file(self, path: str, old_text: str, new_text: str) -> dict[str, Any]:
        """Make surgical edit to file by replacing exact text."""
        self.console.tool(f"edit_file {path}")

        file_path = Path(path)
        if not file_path.exists():
            return {"error": f"File not found: {path}", "success": False}

        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        occurrence_count = content.count(old_text)
        if occurrence_count == 0:
            return {"error": f"Text not found in file: {old_text[:100]}...", "success": False}
        if occurrence_count > 1:
            return {
                "error": (
                    f"old_text matches {occurrence_count} locations in {path}. "
                    "Include more surrounding context in old_text to make it unique."
                ),
                "success": False,
            }

        match_pos = content.index(old_text)
        line_num = content[:match_pos].count("\n") + 1

        new_content = content[:match_pos] + new_text + content[match_pos + len(old_text):]
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        if len(old_text) < 200 and len(new_text) < 200:
            diff_msg = f"Line {line_num}:\n- {old_text}\n+ {new_text}"
        else:
            diff_msg = f"Line {line_num}: replaced {len(old_text)} chars with {len(new_text)} chars"

        return {"message": f"Successfully edited {path}\n{diff_msg}", "success": True}

    def _exec_command(self, command: str, cwd: str | None = None, timeout: int | None = None) -> dict[str, Any]:
        """Execute shell command."""
        self.console.tool(f"exec_command: {command}")

        timeout = max(1, min(timeout or 120, 600))

        if sys.platform == "win32":
            shell_cmd = ["cmd.exe", "/c", command]
        else:
            shell_cmd = ["/bin/bash", "-lc", command]

        try:
            result = subprocess.run(
                shell_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                cwd=cwd or None,
            )
        except subprocess.TimeoutExpired as e:
            partial = (e.stdout or "")[:MAX_EXEC_OUTPUT_BYTES]
            return {"error": f"Command timed out after {timeout}s. Partial output:\n{partial}", "success": False}

        stdout = result.stdout or ""
        stderr = result.stderr or ""

        truncated = False
        combined_len = len(stdout) + len(stderr)
        if combined_len > MAX_EXEC_OUTPUT_BYTES:
            truncated = True
            if len(stdout) > MAX_EXEC_OUTPUT_BYTES:
                stdout = stdout[:MAX_EXEC_OUTPUT_BYTES]
                stderr = ""
            else:
                stderr = stderr[:MAX_EXEC_OUTPUT_BYTES - len(stdout)]

        res = {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": result.returncode,
            "success": result.returncode == 0,
        }
        if truncated:
            res["truncation_notice"] = f"[output truncated: showing first {len(stdout) + len(stderr)} of {combined_len} bytes]"
        return res

    def _glob_files(self, pattern: str, base_dir: str | None = None, recursive: bool = True) -> dict[str, Any]:
        """Find files matching a glob pattern."""
        self.console.tool(f"glob_files {pattern} (recursive={recursive})")

        base = Path(base_dir) if base_dir else Path.cwd()
        if not base.is_dir():
            return {"error": f"Not a directory: {base}", "success": False}

        iterator = base.glob(pattern) if ("**" in pattern or not recursive) else base.rglob(pattern)

        matches = []
        for p in iterator:
            if any(part in _SKIP_DIRS for part in p.parts):
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
        self.console.tool(f"grep_files {pattern}")

        base = Path(path) if path else Path.cwd()

        flags = re.IGNORECASE if case_insensitive else 0
        try:
            compiled = re.compile(pattern, flags)
        except re.error as e:
            return {"error": f"Invalid regex: {e}", "success": False}

        def _candidate_files():
            if base.is_file():
                yield base
            else:
                glob_pat = file_glob or "*"
                for p in base.rglob(glob_pat):
                    if p.is_file() and not any(part in _SKIP_DIRS for part in p.parts):
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
        """Web search placeholder."""
        self.console.tool(f"web_search: {query}")
        return {
            "message": f"TODO: Implement web search for query: {query}",
            "results": [],
            "success": True,
        }

    def _web_fetch(self, url: str) -> dict[str, Any]:
        """Fetch URL content and convert to readable text."""
        self.console.tool(f"web_fetch: {url}")
        try:
            import httpx
            from bs4 import BeautifulSoup
        except ImportError:
            return {"error": "web_fetch requires httpx and beautifulsoup4: pip install httpx beautifulsoup4", "success": False}

        try:
            response = httpx.get(url, timeout=30.0, follow_redirects=True)
            response.raise_for_status()
        except Exception as e:
            return {"error": f"Failed to fetch URL: {e}", "success": False}

        soup = BeautifulSoup(response.text, "html.parser")
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

        return {"content": content, "url": url, "success": True}

    def _remove_file(self, path: str) -> dict[str, Any]:
        """Remove a file or directory."""
        self.console.tool(f"remove_file {path}")

        target = Path(path)
        if not target.exists():
            return {"error": f"Path not found: {path}", "success": False}

        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()

        return {"message": f"Removed {path}", "success": True}

    # ── Sub-agent tools ──────────────────────────────────────

    def _spawn_subagent(
        self,
        task: str = "",
        label: str | None = None,
        mode: str = "run",
        timeout_seconds: int = 0,
    ) -> dict[str, Any]:
        """Spawn a sub-agent in run or session mode."""
        self.console.tool(f"spawn_subagent mode={mode} task={task[:60]}")

        if self.subagent_manager is None:
            return {"error": "Sub-agent spawning is not available.", "success": False}

        if mode == "session":
            if not label:
                return {"error": "label is required for mode=session.", "success": False}
            return self.subagent_manager.spawn_session(label=label)
        else:
            if not task:
                return {"error": "task is required for mode=run.", "success": False}
            return self.subagent_manager.spawn(
                task=task,
                label=label,
                timeout_seconds=timeout_seconds,
            )

    def _list_subagents(self) -> dict[str, Any]:
        """List all sub-agents."""
        self.console.tool("list_subagents")

        if self.subagent_manager is None:
            return {"error": "Sub-agent spawning is not available.", "success": False}

        runs = self.subagent_manager.list_runs()
        return {"runs": runs, "count": len(runs), "success": True}

    def _kill_subagent(self, run_id: str) -> dict[str, Any]:
        """Kill a running sub-agent."""
        self.console.tool(f"kill_subagent {run_id}")

        if self.subagent_manager is None:
            return {"error": "Sub-agent spawning is not available.", "success": False}

        return self.subagent_manager.kill(run_id)

    def _send_to_subagent(self, target: str, message: str) -> dict[str, Any]:
        """Send a message to a persistent sub-agent session."""
        self.console.tool(f"send_to_subagent target={target}")

        if self.subagent_manager is None:
            return {"error": "Sub-agent spawning is not available.", "success": False}

        return self.subagent_manager.send(target, message)


def format_tool_result(result: dict[str, Any]) -> str:
    """Format a tool result dict into a string for the conversation."""
    if not result.get("success", False):
        error_msg = result.get("error", "")
        if not error_msg:
            error_msg = (result.get("stderr") or result.get("stdout") or "Unknown error").strip()
        return f"Error: {error_msg}"

    if "content" in result:
        return result["content"]
    elif "message" in result:
        return result["message"]
    elif "stdout" in result:
        text = result["stdout"]
        if result.get("stderr"):
            text += f"\n[stderr]: {result['stderr']}"
        if result.get("truncation_notice"):
            text += f"\n{result['truncation_notice']}"
        return text
    else:
        return json.dumps(result, indent=2)

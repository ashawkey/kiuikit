"""Formatting helpers for tool calls and their results."""

import json
from typing import Any, Callable

from .constants import MAX_READ_LINES, MAX_TOOL_OUTPUT_CHARS

TOOL_SUMMARY_MAX_LINES = 4
TOOL_SUMMARY_MAX_CHARS = 300
CALL_VALUE_MAX_CHARS = 60


def truncate_text_output(
    text: str,
    guidance: str,
    limit: int = MAX_TOOL_OUTPUT_CHARS,
) -> tuple[str, bool]:
    """Bound model-facing text while preserving a complete recovery notice."""
    if len(text) <= limit:
        return text, False

    notice = ""
    while True:
        shown = limit - len(notice)
        updated = (
            f"\n[output truncated: showing first {shown:,} of {len(text):,} characters. "
            f"{guidance}]"
        )
        if updated == notice:
            break
        notice = updated
    return text[:limit - len(notice)] + notice, True


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


def result_text_failed(result_text: str) -> bool:
    """Whether a formatted result records a failure.

    Only for replay, which has the formatted text but not the result dict that
    carries ``success``. :func:`format_tool_result` renders a failure either as
    the whole text or as its final line, so anchoring on those two positions
    avoids calling a successful result a failure for merely containing the word.
    """
    lines = result_text.splitlines()
    return bool(lines) and (lines[0].startswith("Error: ") or lines[-1].startswith("Error: "))


def _compact_value(value: Any) -> str:
    if isinstance(value, str):
        text = " ".join(value.split())
        return text if len(text) <= CALL_VALUE_MAX_CHARS else text[: CALL_VALUE_MAX_CHARS - 1] + "…"
    if isinstance(value, (list, tuple)):
        return f"[{len(value)} items]"
    if isinstance(value, dict):
        return f"{{{len(value)} keys}}"
    return json.dumps(value, ensure_ascii=False)


def _describe_read_file(args: dict[str, Any]) -> str:
    start = max(1, args.get("offset") or 1)
    limit = args.get("limit")
    effective_limit = limit if limit is not None else MAX_READ_LINES
    return f"read_file {args['file']}:{start}-{start + effective_limit - 1}"


def _describe_grep_files(args: dict[str, Any]) -> str:
    parts = [f"grep_files {args['pattern']}"]
    if args.get("path"):
        parts.append(f"path={args['path']}")
    if args.get("file_glob"):
        parts.append(f"glob={args['file_glob']}")
    if args.get("case_insensitive"):
        parts.append("(case-insensitive)")
    return " ".join(parts)


# One label per built-in tool, derived purely from the call arguments so that a
# live call and a replayed one render identically.
_CALL_DESCRIBERS: dict[str, Callable[[dict[str, Any]], str]] = {
    "exec_command": lambda a: f"exec_command: {a['command']} (cwd={a.get('cwd') or '.'})",
    "read_file": _describe_read_file,
    "read_image": lambda a: f"read_image {a['file']}",
    "write_file": lambda a: f"write_file {a['file']}",
    "edit_file": lambda a: f"edit_file {a['file']}",
    "multi_edit": lambda a: f"multi_edit {a['file']} ({len(a.get('edits') or [])} edits)",
    "ls": lambda a: f"ls {a.get('path') or '.'}" + (" (all)" if a.get("all") else ""),
    "remove_file": lambda a: f"remove_file {a['file']}",
    "glob_files": lambda a: f"glob_files {a['pattern']} (recursive={a.get('recursive', True)})",
    "grep_files": _describe_grep_files,
    "spawn_subagent": lambda a: f"spawn_subagent: {a['task'][:60]}",
    "load_skill": lambda a: f"load_skill {a['name']}",
    "report_goal": lambda a: f"report_goal(met={a['met']})",
    "web_search": lambda a: f"web_search: {a['query']}",
    "web_fetch": lambda a: f"web_fetch: {a['url']}",
}


def describe_tool_call(name: str, args: dict[str, Any]) -> str:
    """Render one line describing a tool call, for live display and for replay.

    Skill tools and any call whose arguments do not fit the built-in shape (the
    model can emit anything, and replay reads whatever was persisted) fall back
    to a compact ``name(key=value)`` form rather than failing to render.
    """
    describer = _CALL_DESCRIBERS.get(name)
    if describer is not None:
        try:
            return describer(args)
        except (KeyError, TypeError, IndexError):
            pass
    if not args:
        return name
    return f"{name}({', '.join(f'{key}={_compact_value(value)}' for key, value in args.items())})"

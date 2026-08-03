"""Formatting helpers for tool calls and their results."""

from dataclasses import dataclass
import json
import re
from typing import Any, Callable

from .constants import MAX_TOOL_OUTPUT_CHARS

TOOL_SUMMARY_MAX_LINES = 4
TOOL_SUMMARY_MAX_CHARS = 300
CALL_VALUE_MAX_CHARS = 60


@dataclass(frozen=True)
class ToolCallDescription:
    """Semantic, one-line description shared by terminal, web, and replay."""

    name: str
    primary: str = ""
    qualifiers: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        head = f"{self.name} {self.primary}" if self.primary else self.name
        return " · ".join((head, *self.qualifiers))


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


def _format_command_result(result: dict[str, Any]) -> str:
    """Append execution metadata to the command's unmodified merged output."""
    text = result.get("stdout", "")
    if result.get("truncation_notice"):
        text += f"\n{result['truncation_notice']}"
    status = (
        f"[exit_code: {result['exit_code']}, "
        f"interrupted: {str(result.get('interrupted', False)).lower()}, "
        f"timed_out: {str(result.get('timed_out', False)).lower()}]"
    )
    if not text:
        return status
    separator = "" if text.endswith("\n") else "\n"
    return f"{text}{separator}{status}"


def format_tool_result(result: dict[str, Any]) -> str:
    """Format a tool result dict into a string for the conversation."""
    if "exit_code" in result and "stdout" in result:
        return _format_command_result(result)

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


def describe_tool_output(
    name: str,
    result: dict[str, Any],
    describer: Callable[[dict[str, Any]], str] | None = None,
) -> str:
    """Return a concise user-facing description of a tool result.

    Tool-owned describers receive the complete result dictionary. Invalid
    descriptions fall back to the generic formatted summary so display logic
    can never break tool execution.
    """
    if describer is None:
        from .builtin_descriptions import BUILTIN_OUTPUT_DESCRIBERS

        describer = BUILTIN_OUTPUT_DESCRIBERS.get(name)
    if describer is not None and result.get("success", False):
        try:
            description = describer(result)
            if not isinstance(description, str) or not description.strip():
                raise TypeError("tool output describer must return a non-empty string")
            return format_tool_summary(description)
        except (KeyError, TypeError, IndexError, ValueError):
            pass
    return format_tool_summary(format_tool_result(result))


def result_text_failed(result_text: str) -> bool:
    """Whether a formatted result records a failure during replay."""
    lines = result_text.splitlines()
    if not lines:
        return False
    status = re.fullmatch(
        r"\[exit_code: (-?\d+), interrupted: (true|false), timed_out: (true|false)\]",
        lines[-1],
    )
    if status:
        return status.group(1) != "0" or status.group(2) == "true" or status.group(3) == "true"
    return lines[0].startswith("Error: ") or lines[-1].startswith("Error: ")


def _compact_text(value: str) -> str:
    text = " ".join(value.split())
    return text if len(text) <= CALL_VALUE_MAX_CHARS else text[: CALL_VALUE_MAX_CHARS - 1] + "…"


def _compact_value(value: Any, key: str = "") -> str:
    if isinstance(value, str):
        if key in {"content", "old_text", "new_text", "text"}:
            return f"<{len(value)} chars>"
        return _compact_text(value)
    if isinstance(value, (list, tuple)):
        return f"[{len(value)} items]"
    if isinstance(value, dict):
        return f"{{{len(value)} keys}}"
    return json.dumps(value, ensure_ascii=False)


def quote_tool_call_value(value: Any, *, compact: bool = True) -> str:
    """Quote a user-facing tool-call value.

    ``compact=True`` (default) collapses whitespace and caps the text so
    labels stay short; ``compact=False`` still collapses whitespace (a
    multi-line shell command becomes one line) but never truncates, which
    ``exec_command`` relies on to show the full command it is about to run.
    """
    text = str(value)
    if compact:
        text = _compact_text(text)
    else:
        text = " ".join(text.split())
    return json.dumps(text, ensure_ascii=False)


def build_tool_call_description(
    name: str,
    args: dict[str, Any],
    describer: Callable[[dict[str, Any]], ToolCallDescription] | None = None,
) -> ToolCallDescription:
    """Build a robust description using an owner-provided formatter or fallback."""
    if describer is None:
        # Keep the public helper useful for built-ins without coupling the
        # generic formatter to their definitions at import time.
        from .builtin_descriptions import BUILTIN_CALL_DESCRIBERS

        describer = BUILTIN_CALL_DESCRIBERS.get(name)
    if describer is not None:
        try:
            description = describer(args)
            if not isinstance(description, ToolCallDescription):
                raise TypeError("tool call describer must return ToolCallDescription")
            return description
        except (KeyError, TypeError, IndexError, ValueError):
            pass
    qualifiers = tuple(f"{key}={_compact_value(value, key)}" for key, value in args.items())
    return ToolCallDescription(name, qualifiers=qualifiers)


def describe_tool_call(
    name: str,
    args: dict[str, Any],
    describer: Callable[[dict[str, Any]], ToolCallDescription] | None = None,
) -> str:
    """Return the canonical plain-text description of a tool call."""
    return build_tool_call_description(name, args, describer).text


def log_tool_call(
    console: Any,
    name: str,
    args: dict[str, Any],
    describer: Callable[[dict[str, Any]], ToolCallDescription] | None = None,
) -> None:
    """Render one call through the console's structured tool-call interface."""
    description = build_tool_call_description(name, args, describer)
    console.tool(
        description.text,
        name=description.name,
        primary=description.primary,
        qualifiers=list(description.qualifiers),
    )

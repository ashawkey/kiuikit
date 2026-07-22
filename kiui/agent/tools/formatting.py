"""Formatting helpers for tool results."""

import json
from typing import Any

from .constants import MAX_TOOL_OUTPUT_CHARS

TOOL_SUMMARY_MAX_LINES = 4
TOOL_SUMMARY_MAX_CHARS = 300


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

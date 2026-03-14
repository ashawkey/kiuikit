"""Context management for kiui agent: truncation, pruning, and compaction.

Three layers of context window management:
1. Tool output truncation — cap individual tool results relative to context window.
2. Context pruning — soft-trim then hard-clear old tool results at 30%/50%.
3. Compaction — LLM-summarize oldest messages when context exceeds 60%.
"""

from __future__ import annotations

import sys
from typing import Any


from kiui.agent.ui import AgentConsole

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHARS_PER_TOKEN = 4

# Layer 1: generic truncation of a single tool result
TRUNCATION_RATIO = 0.3  # max result size as fraction of context window
TRUNCATION_MAX = 400_000
TRUNCATION_MIN = 2000

# Layer 2: pruning thresholds (fraction of context window in chars)
SOFT_TRIM_RATIO = 0.3
HARD_CLEAR_RATIO = 0.5

SOFT_TRIM_THRESHOLD = 6000  # trim results larger than this (head+tail ≈ 3K, so min savings ≈ 50%)
SOFT_TRIM_HEAD = 1500
SOFT_TRIM_TAIL = 1500

MIN_PRUNABLE_CHARS = 50_000
KEEP_LAST_ASSISTANTS = 3
HARD_CLEAR_PLACEHOLDER = "[compacted: tool output removed to free context]"

# Layer 3: compaction threshold
COMPACTION_RATIO = 0.75
COMPACTION_SUMMARY_MAX_CHARS = 50_000

PRUNABLE_TOOLS = frozenset({
    "read_file", "exec_command", "web_fetch",
    "web_search", "glob_files", "grep_files",
})

# ---------------------------------------------------------------------------
# Message helpers (handle both dict messages and OpenAI ChatCompletionMessage)
# ---------------------------------------------------------------------------


def _get_role(msg: Any) -> str:
    if isinstance(msg, dict):
        return msg.get("role", "")
    return getattr(msg, "role", "")


def _get_text(msg: Any) -> str:
    """Extract concatenated text content from a message."""
    if isinstance(msg, dict):
        content = msg.get("content", "")
    else:
        content = getattr(msg, "content", "") or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


def _set_text(msg: dict, text: str) -> dict:
    """Return a shallow copy of *msg* with text content replaced."""
    content = msg.get("content")
    if isinstance(content, list):
        return {**msg, "content": [{"type": "text", "text": text}]}
    return {**msg, "content": text}


def _get_tool_calls(msg: Any) -> list:
    if isinstance(msg, dict):
        return msg.get("tool_calls") or []
    return getattr(msg, "tool_calls", None) or []


def _get_tool_call_id(msg: Any) -> str | None:
    if isinstance(msg, dict):
        return msg.get("tool_call_id")
    return getattr(msg, "tool_call_id", None)


def _msg_chars(msg: Any) -> int:
    """Rough character cost of a single message."""
    chars = len(_get_text(msg))
    if _get_role(msg) == "assistant":
        for tc in _get_tool_calls(msg):
            if isinstance(tc, dict):
                args = tc.get("function", {}).get("arguments", "")
            else:
                args = getattr(getattr(tc, "function", None), "arguments", "") or ""
            chars += len(args) if isinstance(args, str) else 0
    return chars


def estimate_context_chars(messages: list) -> int:
    return sum(_msg_chars(m) for m in messages)


# ---------------------------------------------------------------------------
# Layer 1: Tool result truncation
# ---------------------------------------------------------------------------


def truncate_tool_result(text: str, context_window: int) -> str:
    """Cap a tool result string to a size proportional to the context window."""
    max_chars = max(
        TRUNCATION_MIN,
        min(int(context_window * TRUNCATION_RATIO * CHARS_PER_TOKEN), TRUNCATION_MAX),
    )
    if len(text) <= max_chars:
        return text
    # try to break at a newline within the last 30 % of the budget
    cut = max_chars
    search_start = int(max_chars * 0.7)
    nl = text.rfind("\n", search_start, max_chars)
    if nl > search_start:
        cut = nl
    return text[:cut] + f"\n[truncated: {len(text)} chars total, showing first {cut}]"


# ---------------------------------------------------------------------------
# Layer 2: Context pruning
# ---------------------------------------------------------------------------


def _find_tool_name(messages: list, tool_idx: int) -> str | None:
    """Walk backward from a tool-result message to find the originating tool name."""
    target_id = _get_tool_call_id(messages[tool_idx])
    if not target_id:
        return None
    for i in range(tool_idx - 1, -1, -1):
        if _get_role(messages[i]) != "assistant":
            continue
        for tc in _get_tool_calls(messages[i]):
            tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if tc_id == target_id:
                if isinstance(tc, dict):
                    return tc.get("function", {}).get("name")
                return getattr(getattr(tc, "function", None), "name", None)
        # Tool results always directly follow their originating assistant
        # message, so only the first assistant found can be the match.
        break
    return None


def _soft_trim(text: str) -> str | None:
    """Return head+tail trimmed text, or None if small enough."""
    if len(text) <= SOFT_TRIM_THRESHOLD:
        return None
    head = text[:SOFT_TRIM_HEAD]
    tail = text[-SOFT_TRIM_TAIL:]
    return (
        f"{head}\n...\n{tail}"
        f"\n\n[Trimmed: kept first {SOFT_TRIM_HEAD} and "
        f"last {SOFT_TRIM_TAIL} chars of {len(text)} total]"
    )


def _prunable_range(messages: list) -> tuple[int, int]:
    """Return (start, end) indices of the zone eligible for pruning.

    Returns an empty range when fewer than KEEP_LAST_ASSISTANTS assistant
    turns exist — there is nothing old enough to safely prune.
    """
    start = 0
    for i, m in enumerate(messages):
        if _get_role(m) == "user":
            start = i
            break

    remaining = KEEP_LAST_ASSISTANTS
    end = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        if _get_role(messages[i]) == "assistant":
            remaining -= 1
            if remaining == 0:
                end = i
                break

    if remaining > 0:
        return 0, 0

    return start, end


def prune_context(messages: list, context_window: int) -> list:
    """Prune old tool results to manage context window usage.

    Phase 1 (soft trim, 30 %): keep head + tail of large prunable results.
    Phase 2 (hard clear, 50 %): replace entire results with a placeholder.

    Returns a new list — does not mutate the input.
    """
    if not messages or context_window <= 0:
        return list(messages)

    char_window = context_window * CHARS_PER_TOKEN
    total_chars = estimate_context_chars(messages)

    if total_chars < char_window * SOFT_TRIM_RATIO:
        return list(messages)

    start, end = _prunable_range(messages)

    prunable: list[int] = []
    for i in range(start, end):
        msg = messages[i]
        if _get_role(msg) != "tool" or not isinstance(msg, dict):
            continue
        name = _find_tool_name(messages, i)
        if name and name in PRUNABLE_TOOLS:
            prunable.append(i)

    if not prunable:
        return list(messages)

    result = list(messages)

    # Phase 1: soft trim
    for i in prunable:
        text = _get_text(result[i])
        trimmed = _soft_trim(text)
        if trimmed is not None:
            result[i] = _set_text(result[i], trimmed)
            total_chars += len(trimmed) - len(text)

    ratio = total_chars / char_window
    if ratio < HARD_CLEAR_RATIO:
        return result

    # Phase 2: hard clear (only if enough prunable content exists)
    prunable_chars = sum(len(_get_text(result[i])) for i in prunable)
    if prunable_chars < MIN_PRUNABLE_CHARS:
        return result

    for i in prunable:
        if ratio < HARD_CLEAR_RATIO:
            break
        old_len = len(_get_text(result[i]))
        result[i] = _set_text(result[i], HARD_CLEAR_PLACEHOLDER)
        total_chars += len(HARD_CLEAR_PLACEHOLDER) - old_len
        ratio = total_chars / char_window

    return result


# ---------------------------------------------------------------------------
# Layer 3: Compaction
# ---------------------------------------------------------------------------

# Per-role character limits when building the summarization input.
# These are generous because the total is capped by COMPACTION_SUMMARY_MAX_CHARS.
# System limit is intentionally high to preserve previous compaction summaries.
_SUMMARY_CHAR_LIMITS: dict[str, int] = {
    "system": 800,
    "user": 800,
    "assistant": 800,
    "tool": 500,
}

_COMPACTION_PROMPT = """\
You are summarizing a conversation between a user and an AI coding assistant.
Produce a concise but specific summary that preserves:
- The user's original request and any clarifications
- Key decisions, conclusions, and rationale
- Exact file paths, function/class names, and code snippets that were discussed or modified
- Commands that were run and their significant output
- Current task status and agreed-upon next steps

Preserve technical details (paths, names, values) verbatim — do not paraphrase them.
Keep the summary under 2000 words.

Conversation:
{conversation}

Summary:"""


def needs_compaction(messages: list, context_window: int) -> bool:
    char_window = context_window * CHARS_PER_TOKEN
    return estimate_context_chars(messages) > char_window * COMPACTION_RATIO


def _safe_split_index(messages: list, idx: int) -> int:
    """Walk backward so the split never lands inside a tool-call / result pair."""
    while idx > 1 and _get_role(messages[idx]) == "tool":
        idx -= 1
    return idx


def _messages_to_text(messages: list) -> str:
    """Convert messages to a concise text for the summarization prompt."""
    parts: list[str] = []
    for msg in messages:
        role = _get_role(msg)
        text = _get_text(msg)
        limit = _SUMMARY_CHAR_LIMITS.get(role, 500)

        if role == "system":
            parts.append(f"[System]: {text[:limit]}")
        elif role == "user":
            parts.append(f"User: {text[:limit]}")
        elif role == "assistant":
            if text:
                parts.append(f"Assistant: {text[:limit]}")
            for tc in _get_tool_calls(msg):
                name = (
                    tc.get("function", {}).get("name", "?")
                    if isinstance(tc, dict)
                    else getattr(getattr(tc, "function", None), "name", "?")
                )
                parts.append(f"  [Called tool: {name}]")
        elif role == "tool":
            snippet = text[:limit] + "..." if len(text) > limit else text
            parts.append(f"  [Tool result]: {snippet}")

    joined = "\n".join(parts)
    if len(joined) > COMPACTION_SUMMARY_MAX_CHARS:
        joined = joined[:COMPACTION_SUMMARY_MAX_CHARS] + "\n[... truncated for summarization]"
    return joined


def compact_context(
    messages: list,
    client: Any,
    model: str,
    console: AgentConsole | None = None,
) -> list:
    """Compact messages by LLM-summarizing the oldest ~40 %.

    *messages* should **not** include the system prompt (it is managed
    separately by ContextManager).  Returns a new list.
    """
    if len(messages) <= 2:
        return list(messages)

    split_index = max(1, int(len(messages) * 0.4))
    split_index = _safe_split_index(messages, split_index)

    to_compact = messages[:split_index]
    to_keep = messages[split_index:]

    conversation_text = _messages_to_text(to_compact)
    prompt = _COMPACTION_PROMPT.format(conversation=conversation_text)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            timeout=60,
        )
        summary = response.choices[0].message.content.strip()
    except Exception:
        if console is not None:
            console.warn("Context compaction LLM call failed", exc_info=True)
        else:
            print("[WARNING] Context compaction LLM call failed", file=sys.stderr)
        summary = "[Compaction failed — older context may be incomplete]"

    summary_msg: dict[str, Any] = {
        "role": "system",
        "content": f"[Previous conversation summary]\n{summary}",
    }

    return [summary_msg] + list(to_keep)

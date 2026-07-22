"""Context management for kiui agent: tool-result filtering and compaction.

Three layers of context window management:
1. Tool-result ingress — compact oversized results before they reach history.
2. Context pruning — soft-trim then hard-clear old tool results at 30%/50%.
3. Compaction — LLM-summarize oldest messages when context exceeds 75%.
"""

from __future__ import annotations

import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


from kiui.agent.ui import AgentConsole


class ContextManager:
    """Flat conversation history with context-management hooks.

    Messages use plain dictionaries in the OpenAI wire format. SDK response
    objects are normalized at ingress by :mod:`kiui.agent.utils.streaming`.
    """

    def __init__(self, system_prompt: str):
        self.system_prompt: dict[str, str] = {
            "role": "system",
            "content": system_prompt,
        }
        self.messages: list[dict[str, Any]] = []
        # Running character total of ``self.messages`` (excluding the system
        # prompt, matching estimate_context_chars). Maintained incrementally on
        # append and invalidated (None) on any bulk mutation.
        self._char_cache: int | None = 0

    @property
    def estimated_chars(self) -> int:
        """Cached character total of the history, recomputed only when stale."""
        if self._char_cache is None:
            self._char_cache = estimate_context_chars(self.messages)
        return self._char_cache

    def add(self, message: dict[str, Any]) -> None:
        self.messages.append(message)
        if self._char_cache is not None:
            self._char_cache += msg_chars(message)

    def get(self, include_system: bool = True) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if include_system:
            messages.append(self.system_prompt)
        messages.extend(self.messages)
        return messages

    def replace_messages(self, new_messages: list[dict[str, Any]]) -> None:
        # A same-object call is a no-op (content unchanged) so the cache stays
        # valid; only a genuine replacement invalidates it.
        if new_messages is not self.messages:
            self.messages = list(new_messages)
            self._char_cache = None

    def checkpoint(self) -> list[dict[str, Any]]:
        """Return a shallow copy for rollback on interruption."""
        return list(self.messages)

    def rollback(self, snapshot: list[dict[str, Any]]) -> None:
        self.messages = snapshot
        self._char_cache = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CHARS_PER_TOKEN = 3.3  # conservative for code-heavy workloads

# Layer 1: proactive compaction of a single tool result. Keeping this budget
# small prevents one noisy command from occupying the context until later pruning.
TOOL_RESULT_RATIO = 0.04
TOOL_RESULT_MIN_CHARS = 3_000
TOOL_RESULT_DEFAULT_MAX_CHARS = 12_000
TOOL_RESULT_MAX_CHARS = {
    "read_file": 24_000,
    "inspect_processes": 24_000,
    "web_fetch": 20_000,
}
GENERIC_RESULT_RATIO = 0.1
GENERIC_RESULT_MIN_CHARS = 8_000
GENERIC_RESULT_MAX_CHARS = 24_000

PROACTIVE_TOOLS = frozenset({
    "read_file", "exec_command", "inspect_processes", "ls", "glob_files", "grep_files",
    "web_fetch", "web_search",
})
STRUCTURED_TOOLS = frozenset({"ls", "glob_files", "grep_files", "web_search"})

_NOTABLE_OUTPUT_RE = re.compile(
    r"(?:traceback|error|exception|fail(?:ed|ure)?|fatal|panic|warning|warn|"
    r"assert|passed|skipped|summary|exit code|not found|denied|"
    r"https?://|verification (?:code|uri)|device code|authenticate|authorization)",
    re.IGNORECASE,
)
_ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_PROGRESS_RE = re.compile(r"^\s*(?:[\[\]=\->#|. ]{5,}\d*%?|[⠀-⣿]+)\s*$")
_PYTEST_SUMMARY_RE = re.compile(
    r"(?:=+ .*?(?:passed|failed|error|skipped).*? =+|short test summary info|"
    r"FAILED\s+\S+|ERROR\s+\S+)",
    re.IGNORECASE,
)
_PYTEST_FAILURE_RE = re.compile(
    r"^(?:_{3,} .* _{3,}|E\s+|FAILED\s+|ERROR\s+|>\s+|\s*File \".*\", line \d+)",
    re.IGNORECASE,
)
_COMPILER_DIAGNOSTIC_RE = re.compile(
    r"(?:[^\s:()]+(?:[/\\][^\s:()]+)*\.(?:py|js|jsx|ts|tsx|rs|go|c|cc|cpp|h|hpp|cs|java|kt))"
    r"(?::\d+(?::\d+)?|\(\d+,\d+\)).*\b(?:error|warning)\b|"
    r"\b(?:error|warning)\s+(?:TS\d+|[A-Z]{1,5}\d{3,5})\b",
    re.IGNORECASE,
)

_TOOL_RESULT_GUIDANCE = {
    "read_file": "Use grep_files to locate relevant lines, or read_file with a narrower offset/limit.",
    "exec_command": "Search the saved output with grep_files/read_file, or rerun with quiet flags or a targeted filter.",
    "ls": "Use glob_files with a narrower pattern instead of listing the whole directory.",
    "glob_files": "Use a narrower glob pattern or base_dir.",
    "grep_files": "Use a narrower regex, path, or file_glob.",
    "web_fetch": "Search the saved text or fetch a more specific source.",
    "web_search": "Refine the query before requesting more results.",
}

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
COMPACTION_TARGET_RATIO = 0.5  # target context usage after compaction
COMPACTION_SUMMARY_MAX_CHARS = 50_000

PRUNABLE_TOOLS = frozenset({
    "read_file", "exec_command", "web_fetch", "inspect_processes",
    "web_search", "glob_files", "grep_files", "ls",
})


# ---------------------------------------------------------------------------
# Self-calibrating token estimator
# ---------------------------------------------------------------------------


class TokenEstimator:
    """Self-calibrating chars-per-token estimator.

    Starts with a conservative default and refines via exponential
    moving average each time real token counts are observed from the API.
    """

    EMA_ALPHA = 0.15  # smoothing factor (higher = faster adaptation)
    MIN_RATIO = 2.0
    MAX_RATIO = 6.0

    def __init__(self, initial: float = DEFAULT_CHARS_PER_TOKEN):
        self._ratio = initial
        self._calibrated = False

    @property
    def chars_per_token(self) -> float:
        return self._ratio

    def calibrate(self, prompt_chars: int, prompt_tokens: int) -> None:
        """Update the ratio from an observed (chars, tokens) pair."""
        if prompt_tokens < 100:  # too few tokens to be meaningful
            return
        observed = prompt_chars / prompt_tokens
        observed = max(self.MIN_RATIO, min(self.MAX_RATIO, observed))
        if not self._calibrated:
            self._ratio = observed
            self._calibrated = True
        else:
            self._ratio = (1 - self.EMA_ALPHA) * self._ratio + self.EMA_ALPHA * observed

    def tokens_to_chars(self, tokens: int) -> int:
        """Convert a token count to an estimated character count."""
        return int(tokens * self._ratio)

    def chars_to_tokens(self, chars: int) -> int:
        """Convert a character count to an estimated token count."""
        return int(chars / self._ratio) if self._ratio > 0 else chars


# ---------------------------------------------------------------------------
# Message helpers (context messages are plain dicts in the OpenAI wire
# format; SDK response objects are normalized at ingress, see streaming.py)
# ---------------------------------------------------------------------------


def get_role(msg: dict) -> str:
    return msg.get("role", "")


def get_text(msg: dict) -> str:
    """Extract concatenated text content from a message."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


def set_text(msg: dict, text: str) -> dict:
    """Return a shallow copy of *msg* with text content replaced."""
    content = msg.get("content")
    if isinstance(content, list):
        return {**msg, "content": [{"type": "text", "text": text}]}
    return {**msg, "content": text}


def get_tool_calls(msg: dict) -> list:
    return msg.get("tool_calls") or []


def get_tool_call_id(msg: dict) -> str | None:
    return msg.get("tool_call_id")


def msg_chars(msg: dict) -> int:
    """Rough character cost of a single message."""
    chars = len(get_text(msg))
    if get_role(msg) == "assistant":
        for tc in get_tool_calls(msg):
            chars += len(tc.get("function", {}).get("arguments") or "")
    return chars


def estimate_context_chars(messages: list) -> int:
    return sum(msg_chars(m) for m in messages)


# ---------------------------------------------------------------------------
# Layer 1: Tool-result ingress
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolResultEnvelope:
    """Tool result plus invocation metadata used to select a reducer."""

    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    text: str

    @property
    def original_chars(self) -> int:
        return self.result.get("original_output_chars", len(self.text))


@dataclass(frozen=True)
class ToolCompaction:
    text: str
    compacted: bool
    reducer: str = "passthrough"
    tier: str = "passthrough"
    original_chars: int = 0
    retained_chars: int = 0
    artifact_path: str | None = None
    warnings: tuple[str, ...] = ()


def tool_result_char_budget(
    context_length: int,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    tool_name: str = "",
) -> int:
    """Return the maximum characters one tool result may add to history."""
    if tool_name in PROACTIVE_TOOLS:
        ratio = TOOL_RESULT_RATIO
        min_chars = TOOL_RESULT_MIN_CHARS
        max_chars = TOOL_RESULT_MAX_CHARS.get(tool_name, TOOL_RESULT_DEFAULT_MAX_CHARS)
    else:
        ratio = GENERIC_RESULT_RATIO
        min_chars = GENERIC_RESULT_MIN_CHARS
        max_chars = GENERIC_RESULT_MAX_CHARS
    if context_length <= 0:
        return max_chars
    return max(min_chars, min(int(context_length * ratio * chars_per_token), max_chars))


def _normalize_exec_output(text: str) -> str:
    """Remove terminal control sequences and low-signal progress output."""
    text = _ANSI_RE.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))
    lines: list[str] = []
    blank = False
    for line in text.split("\n"):
        if _PROGRESS_RE.fullmatch(line):
            continue
        if not line.strip():
            if blank:
                continue
            blank = True
        else:
            blank = False
        lines.append(line.rstrip())
    normalized = "\n".join(lines).strip("\n")
    if text.endswith("\n"):
        normalized += "\n"
    return normalized


def _collapse_repeated_lines(text: str) -> str:
    """Collapse consecutive duplicate log lines without hiding unique events."""
    lines = text.splitlines(keepends=True)
    if len(lines) < 3:
        return text

    result: list[str] = []
    i = 0
    while i < len(lines):
        j = i + 1
        while j < len(lines) and lines[j] == lines[i]:
            j += 1
        result.append(lines[i])
        repeats = j - i
        if repeats > 1:
            result.append(f"[previous line repeated {repeats - 1} more times]\n")
        i = j
    return "".join(result)


def _stderr_excerpt(text: str, max_chars: int) -> str:
    """Keep trailing stderr lines within *max_chars*, preserving their order."""
    selected: list[str] = []
    used = 0
    for line in reversed(text.splitlines()):
        if not line.startswith("[stderr]"):
            continue
        separator = 1 if selected else 0
        remaining = max_chars - used - separator
        if remaining <= 0:
            break
        selected.insert(0, line[-remaining:])
        used += len(selected[0]) + separator
    return "\n".join(selected)


def _notable_lines(text: str, max_chars: int) -> str:
    """Extract unique diagnostics from both ends within *max_chars*."""
    candidates: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or not _NOTABLE_OUTPUT_RE.search(line) or line in seen:
            continue
        candidates.append(line)
        seen.add(line)

    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0][:max_chars]

    line_limit = max(1, min(500, max_chars // 2 - 1))
    indices: list[int] = []
    left, right = 0, len(candidates) - 1
    used = 0
    while left <= right:
        for index in (left, right):
            if index in indices:
                continue
            line = candidates[index][:line_limit]
            cost = len(line) + (1 if indices else 0)
            if used + cost > max_chars:
                break
            indices.append(index)
            used += cost
        left += 1
        right -= 1
    return "\n".join(candidates[i][:line_limit] for i in sorted(indices))


def _command_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return []


def _command_reducer(command: str) -> str | None:
    tokens = _command_tokens(command)
    if not tokens:
        return None
    executable = Path(tokens[0]).stem.lower()
    lowered = [token.lower() for token in tokens]

    if executable in {"pytest", "py.test"} or (
        executable in {"python", "python3"} and len(lowered) > 2
        and lowered[1:3] == ["-m", "pytest"]
    ):
        return "pytest"
    if executable == "git" and len(lowered) > 1 and lowered[1] in {"status", "log", "diff", "show"}:
        return f"git-{lowered[1]}"
    if executable in {"pip", "pip3", "uv", "poetry"} or (
        executable in {"python", "python3"} and len(lowered) > 2
        and lowered[1:3] == ["-m", "pip"]
    ):
        return "python-package"
    return None


def _select_lines(text: str, predicate: Callable[[str], bool], max_chars: int) -> str:
    """Select matching lines in order, preserving meaningful repetitions."""
    selected: list[str] = []
    used = 0
    for line in text.splitlines():
        if not predicate(line):
            continue
        remaining = max_chars - used - (1 if selected else 0)
        if remaining <= 0:
            break
        selected.append(line[:remaining])
        used += len(selected[-1]) + (1 if len(selected) > 1 else 0)
    return "\n".join(selected)


def _reduce_pytest(text: str, budget: int) -> str:
    return _select_lines(
        text,
        lambda line: bool(
            _PYTEST_FAILURE_RE.search(line)
            or _PYTEST_SUMMARY_RE.search(line)
            or _NOTABLE_OUTPUT_RE.search(line)
        ),
        budget,
    )


def _reduce_git(text: str, reducer: str, budget: int) -> str:
    if reducer == "git-status":
        pattern = re.compile(
            r"^(?:On branch |HEAD detached|Your branch |Changes to be committed:|"
            r"Changes not staged for commit:|Untracked files:|both modified:|\s+(?:modified:|new file:|deleted:|renamed:)|"
            r"[ MADRCU?!]{2}\s+\S+|##\s+)"
        )
    elif reducer == "git-log":
        pattern = re.compile(r"^(?:commit\s+[0-9a-f]{7,40}|Author:|Date:|[0-9a-f]{7,40}\s+)")
    else:
        pattern = re.compile(r"^(?:diff --git |index |--- |\+\+\+ |@@|\+[^+]|-[^-])")
    return _select_lines(text, lambda line: bool(pattern.search(line)), budget)


def _reduce_python_package(text: str, budget: int) -> str:
    pattern = re.compile(
        r"(?:error|warning|conflict|incompatible|requires|resolution|successfully installed|"
        r"installed|uninstalled|removed|no matching distribution|could not find)",
        re.IGNORECASE,
    )
    return _select_lines(text, lambda line: bool(pattern.search(line)), budget)


def _reduce_compiler(text: str, budget: int) -> str:
    return _select_lines(
        text,
        lambda line: bool(_COMPILER_DIAGNOSTIC_RE.search(line) or _NOTABLE_OUTPUT_RE.search(line)),
        budget,
    )


def _semantic_reduce(envelope: ToolResultEnvelope, text: str, budget: int) -> tuple[str, str] | None:
    if envelope.tool_name != "exec_command":
        return None
    command = envelope.arguments["command"]
    reducer = _command_reducer(command)
    if reducer == "pytest":
        reduced = _reduce_pytest(text, budget)
    elif reducer and reducer.startswith("git-"):
        reduced = _reduce_git(text, reducer, budget)
    elif reducer == "python-package":
        reduced = _reduce_python_package(text, budget)
    else:
        reduced = _reduce_compiler(text, budget) if _COMPILER_DIAGNOSTIC_RE.search(text) else ""
        reducer = "compiler-diagnostics" if reduced else None
    if not reducer or not reduced or len(reduced) >= len(text):
        return None
    return reduced, reducer


def _sample_edges(text: str, available: int, head_ratio: float = 0.5) -> str:
    """Sample line-aligned head and tail excerpts within *available* chars."""
    label = "\n[... middle omitted ...]\n"
    if available <= len(label):
        return label[:available]

    content_budget = available - len(label)
    head_budget = int(content_budget * head_ratio)
    tail_budget = content_budget - head_budget

    head = text[:head_budget]
    newline = head.rfind("\n", int(head_budget * 0.6))
    if newline >= 0:
        head = head[:newline]

    tail = text[-tail_budget:]
    newline = tail.find("\n", 0, int(tail_budget * 0.4))
    if newline >= 0:
        tail = tail[newline + 1:]
    return f"{head}{label}{tail}"


def _sample_prefix(text: str, available: int) -> str:
    """Keep one contiguous prefix so a subsequent offset read is unambiguous."""
    label = "\n[... remainder omitted ...]\n"
    if available <= len(label):
        return label[:available]
    limit = available - len(label)
    prefix = text[:limit]
    newline = prefix.rfind("\n", int(limit * 0.7))
    if newline >= 0:
        prefix = prefix[:newline]
    return f"{prefix}{label}"


def _compaction_footer(
    original_tokens: int,
    retained_tokens: int,
    reducer: str,
    tier: str,
    location: str,
    warning: str,
    guidance: str,
) -> str:
    saved_percent = max(0, min(99, round((1 - retained_tokens / original_tokens) * 100)))
    return (
        f"\n[compacted: ~{original_tokens:,}→~{retained_tokens:,} tokens, "
        f"-{saved_percent}%, reducer={reducer}, tier={tier}.{location}"
        f"{warning} {guidance}]"
    )


def _fit_with_footer(body: str, footer: str, budget: int) -> str:
    """Fit content while preserving the complete recovery footer."""
    if len(footer) >= budget:
        raise ValueError("Tool-result metadata exceeds the compaction budget")
    return f"{body[:budget - len(footer)]}{footer}"


def compact_tool_result_envelope(
    envelope: ToolResultEnvelope,
    context_length: int,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    artifact_path: str | None = None,
) -> ToolCompaction:
    """Compact a tool result with invocation-aware deterministic reducers."""
    tool_name = envelope.tool_name
    raw_text = envelope.text
    budget = tool_result_char_budget(context_length, chars_per_token, tool_name)
    if envelope.original_chars <= budget:
        return ToolCompaction(
            text=raw_text,
            compacted=False,
            original_chars=envelope.original_chars,
            retained_chars=len(raw_text),
            artifact_path=artifact_path,
        )

    original_chars = envelope.original_chars
    warnings = (
        ("recovery artifact reached its 100 MiB capture limit",)
        if envelope.result.get("artifact_truncated")
        else ()
    )
    trailing_notice = ""
    if tool_name == "read_file":
        last_line = raw_text.rstrip().rsplit("\n", 1)[-1]
        if last_line.startswith("[output truncated:"):
            trailing_notice = f"\n{last_line}"

    text = raw_text
    if tool_name == "exec_command":
        text = _collapse_repeated_lines(_normalize_exec_output(raw_text))

    guidance = _TOOL_RESULT_GUIDANCE.get(
        tool_name,
        "Use a narrower tool call or read the captured output in slices.",
    )
    if artifact_path:
        display_path = artifact_path
        if len(display_path) > 512:
            display_path = f"...{display_path[-509:]}"
        location = f" Captured output: {display_path}."
    else:
        location = " Captured output could not be saved."
        guidance = guidance.replace(
            "Search the saved output with grep_files/read_file, or ",
            "",
        ).replace(
            "Search the saved text or ",
            "",
        )

    header = f"[Large {tool_name[:80]} result compacted: {original_chars:,} characters]\n"
    warning_text = f" Warning: {warnings[0]}." if warnings else ""
    original_tokens = max(1, int(original_chars / chars_per_token))
    provisional_footer = _compaction_footer(
        original_tokens,
        original_tokens,
        "compiler-diagnostics",
        "semantic",
        location,
        warning_text,
        guidance,
    )
    content_budget = budget - len(header) - len(trailing_notice) - len(provisional_footer)
    stderr_block = ""
    if tool_name == "exec_command":
        stderr_budget = min(1800, max(400, content_budget // 4))
        stderr = _stderr_excerpt(text, stderr_budget)
        if stderr:
            stderr_block = f"[stderr excerpt]\n{stderr}\n\n"
            content_budget -= len(stderr_block)

    semantic = _semantic_reduce(envelope, text, content_budget)
    reducer = "generic"
    tier = "generic"
    notable_block = ""
    if semantic is not None:
        excerpt, reducer = semantic
        tier = "semantic"
    else:
        if tool_name == "exec_command":
            notable_budget = min(1800, max(400, content_budget // 6))
            notable = _notable_lines(text, notable_budget)
            if notable:
                notable_block = f"\n\n[Notable lines]\n{notable}"
        excerpt_budget = content_budget - len(notable_block)
        if tool_name == "read_file":
            excerpt = _sample_prefix(text, excerpt_budget)
        elif tool_name in STRUCTURED_TOOLS:
            excerpt = _sample_edges(text, excerpt_budget, head_ratio=0.7)
        else:
            excerpt = _sample_edges(text, excerpt_budget, head_ratio=0.4)

    body = f"{header}{stderr_block}{excerpt}{trailing_notice}{notable_block}"
    retained_tokens = max(1, int((len(body) + len(provisional_footer)) / chars_per_token))
    footer = _compaction_footer(
        original_tokens,
        retained_tokens,
        reducer,
        tier,
        location,
        warning_text,
        guidance,
    )
    compacted = _fit_with_footer(body, footer, budget)
    return ToolCompaction(
        text=compacted,
        compacted=True,
        reducer=reducer,
        tier=tier,
        original_chars=original_chars,
        retained_chars=len(compacted),
        artifact_path=artifact_path,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Layer 2: Context pruning
# ---------------------------------------------------------------------------


def build_tool_name_index(messages: list) -> dict[str, str]:
    """Build a tool_call_id → tool_name lookup from all assistant messages.

    Single O(n) pass replaces the per-tool-message backward walk.
    """
    index: dict[str, str] = {}
    for msg in messages:
        if get_role(msg) != "assistant":
            continue
        for tc in get_tool_calls(msg):
            tc_id = tc.get("id")
            name = tc.get("function", {}).get("name")
            if tc_id and name:
                index[tc_id] = name
    return index


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
        if get_role(m) == "user":
            start = i
            break

    remaining = KEEP_LAST_ASSISTANTS
    end = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        if get_role(messages[i]) == "assistant":
            remaining -= 1
            if remaining == 0:
                end = i
                break

    if remaining > 0:
        return 0, 0

    return start, end


def prune_context(
    messages: list,
    context_length: int,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    total_chars: int | None = None,
) -> list:
    """Prune old tool results to manage context window usage.

    Phase 1 (soft trim, 30 %): keep head + tail of large prunable results.
    Phase 2 (hard clear, 50 %): replace entire results with a placeholder.

    Returns the original list when no pruning is needed, or a new list
    with pruned messages otherwise. Pass *total_chars* to reuse an already
    computed character total and skip the full-history scan.
    """
    if not messages or context_length <= 0:
        return messages

    char_window = context_length * chars_per_token
    if total_chars is None:
        total_chars = estimate_context_chars(messages)

    if total_chars < char_window * SOFT_TRIM_RATIO:
        return messages

    start, end = _prunable_range(messages)

    tool_name_index = build_tool_name_index(messages)

    prunable: list[int] = []
    for i in range(start, end):
        msg = messages[i]
        if get_role(msg) != "tool":
            continue
        tc_id = get_tool_call_id(msg)
        name = tool_name_index.get(tc_id) if tc_id else None
        if name and name in PRUNABLE_TOOLS:
            prunable.append(i)

    if not prunable:
        return messages

    result = list(messages)

    # Phase 1: soft trim
    for i in prunable:
        text = get_text(result[i])
        trimmed = _soft_trim(text)
        if trimmed is not None:
            result[i] = set_text(result[i], trimmed)
            total_chars += len(trimmed) - len(text)

    ratio = total_chars / char_window
    if ratio < HARD_CLEAR_RATIO:
        return result

    # Phase 2: hard clear (only if enough prunable content exists)
    prunable_chars = sum(len(get_text(result[i])) for i in prunable)
    if prunable_chars < MIN_PRUNABLE_CHARS:
        return result

    for i in prunable:
        if ratio < HARD_CLEAR_RATIO:
            break
        old_len = len(get_text(result[i]))
        result[i] = set_text(result[i], HARD_CLEAR_PLACEHOLDER)
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


def needs_compaction(
    messages: list,
    context_length: int,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    total_chars: int | None = None,
) -> bool:
    char_window = context_length * chars_per_token
    if total_chars is None:
        total_chars = estimate_context_chars(messages)
    return total_chars > char_window * COMPACTION_RATIO


def _safe_split_index(messages: list, idx: int) -> int:
    """Walk backward so the split never lands inside a tool-call / result pair."""
    while idx > 0 and get_role(messages[idx]) == "tool":
        idx -= 1
    return idx


def _messages_to_text(messages: list) -> str:
    """Convert messages to a concise text for the summarization prompt."""
    parts: list[str] = []
    for msg in messages:
        role = get_role(msg)
        text = get_text(msg)
        limit = _SUMMARY_CHAR_LIMITS.get(role, 500)

        if role == "system":
            parts.append(f"[System]: {text[:limit]}")
        elif role == "user":
            parts.append(f"User: {text[:limit]}")
        elif role == "assistant":
            if text:
                parts.append(f"Assistant: {text[:limit]}")
            for tc in get_tool_calls(msg):
                name = tc.get("function", {}).get("name", "?")
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
    context_length: int = 0,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
) -> list:
    """Compact messages by LLM-summarizing the oldest portion.

    When *context_length* is provided the split point is calculated to
    bring context usage down to ~50 %.  Otherwise falls back to 40 % of
    messages.

    *messages* should **not** include the system prompt (it is managed
    separately by ContextManager).  Returns a new list.
    """
    if len(messages) <= 2:
        return list(messages)

    max_split = max(1, len(messages) - 2)  # keep the 2 latest messages

    if context_length > 0:
        char_window = context_length * chars_per_token
        total_chars = estimate_context_chars(messages)
        chars_to_free = total_chars - char_window * COMPACTION_TARGET_RATIO

        if chars_to_free > 0:
            # Walk forward accumulating chars until we've covered enough
            cumulative = 0
            split_index = 1
            for i in range(0, len(messages)):
                cumulative += msg_chars(messages[i])
                if cumulative >= chars_to_free:
                    split_index = i + 1
                    break
            else:
                split_index = max_split
        else:
            # Already under target ratio; compact a fixed fraction of oldest messages
            split_index = int(len(messages) * 0.4)
    else:
        # unknown context length, compact a fixed fraction of oldest messages
        split_index = int(len(messages) * 0.4)

    split_index = max(1, min(split_index, max_split))

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
            console.warn("Context compaction LLM call failed; preserving context", exc_info=True)
        else:
            print("[WARNING] Context compaction LLM call failed; preserving context", file=sys.stderr)
        return list(messages)

    summary_msg: dict[str, Any] = {
        "role": "user",
        "content": f"[Previous conversation summary]\n{summary}",
    }

    return [summary_msg] + to_keep

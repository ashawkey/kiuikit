"""Context management for kiui agent: tool-result filtering and compaction.

Three layers of context window management:
1. Tool-result ingress — compact oversized results before they reach history.
2. Context eviction — trim/clear old tool results once the window is half full.
3. Compaction — LLM-summarize oldest messages near the end of the window.

Layers 2 and 3 rewrite history the provider has already cached, so both are
deliberately rare and bulk: usage is measured against the last real token count
reported by the API, and an eviction pass runs only when it frees enough to pay
for the cache invalidation it causes.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from typing import Any, Callable


from kiui.agent.ui import AgentConsole
from kiui.agent.utils.interrupt import RequestInterrupted


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
        # Facts compaction has already summarized the evidence away for; see
        # CompactionState. Empty until the first pass.
        self.compaction_state = CompactionState()
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

    @property
    def total_chars(self) -> int:
        """Character total of the whole prompt payload, system prompt included.

        This is the basis token estimates are anchored on, so it must match what
        :meth:`get` actually sends. The system prompt is mutated in place on
        persona and skill changes, hence the live ``len`` rather than a cache.
        """
        return len(self.system_prompt["content"]) + self.estimated_chars

    def add(self, message: dict[str, Any]) -> None:
        self.messages.append(message)
        if self._char_cache is not None:
            self._char_cache += msg_chars(message)

    def drop_last(self, message: dict[str, Any]) -> bool:
        """Withdraw *message* if it is still the newest one. Returns whether it was.

        Identity-checked so a caller can only undo its own append: a turn that
        has already moved on must not lose someone else's message.
        """
        if not self.messages or self.messages[-1] is not message:
            return False
        self.messages.pop()
        if self._char_cache is not None:
            self._char_cache -= msg_chars(message)
        return True

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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CHARS_PER_TOKEN = 3.3  # conservative for code-heavy workloads

# Layer 1: how much of the window one tool result may occupy, as
# (window ratio, floor, ceiling). Proactive tools are called repeatedly and can
# be re-run with a narrower argument, so they get the tighter budget; anything
# else — including skill tools — cannot be re-narrowed and gets more.
#
# Every ceiling must stay strictly under the tools' own MAX_TOOL_OUTPUT_CHARS
# (24_000). A ceiling at that cap can never be exceeded, which silently turns
# ingress compaction into dead code for the tool rather than making it generous.
PROACTIVE_RESULT_BUDGET = (0.02, 2_000, 6_000)
GENERIC_RESULT_BUDGET = (0.06, 4_000, 16_000)
# Tools that earn a tier of their own. A file read is content the model asked
# for by name and usually needs whole: truncating it buys a re-read round trip
# and risks an edit against a string that was cut away, so it keeps the widest
# budget. Fetched pages and process snapshots are verbose and low-density, so
# they sit between the tiers instead of at the tool cap.
RESULT_BUDGET_OVERRIDES = {
    "read_file": (0.06, 4_000, 16_000),
    "web_fetch": (0.04, 3_000, 10_000),
    "inspect_processes": (0.04, 3_000, 10_000),
}

# Upper bound on the raw capture fed into the semantic reducers. Well above any
# ingress budget so diagnostics keep their surrounding context, but small enough
# that a pathological multi-MiB capture is not scanned in full (the reducers only
# ever emit a small excerpt and never surface the middle of a huge log).
COMPACTION_INPUT_MAX_CHARS = 512_000

PROACTIVE_TOOLS = frozenset({
    "read_file", "exec_command", "inspect_processes", "ls", "glob_files", "grep_files",
    "web_fetch", "web_search",
})
PREFIX_TOOLS = frozenset({
    "read_file", "web_fetch", "ls", "glob_files", "grep_files", "web_search",
})

_ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_PROGRESS_RE = re.compile(r"^\s*(?:[\[\]=\->#|. ]{5,}\d*%?|[⠀-⣿]+)\s*$")

_TOOL_RESULT_GUIDANCE = {
    "read_file": "Use grep_files to locate relevant lines, or read_file with a narrower offset/limit.",
    "exec_command": "Search the saved output with grep_files/read_file, or rerun with quiet flags or a targeted filter.",
    "ls": "Use glob_files with a narrower pattern instead of listing the whole directory.",
    "glob_files": "Use a narrower glob pattern or base_dir.",
    "grep_files": "Use a narrower regex, path, or file_glob.",
    "web_fetch": "Search the saved text or fetch a more specific source.",
    "web_search": "Refine the query before requesting more results.",
}

# Layer 2: eviction thresholds, as a fraction of the context window in tokens.
# Evicting rewrites already-cached history, so the window is spent rather than
# hoarded: nothing is touched below PRUNE_TRIGGER_RATIO, and a pass that cannot
# free PRUNE_MIN_YIELD_RATIO does not run at all (the reasoning behind the
# Anthropic API's `clear_at_least`). One deep pass beats a nibble every round.
PRUNE_TRIGGER_RATIO = 0.55
PRUNE_DROP_RATIO = 0.10  # how far below the trigger a single pass aims to land
PRUNE_MIN_YIELD_RATIO = 0.08

# Newest slice of the window eviction never touches. Token-denominated so it
# does not swing with message size; KEEP_LAST_ASSISTANTS is only a floor.
PROTECTED_TAIL_RATIO = 0.25
PROTECTED_TAIL_MAX_TOKENS = 40_000
KEEP_LAST_ASSISTANTS = 3

SOFT_TRIM_THRESHOLD = 3000  # trim results larger than this (head+tail ≈ 1.5K, so min savings ≈ 50%)
SOFT_TRIM_HEAD = 750
SOFT_TRIM_TAIL = 750

# Tools whose output cannot be produced again. A file read or a search can just
# be repeated, but a command runs once, with side effects, against a world that
# has since moved on. Results in the band between SOFT_TRIM_THRESHOLD and the
# ingress budget are small enough to enter history untouched yet still large
# enough for eviction to trim later, so for these tools the capture is kept on
# disk even though no ingress compaction happens.
UNREPEATABLE_TOOLS = frozenset({"exec_command"})

# Layer 3: compaction threshold. A flat ratio leaves too little room on a small
# window and compacts absurdly early on a very large one, so the trigger also
# reserves space for the model's own output plus the compaction round-trip.
COMPACTION_RATIO = 0.85
COMPACTION_RESERVE_TOKENS = 8_000
COMPACTION_MIN_TRIGGER_RATIO = 0.5  # floor, for models with a huge output cap
COMPACTION_TARGET_RATIO = 0.5  # ceiling on post-compaction usage
# How far below the trigger a pass must land. A model whose output reserve pulls
# the trigger down onto its floor puts trigger and target on the same token
# (gpt-5 reserves 128k of a 258k window, landing both on 129k), so a pass that
# hits its target lands back on the trigger and the next tool result fires
# another one. Holding the target this far under the trigger keeps the two apart
# whatever the reserve does.
COMPACTION_HYSTERESIS_RATIO = 0.15
COMPACTION_SUMMARY_MAX_CHARS = 50_000  # summarization input budget
# Ceiling on the summarization response, and the allowance used when deciding
# whether a split can free enough: the summary is the one part of a compaction
# that adds context back. Generous because reasoning tokens count against the
# same budget, and a truncated handoff is worse than a slightly long one.
COMPACTION_SUMMARY_MAX_TOKENS = 8_000

# Newest slice of the history compaction never summarizes away, so the split
# cannot swallow the work currently in progress.
COMPACTION_KEEP_RECENT_RATIO = 0.15
COMPACTION_KEEP_RECENT_MAX_TOKENS = 20_000

# Below this yield a compaction is considered ineffective and the caller should
# wait for real growth before paying for another one.
COMPACTION_MIN_YIELD_RATIO = 0.05

PRUNABLE_TOOLS = frozenset({
    "read_file", "exec_command", "web_fetch", "inspect_processes",
    "web_search", "glob_files", "grep_files", "ls",
})


# ---------------------------------------------------------------------------
# Self-calibrating token estimator
# ---------------------------------------------------------------------------


class TokenEstimator:
    """Self-calibrating chars-per-token estimator anchored on real usage.

    Starts with a conservative default and refines via exponential moving
    average each time real token counts are observed from the API. The most
    recent observation is also kept as an anchor: character counting cannot see
    tool schemas, per-message wire framing, or provider-side prompt additions,
    so only the *delta* since that observation is estimated from characters.
    """

    EMA_ALPHA = 0.15  # smoothing factor (higher = faster adaptation)
    MIN_RATIO = 2.0
    MAX_RATIO = 6.0

    def __init__(self, initial: float = DEFAULT_CHARS_PER_TOKEN):
        self._ratio = initial
        self._calibrated = False
        self._anchor: tuple[int, int] | None = None  # (prompt_chars, prompt_tokens)

    @property
    def chars_per_token(self) -> float:
        return self._ratio

    @property
    def anchored(self) -> bool:
        return self._anchor is not None

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

    def observe(self, prompt_chars: int, prompt_tokens: int) -> None:
        """Record a ground-truth prompt measurement reported by the API."""
        if prompt_tokens <= 0:
            return
        self.calibrate(prompt_chars, prompt_tokens)
        self._anchor = (prompt_chars, prompt_tokens)

    def prompt_tokens(self, prompt_chars: int) -> int:
        """Estimate the prompt tokens a request of *prompt_chars* would cost.

        *prompt_chars* must be measured on the same basis as :meth:`observe`
        (system prompt included), otherwise the anchor and the delta disagree.
        """
        if self._anchor is None:
            return self.chars_to_tokens(prompt_chars)
        anchor_chars, anchor_tokens = self._anchor
        return max(0, anchor_tokens + self.chars_to_tokens(prompt_chars - anchor_chars))

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


def get_display_text(msg: dict) -> str:
    """Return user-facing text when a message carries a separate display form."""
    display = msg.get("display_content")
    return display if isinstance(display, str) else get_text(msg)


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
        provider_state = msg.get("provider_state")
        if provider_state:
            # Responses providers replay opaque output items instead of the
            # canonical text/tool projection, so count the larger form once.
            state_chars = len(json.dumps(provider_state, ensure_ascii=False))
            chars = max(chars, state_chars)
    return chars


def estimate_context_chars(messages: list) -> int:
    return sum(msg_chars(m) for m in messages)


# ---------------------------------------------------------------------------
# Layer 1: Tool-result ingress
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolResultEnvelope:
    """Tool result plus the invocation that produced it."""

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
    original_chars: int
    retained_chars: int
    artifact_path: str | None = None


def tool_result_char_budget(
    context_length: int,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    tool_name: str = "",
) -> int:
    """Return the maximum characters one tool result may add to history."""
    ratio, min_chars, max_chars = RESULT_BUDGET_OVERRIDES.get(
        tool_name,
        PROACTIVE_RESULT_BUDGET if tool_name in PROACTIVE_TOOLS else GENERIC_RESULT_BUDGET,
    )
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


def _sample_edges(text: str, available: int) -> str:
    """Sample line-aligned head and tail excerpts within *available* chars."""
    label = "\n[... middle omitted ...]\n"
    if available <= len(label):
        return label[:available]

    content_budget = available - len(label)
    head_budget = content_budget // 2
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
    """Keep one contiguous prefix so a subsequent focused call is unambiguous."""
    label = "\n[... remainder omitted ...]\n"
    if available <= len(label):
        return label[:available]
    limit = available - len(label)
    prefix = text[:limit]
    newline = prefix.rfind("\n", int(limit * 0.7))
    if newline >= 0:
        prefix = prefix[:newline]
    return f"{prefix}{label}"


def _sample_suffix(text: str, available: int) -> str:
    """Keep one contiguous suffix for logs and other latest-first output."""
    label = "[... earlier output omitted ...]\n"
    if available <= len(label):
        return label[-available:]
    limit = available - len(label)
    suffix = text[-limit:]
    newline = suffix.find("\n", 0, int(limit * 0.3))
    if newline >= 0:
        suffix = suffix[newline + 1:]
    return f"{label}{suffix}"


def _compact_process_output(result: dict[str, Any], text: str, available: int) -> str:
    """Keep process status metadata plus the latest requested log tail.

    Falls back to *text* when the result is not the expected shape — at ingress
    that is the formatted result, during eviction the stored message body.
    """
    processes = result.get("processes")
    if not isinstance(processes, list):
        return _sample_suffix(text, available)

    status = [
        {key: value for key, value in process.items() if key != "log_tail"}
        for process in processes
    ]
    status_text = json.dumps({"processes": status}, indent=2)
    tail = next(
        (
            process["log_tail"]
            for process in processes
            if isinstance(process, dict) and process.get("log_tail")
        ),
        "",
    )
    if not tail:
        return _sample_prefix(status_text, available)

    status_header = "[process status]\n"
    tail_header = "\n\n[latest log tail]\n"
    content_budget = available - len(status_header) - len(tail_header)
    if content_budget <= 0:
        return _sample_suffix(tail, available)
    tail_budget = min(len(tail), int(content_budget * 0.65))
    status_budget = content_budget - tail_budget
    return (
        status_header
        + _sample_prefix(status_text, status_budget)
        + tail_header
        + _sample_suffix(tail, tail_budget)
    )


def _excerpt_for(
    tool_name: str, result: dict[str, Any], text: str, budget: int
) -> tuple[str, str]:
    """Excerpt *text* from whichever end carries the signal, with a label.

    The full output is already on disk under a pointer, so an excerpt only has
    to be enough to decide whether to go and read it — which end it comes from
    is the only thing worth getting right. A document or listing is read from the
    top, a process snapshot keeps its status and latest tail, and anything else
    is sampled at both ends.

    Command output is deliberately *not* tail-only: stdout and stderr share one
    pipe, and a piped child block-buffers stdout while stderr stays unbuffered,
    so a diagnostic routinely lands ahead of the bulk output it describes.

    Shared by ingress and eviction so a result is cut the same way whenever it
    is cut; the label is only used by the eviction placeholder.
    """
    if tool_name in PREFIX_TOOLS:
        return _sample_prefix(text, budget), "beginning"
    if tool_name == "inspect_processes":
        return _compact_process_output(result, text, budget), "status and latest log"
    return _sample_edges(text, budget), "beginning and end"


def _compaction_footer(
    original_tokens: int,
    retained_tokens: int,
    location: str,
    warning: str,
    guidance: str,
) -> str:
    saved_percent = max(0, min(99, round((1 - retained_tokens / original_tokens) * 100)))
    return (
        f"\n[compacted: ~{original_tokens:,}→~{retained_tokens:,} tokens, "
        f"-{saved_percent}%.{location}{warning} {guidance}]"
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
    """Compact a tool result with invocation-aware deterministic reducers.

    Call only for a result that exceeds :func:`tool_result_char_budget`; the
    caller already computes that budget to decide whether to save an artifact.
    """
    tool_name = envelope.tool_name
    raw_text = envelope.text
    budget = tool_result_char_budget(context_length, chars_per_token, tool_name)
    original_chars = envelope.original_chars
    # Preserve the first-level truncation notice (which records the true
    # pre-cap size) so prefix-sampled compaction keeps an accurate recovery
    # hint. Only prefix tools append this line at the very end; latest/edge
    # tools reconstruct their own metrics from the full artifact.
    trailing_notice = ""
    if tool_name in PREFIX_TOOLS:
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
    warning_text = (
        " Warning: recovery artifact reached its 100 MiB capture limit."
        if envelope.result.get("artifact_truncated")
        else ""
    )
    original_tokens = max(1, int(original_chars / chars_per_token))
    footer_reserve = len(_compaction_footer(
        original_tokens, original_tokens, location, warning_text, guidance,
    ))
    content_budget = budget - len(header) - len(trailing_notice) - footer_reserve

    excerpt, _ = _excerpt_for(tool_name, envelope.result, text, content_budget)
    body = f"{header}{excerpt}{trailing_notice}"
    retained_tokens = max(1, int((len(body) + footer_reserve) / chars_per_token))
    footer = _compaction_footer(
        original_tokens, retained_tokens, location, warning_text, guidance
    )
    compacted = _fit_with_footer(body, footer, budget)
    return ToolCompaction(
        text=compacted,
        original_chars=original_chars,
        retained_chars=len(compacted),
        artifact_path=artifact_path,
    )


# ---------------------------------------------------------------------------
# Layer 2: Context pruning
# ---------------------------------------------------------------------------


def build_tool_call_index(messages: list) -> dict[str, tuple[str, dict[str, Any]]]:
    """Build a tool_call_id → (tool_name, arguments) lookup in one O(n) pass."""
    index: dict[str, tuple[str, dict[str, Any]]] = {}
    for msg in messages:
        if get_role(msg) != "assistant":
            continue
        for tc in get_tool_calls(msg):
            tc_id = tc.get("id")
            function = tc.get("function", {})
            name = function.get("name")
            if not tc_id or not name:
                continue
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            index[tc_id] = (name, arguments if isinstance(arguments, dict) else {})
    return index


def build_tool_name_index(messages: list) -> dict[str, str]:
    """Build a tool_call_id → tool_name lookup from all assistant messages."""
    return {tc_id: name for tc_id, (name, _) in build_tool_call_index(messages).items()}


# Arguments that identify *what* a result describes. A later call with the same
# identity makes every earlier result for it stale, which makes those the
# cheapest eviction targets available: their content is already known to be
# superseded, so dropping them costs no capability at all.
_IDENTITY_ARGS: dict[str, tuple[str, ...]] = {
    "read_file": ("file", "offset", "limit"),
    "ls": ("path", "all"),
    "glob_files": ("pattern", "base_dir", "recursive"),
    "grep_files": ("pattern", "path", "file_glob"),
    "web_fetch": ("url",),
    "web_search": ("query",),
    "exec_command": ("command", "cwd"),
    # Per process: a snapshot of one process says nothing about another.
    "inspect_processes": ("process_id",),
}

# Writing to a file invalidates every earlier read of it, whatever range that
# read covered. These are not prunable results themselves, only invalidators.
_WRITE_TOOLS = frozenset({"write_file", "edit_file", "multi_edit", "remove_file"})

# Argument shown in an eviction placeholder so the model can still tell which
# call the cleared result came from.
_DIGEST_ARGS = ("file", "path", "command", "url", "query", "pattern")

# The ingress footer records where the full output was captured; eviction must
# carry that pointer through instead of dropping it with the rest of the text.
_ARTIFACT_PATH_RE = re.compile(r"Captured output: (.+?\.txt)\.")


def _identity_key(tool_name: str, arguments: dict[str, Any]) -> tuple | None:
    fields = _IDENTITY_ARGS.get(tool_name)
    if fields is None:
        return None
    return (tool_name, *(str(arguments.get(field, "")) for field in fields))


def _written_path(tool_name: str, arguments: dict[str, Any]) -> str:
    return str(arguments.get("file", "")) if tool_name in _WRITE_TOOLS else ""


def _call_digest(arguments: dict[str, Any]) -> str:
    for key in _DIGEST_ARGS:
        value = arguments.get(key)
        if value:
            text = str(value).replace("\n", " ")
            return text if len(text) <= 120 else f"{text[:117]}..."
    return ""


def _artifact_path_in(text: str) -> str | None:
    match = _ARTIFACT_PATH_RE.search(text)
    return match.group(1) if match else None


def _soft_trim(text: str, tool_name: str, artifact_path: str | None = None) -> str | None:
    """Trim old tool output using the same retention policy as ingress.

    *artifact_path* is re-appended when the retained excerpt does not already
    contain it: the ingress footer sits at the end of the text, so a prefix
    policy would otherwise drop the only pointer to the full output — and with
    it the ability of a later pass to clear the message down to a placeholder.
    """
    if len(text) <= SOFT_TRIM_THRESHOLD:
        return None
    retained = SOFT_TRIM_HEAD + SOFT_TRIM_TAIL
    result: dict[str, Any] = {}
    if tool_name == "inspect_processes":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None  # already compacted at ingress; excerpt the text
        if isinstance(parsed, dict):
            result = parsed
    excerpt, policy = _excerpt_for(tool_name, result, text, retained)
    trimmed = f"{excerpt}\n[Trimmed: kept {policy} {retained} chars of {len(text)} total]"
    if artifact_path and artifact_path not in trimmed:
        trimmed += f"\n[Captured output: {artifact_path}.]"
    return trimmed


def _prunable_range(
    messages: list, context_length: int, chars_per_token: float
) -> tuple[int, int]:
    """Return (start, end) indices of the zone eligible for eviction.

    The tail is protected by size rather than by message count: the newest
    PROTECTED_TAIL_RATIO of the window (capped at PROTECTED_TAIL_MAX_TOKENS) is
    always kept whole, and never fewer than KEEP_LAST_ASSISTANTS turns. Both
    constraints apply, so the earlier of the two boundaries wins.
    """
    start = 0
    for i, m in enumerate(messages):
        if get_role(m) == "user":
            start = i
            break

    remaining = KEEP_LAST_ASSISTANTS
    assistant_end = 0
    for i in range(len(messages) - 1, -1, -1):
        if get_role(messages[i]) == "assistant":
            remaining -= 1
            if remaining == 0:
                assistant_end = i
                break
    if remaining > 0:
        return 0, 0

    protected_chars = min(
        context_length * PROTECTED_TAIL_RATIO, PROTECTED_TAIL_MAX_TOKENS
    ) * chars_per_token
    used = 0
    size_end = 0
    for i in range(len(messages) - 1, -1, -1):
        used += msg_chars(messages[i])
        if used >= protected_chars:
            size_end = i
            break

    return start, min(assistant_end, size_end)


@dataclass(frozen=True)
class _Evictable:
    """A tool result in the eviction zone, with why it may already be stale."""

    index: int
    tool_name: str
    arguments: dict[str, Any]
    stale_reason: str | None


def _evictable_results(
    messages: list, start: int, end: int, call_index: dict[str, tuple[str, dict]]
) -> list[_Evictable]:
    """Collect prunable results in [start, end), flagging superseded ones.

    Identity and invalidation are tracked across the *whole* history, not just
    the eviction zone, so a recent re-read or write correctly marks an old
    result stale.
    """
    latest_identity: dict[tuple, int] = {}
    latest_write: dict[str, int] = {}
    zone: list[tuple[int, str, dict[str, Any]]] = []

    for i, msg in enumerate(messages):
        if get_role(msg) != "tool":
            continue
        tc_id = get_tool_call_id(msg)
        call = call_index.get(tc_id) if tc_id else None
        if call is None:
            continue
        name, arguments = call
        key = _identity_key(name, arguments)
        if key is not None:
            latest_identity[key] = i
        written = _written_path(name, arguments)
        if written:
            latest_write[written] = i
        if start <= i < end and name in PRUNABLE_TOOLS:
            zone.append((i, name, arguments))

    evictable: list[_Evictable] = []
    for i, name, arguments in zone:
        key = _identity_key(name, arguments)
        reason = None
        if key is not None and latest_identity.get(key, i) > i:
            reason = "superseded by a later identical call"
        elif name == "read_file":
            path = str(arguments.get("file", ""))
            if path and latest_write.get(path, i) > i:
                reason = "file modified after this read"
        evictable.append(_Evictable(i, name, arguments, reason))
    return evictable


def _cleared_text(item: _Evictable, artifact_path: str | None, reason: str) -> str:
    """Placeholder that keeps the call identity and the recovery pointer."""
    digest = _call_digest(item.arguments)
    call = f"{item.tool_name}({digest})" if digest else item.tool_name
    recovery = f" Full output: {artifact_path}." if artifact_path else ""
    return f"[cleared: {call} — {reason}.{recovery}]"


def _plan_evictions(
    messages: list, evictable: list[_Evictable], target_chars: float
) -> tuple[dict[int, str], int]:
    """Choose replacement text per message, stopping once *target_chars* is met.

    Superseded results are cleared first and unconditionally within the pass —
    their content is wrong, so keeping it costs tokens *and* misleads. The rest
    are trimmed to their per-tool retention policy, and only fully cleared when
    trimming everything still falls short and the complete output is
    recoverable from disk. (The caller may still discard the whole plan if it
    frees too little to pay for the cache invalidation.)
    """
    planned: dict[int, str] = {}
    freed = 0

    def apply(index: int, new_text: str) -> None:
        nonlocal freed
        previous = planned.get(index, get_text(messages[index]))
        saving = len(previous) - len(new_text)
        if saving <= 0:
            return
        planned[index] = new_text
        freed += saving

    fresh: list[_Evictable] = []
    for item in evictable:
        if item.stale_reason is None:
            fresh.append(item)
            continue
        text = get_text(messages[item.index])
        apply(item.index, _cleared_text(item, _artifact_path_in(text), item.stale_reason))

    for item in fresh:
        if freed >= target_chars:
            return planned, freed
        text = get_text(messages[item.index])
        trimmed = _soft_trim(text, item.tool_name, _artifact_path_in(text))
        if trimmed is not None:
            apply(item.index, trimmed)

    for item in fresh:
        if freed >= target_chars:
            break
        text = get_text(messages[item.index])
        artifact_path = _artifact_path_in(text)
        if artifact_path:
            apply(item.index, _cleared_text(item, artifact_path, "evicted to free context"))

    return planned, freed


def eviction_trigger_tokens(context_length: int, max_output_tokens: int = 0) -> int:
    """Token usage at which eviction starts — always below the compaction trigger.

    Eviction is deterministic and keeps a recovery pointer; compaction costs an
    LLM round-trip and discards detail, so layer 2 must get a pass first. A model
    whose output reserve is large relative to its window (gpt-5: 128k of 258k)
    pulls the compaction trigger down to its floor, below a flat 55 %; the
    eviction trigger follows it down instead of being stranded above it.
    """
    compaction = compaction_trigger_tokens(context_length, max_output_tokens)
    return int(min(
        context_length * PRUNE_TRIGGER_RATIO,
        compaction - context_length * PRUNE_MIN_YIELD_RATIO,
    ))


def prune_context(
    messages: list,
    context_length: int,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    used_tokens: int | None = None,
    max_output_tokens: int = 0,
) -> list:
    """Evict old tool results once the context window is over half full.

    Eviction rewrites messages the provider has already cached, which costs a
    full prefix re-read on the next request. The whole pass is therefore gated
    on freeing at least PRUNE_MIN_YIELD_RATIO of the window, so the cost is
    paid rarely and in bulk rather than a little every round.

    Returns the original list when nothing is evicted. Pass *used_tokens* to
    reuse the live usage estimate instead of re-deriving it from characters.
    """
    if not messages or context_length <= 0:
        return messages

    if used_tokens is None:
        used_tokens = int(estimate_context_chars(messages) / chars_per_token)
    trigger = eviction_trigger_tokens(context_length, max_output_tokens)
    if used_tokens < trigger:
        return messages

    start, end = _prunable_range(messages, context_length, chars_per_token)
    if start >= end:
        return messages

    evictable = _evictable_results(
        messages, start, end, build_tool_call_index(messages)
    )
    if not evictable:
        return messages

    min_yield_chars = context_length * PRUNE_MIN_YIELD_RATIO * chars_per_token
    target_tokens = trigger - context_length * PRUNE_DROP_RATIO
    need_chars = (used_tokens - target_tokens) * chars_per_token
    planned, freed = _plan_evictions(
        messages, evictable, max(need_chars, min_yield_chars)
    )

    if freed < min_yield_chars:
        return messages  # not worth invalidating the cached prefix

    result = list(messages)
    for index, text in planned.items():
        result[index] = set_text(result[index], text)
    return result


# ---------------------------------------------------------------------------
# Layer 3: Compaction
# ---------------------------------------------------------------------------

SUMMARY_MARKER = "[Previous conversation summary]"

# Headings the emitted summary uses. The deterministic ones are parsed back out
# on the next compaction rather than trusted to survive re-summarization.
_ORIGINAL_REQUEST_HEADING = "## Original request"
_FILES_HEADING = "## Files"
_SKILLS_HEADING = "## Loaded skills"

# Per-message excerpt limits when building the summarization input. Generous
# because the total is capped by COMPACTION_SUMMARY_MAX_CHARS, and recency is
# already handled by _fit_summary_input dropping from the middle.
_SUMMARY_MESSAGE_CHARS = 800
_SUMMARY_TOOL_RESULT_CHARS = 500

# A prior summary stands in for everything before it, so it is passed to the
# summarizer separately with a far larger budget than an ordinary message.
_PRIOR_SUMMARY_CHAR_LIMIT = 12_000

# The original request is reproduced verbatim rather than re-summarized.
_FIRST_REQUEST_CHAR_LIMIT = 4_000

# Files are carried as a list, never as contents: re-reading them here is what
# makes compaction refill the window and cascade into compacting again.
_SUMMARY_FILE_LIST_LIMIT = 25

_SUMMARY_FORMAT = """\
## Goal
[What the user is trying to accomplish. Several items if the session covered several tasks.]

## Constraints & Preferences
- [Requirements, preferences, or constraints the user stated, or "(none)"]

## Progress
### Done
- [Work that is finished]
### In Progress
- [What is being worked on right now]
### Blocked
- [Anything preventing progress, or "(none)"]

## Key Decisions
- **[Decision]**: [Why it was made]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Data, error text, or references needed to continue, or "(none)"]"""

_SUMMARY_RULES = """\
Preserve exact file paths, function and class names, commands, and error text
verbatim — never paraphrase an identifier. Keep each section concise."""

_COMPACTION_PROMPT = """\
You are compacting a conversation between a user and an AI coding assistant.
Write a handoff summary that another assistant will use to continue this work
without access to the raw history.

Use this EXACT section structure:

{sections}

{rules}

<conversation>
{conversation}
</conversation>"""

_COMPACTION_UPDATE_PROMPT = """\
You are updating an existing handoff summary with newer conversation history.

Rules:
- PRESERVE every still-relevant fact from the previous summary. The raw history
  behind it is already gone and cannot be recovered if you drop it.
- ADD progress, decisions, and context from the new messages.
- MOVE items from "In Progress" to "Done" as they complete, and update
  "Next Steps" to match the current state.
- REMOVE only what is genuinely obsolete.

Use this EXACT section structure:

{sections}

{rules}

<previous-summary>
{previous}
</previous-summary>

<conversation>
{conversation}
</conversation>"""


def compaction_trigger_tokens(context_length: int, max_output_tokens: int = 0) -> int:
    """Token usage at which compaction has to run.

    Reserves room for the model's own output plus the compaction round-trip on
    top of a plain ratio, so the trigger stays safe on a small window without
    firing absurdly early on a very large one.
    """
    reserve = max_output_tokens + COMPACTION_RESERVE_TOKENS
    return int(max(
        context_length * COMPACTION_MIN_TRIGGER_RATIO,
        min(context_length * COMPACTION_RATIO, context_length - reserve),
    ))


def compaction_target_tokens(context_length: int, max_output_tokens: int = 0) -> int:
    """Token usage a compaction pass aims to land on.

    Held below the trigger rather than set by ratio alone, so a pass that hits
    its target cannot be re-triggered by the very next tool result. The ratio
    still binds on models where the trigger sits high enough on its own, which
    keeps a single pass deep rather than turning one big compaction into several
    shallow ones.
    """
    return int(min(
        context_length * COMPACTION_TARGET_RATIO,
        compaction_trigger_tokens(context_length, max_output_tokens)
        - context_length * COMPACTION_HYSTERESIS_RATIO,
    ))


def needs_compaction(
    messages: list,
    context_length: int,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    used_tokens: int | None = None,
    max_output_tokens: int = 0,
) -> bool:
    if context_length <= 0:
        return False
    if used_tokens is None:
        used_tokens = int(estimate_context_chars(messages) / chars_per_token)
    return used_tokens > compaction_trigger_tokens(context_length, max_output_tokens)


def _safe_split_index(messages: list, idx: int) -> int:
    """Walk backward so the split never lands inside a tool-call / result pair."""
    while idx > 0 and get_role(messages[idx]) == "tool":
        idx -= 1
    return idx


def _is_summary(msg: dict) -> bool:
    return get_role(msg) == "user" and get_text(msg).startswith(SUMMARY_MARKER)


def _render_summary_input(messages: list) -> list[str]:
    """Render each message for the summarizer, one entry per message.

    A prior summary is skipped: it is handed to the summarizer separately so it
    can be updated rather than re-compressed.
    """
    parts: list[str] = []
    for msg in messages:
        if _is_summary(msg):
            continue
        role = get_role(msg)
        text = get_text(msg)
        limit = _SUMMARY_TOOL_RESULT_CHARS if role == "tool" else _SUMMARY_MESSAGE_CHARS

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
    return parts


def _fit_summary_input(parts: list[str], budget: int) -> str:
    """Fit rendered messages into *budget*, dropping from the middle.

    Truncating the tail would discard the most recent state — exactly what the
    summary exists to carry forward — so the oldest and newest messages are
    kept and the middle is elided. The first entry is always retained because
    it anchors what the session set out to do.
    """
    if sum(len(part) + 1 for part in parts) <= budget:
        return "\n".join(parts)

    marker = "\n[... middle of the conversation omitted ...]\n"
    head = [parts[0]]
    tail: list[str] = []
    used = len(marker) + len(parts[0]) + 1
    low, high = 1, len(parts) - 1

    step = 0
    while low <= high:
        # Two of every three entries come from the newest end.
        from_tail = step % 3 != 2
        index = high if from_tail else low
        cost = len(parts[index]) + 1
        if used + cost > budget:
            break
        if from_tail:
            tail.insert(0, parts[index])
            high -= 1
        else:
            head.append(parts[index])
            low += 1
        used += cost
        step += 1

    return "\n".join(head) + marker + "\n".join(tail)


def _merge_recent(previous: list[str], current: list[str]) -> list[str]:
    """Combine two last-touch-ordered lists, newest last, bounded in length."""
    merged = {name: None for name in previous}
    for name in current:
        merged.pop(name, None)
        merged[name] = None
    return list(merged)[-_SUMMARY_FILE_LIST_LIMIT:]


def _collect_tool_usage(messages: list) -> tuple[list[str], list[str], list[str]]:
    """Return (files read, files modified, skills loaded) in last-touch order."""
    read: dict[str, None] = {}
    modified: dict[str, None] = {}
    skills: dict[str, None] = {}

    for msg in messages:
        if get_role(msg) != "assistant":
            continue
        for tc in get_tool_calls(msg):
            function = tc.get("function", {})
            name = function.get("name")
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(arguments, dict):
                continue
            if name == "load_skill":
                skill = str(arguments.get("name", ""))
                if skill:
                    skills.pop(skill, None)
                    skills[skill] = None
                continue
            path = str(arguments.get("file", ""))
            if not path:
                continue
            if name in _WRITE_TOOLS:
                read.pop(path, None)
                modified.pop(path, None)
                modified[path] = None
            elif name == "read_file" and path not in modified:
                read.pop(path, None)
                read[path] = None

    return list(read), list(modified), list(skills)


@dataclass(frozen=True)
class CompactionState:
    """Facts that have to outlive the messages they were derived from.

    Compaction deletes the tool calls and the opening user turn, so a later pass
    can no longer see who asked for what or which files were touched. Carrying
    that as data — rather than writing it into the summary prose and parsing it
    back out on the next pass — keeps it exact: a model cannot drop a path it
    never had to reproduce, and a heading it invents cannot shadow ours.
    """

    original_request: str = ""
    read_files: tuple[str, ...] = ()
    modified_files: tuple[str, ...] = ()
    skills: tuple[str, ...] = ()

    def absorb(self, messages: list) -> CompactionState:
        """Fold the messages about to be summarized into the carried state."""
        read, modified, skills = _collect_tool_usage(messages)
        modified_files = _merge_recent(list(self.modified_files), modified)
        written = set(modified_files)
        return CompactionState(
            # The first user turn defines the task, and only the first: a later
            # pass must not relabel the session with a mid-session instruction.
            original_request=self.original_request or _first_user_text(messages),
            read_files=tuple(
                path for path in _merge_recent(list(self.read_files), read)
                if path not in written
            ),
            modified_files=tuple(modified_files),
            skills=tuple(_merge_recent(list(self.skills), skills)),
        )

    def render(self) -> list[str]:
        """Render the carried facts as trailing sections of a summary message."""
        lines: list[str] = []
        if self.read_files or self.modified_files:
            lines += ["", _FILES_HEADING]
            if self.modified_files:
                lines.append(f"- Modified: {', '.join(self.modified_files)}")
            if self.read_files:
                lines.append(f"- Read: {', '.join(self.read_files)}")
            lines.append("- Contents are not included; re-read a file before relying on it.")
        if self.skills:
            lines += [
                "",
                _SKILLS_HEADING,
                f"- Active: {', '.join(self.skills)}",
                "- Call load_skill again if you need the full instructions.",
            ]
        return lines


def _first_user_text(messages: list) -> str:
    """The opening request, skipping a summary standing in for earlier turns."""
    for msg in messages:
        if get_role(msg) == "user" and not _is_summary(msg):
            return get_text(msg)[:_FIRST_REQUEST_CHAR_LIMIT]
    return ""


def _build_summary_message(summary: str, state: CompactionState) -> dict[str, Any]:
    """Assemble the replacement message, carried sections included."""
    lines = [
        SUMMARY_MARKER,
        "The earlier conversation was compacted. This is a system-generated "
        "handoff, not a new instruction from the user.",
    ]
    if state.original_request:
        lines += ["", _ORIGINAL_REQUEST_HEADING, state.original_request]
    lines += ["", summary]
    lines += state.render()
    return {"role": "user", "content": "\n".join(lines)}


def _keep_recent_split_limit(
    messages: list, context_length: int, chars_per_token: float
) -> int:
    """Highest split index that leaves the protected recent window intact.

    Returns 0 when the whole history fits inside that window — there is then
    nothing old enough to summarize, however full the context window is.
    """
    keep_chars = min(
        context_length * COMPACTION_KEEP_RECENT_RATIO,
        COMPACTION_KEEP_RECENT_MAX_TOKENS,
    ) * chars_per_token
    used = 0
    for i in range(len(messages) - 1, -1, -1):
        used += msg_chars(messages[i])
        if used >= keep_chars:
            return i
    return 0


def _yields_enough(
    messages: list, split_index: int, context_length: int, chars_per_token: float
) -> bool:
    """Whether summarizing ``messages[:split_index]`` is worth the round-trip.

    What a pass really frees is the compacted messages *minus* the summary it
    writes back. A split that nets less than COMPACTION_MIN_YIELD_RATIO is one
    the caller's floor would suppress on the next round anyway, so it is cheaper
    to notice here than after paying for the summarization.
    """
    freed = sum(msg_chars(m) for m in messages[:split_index])
    written_back = COMPACTION_SUMMARY_MAX_TOKENS * chars_per_token
    min_yield = context_length * COMPACTION_MIN_YIELD_RATIO * chars_per_token
    return freed - written_back >= min_yield


def compact_context(
    messages: list,
    summarize: Callable[[str], str],
    console: AgentConsole | None = None,
    context_length: int = 0,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    used_tokens: int | None = None,
    max_output_tokens: int = 0,
    state: CompactionState = CompactionState(),
) -> tuple[list, CompactionState]:
    """Compact messages by LLM-summarizing the oldest portion.

    When *context_length* is provided the split point is calculated to bring
    usage down to :func:`compaction_target_tokens`. Otherwise falls back to 40 %
    of messages.

    *messages* should **not** include the system prompt (it is managed
    separately by ContextManager). *summarize* receives the prepared prompt
    and returns summary text through the active provider. *state* carries the
    facts earlier passes summarized away; the caller keeps the one returned.

    Returns a new list, or *messages* itself when no compaction happened —
    nothing old enough to summarize, too little to be worth the round-trip, or a
    failed summarization. Callers rely on that identity to tell a pass that
    shrank the context from one that changed nothing.
    """
    if len(messages) <= 2:
        return messages, state

    max_split = max(1, len(messages) - 2)  # keep the 2 latest messages
    if context_length > 0:
        # The protected recent window can cover the entire history — a short
        # session, or one whose usage is dominated by the system prompt and
        # tool schemas rather than by messages. Summarizing anyway would rewrite
        # the oldest message for the price of a round-trip and grow the context,
        # so report nothing to do and let the caller decide.
        max_split = min(
            max_split,
            _keep_recent_split_limit(messages, context_length, chars_per_token),
        )
        if max_split < 1:
            return messages, state

    if context_length > 0:
        if used_tokens is None:
            used_tokens = int(estimate_context_chars(messages) / chars_per_token)
        # Only message characters can be freed, but the overhead the anchor
        # accounts for (system prompt, tool schemas) still has to come out of
        # the target — so convert the token shortfall back into characters.
        chars_to_free = (
            used_tokens - compaction_target_tokens(context_length, max_output_tokens)
        ) * chars_per_token

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
    if split_index <= 0:
        # Walking off the front means the whole history is one tool-call group.
        # Summarizing nothing would prepend an invented summary and *grow* the
        # context, so leave it to the caller (or the API) to fail loudly.
        return messages, state

    if context_length > 0 and not _yields_enough(
        messages, split_index, context_length, chars_per_token
    ):
        return messages, state

    to_compact = messages[:split_index]
    to_keep = messages[split_index:]

    previous = next((get_text(m) for m in to_compact if _is_summary(m)), "")
    conversation = _fit_summary_input(
        _render_summary_input(to_compact), COMPACTION_SUMMARY_MAX_CHARS
    )
    if previous:
        prompt = _COMPACTION_UPDATE_PROMPT.format(
            sections=_SUMMARY_FORMAT,
            rules=_SUMMARY_RULES,
            previous=previous[:_PRIOR_SUMMARY_CHAR_LIMIT],
            conversation=conversation,
        )
    else:
        prompt = _COMPACTION_PROMPT.format(
            sections=_SUMMARY_FORMAT,
            rules=_SUMMARY_RULES,
            conversation=conversation,
        )

    try:
        summary = summarize(prompt).strip()
    except RequestInterrupted:
        raise  # the user cancelled; the caller unwinds, nothing failed
    except Exception:
        if console is not None:
            console.warn("Context compaction LLM call failed; preserving context", exc_info=True)
        else:
            print("[WARNING] Context compaction LLM call failed; preserving context", file=sys.stderr)
        return messages, state

    state = state.absorb(to_compact)
    return [_build_summary_message(summary, state)] + to_keep, state

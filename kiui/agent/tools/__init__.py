"""Built-in tool definitions and executor for kiui agent.

The package re-exports the historical ``kiui.agent.tools`` API while keeping
each tool family in a focused implementation module.
"""

from .files import _human_size, _normalize_ws, apply_edit, find_match
from .search import _decode_bytes
from .constants import (
    EXEC_READER_JOIN_TIMEOUT,
    MAX_EXEC_ARTIFACT_BYTES,
    MAX_EXEC_OUTPUT_CHARS,
    MAX_GLOB_RESULTS,
    MAX_GREP_MATCHES,
    MAX_PROCESS_LOG_BYTES,
    MAX_READ_BYTES,
    MAX_READ_LINES,
    MAX_STREAMING_BUFFER_CHARS,
    MAX_WEB_FETCH_BYTES,
    MAX_WEB_FETCH_CHARS,
    MAX_WEB_REDIRECTS,
    IPV6_TRANSITION_NETWORKS as _IPV6_TRANSITION_NETWORKS,
    SKIP_DIRS as _SKIP_DIRS,
)
from .formatting import (
    TOOL_SUMMARY_MAX_CHARS,
    TOOL_SUMMARY_MAX_LINES,
    format_tool_result,
    format_tool_summary,
)
from .web import _resolve_public_addresses
from .schemas import get_tool_definitions
from .executor import ToolExecutor

__all__ = [
    "EXEC_READER_JOIN_TIMEOUT",
    "MAX_EXEC_ARTIFACT_BYTES",
    "MAX_EXEC_OUTPUT_CHARS",
    "MAX_GLOB_RESULTS",
    "MAX_GREP_MATCHES",
    "MAX_PROCESS_LOG_BYTES",
    "MAX_READ_BYTES",
    "MAX_READ_LINES",
    "MAX_STREAMING_BUFFER_CHARS",
    "MAX_WEB_FETCH_BYTES",
    "MAX_WEB_FETCH_CHARS",
    "MAX_WEB_REDIRECTS",
    "TOOL_SUMMARY_MAX_CHARS",
    "TOOL_SUMMARY_MAX_LINES",
    "ToolExecutor",
    "apply_edit",
    "find_match",
    "format_tool_result",
    "format_tool_summary",
    "get_tool_definitions",
]

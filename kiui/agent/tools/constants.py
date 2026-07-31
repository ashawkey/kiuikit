"""Limits and shared constants for built-in tools."""

import ipaddress

MAX_TOOL_OUTPUT_CHARS = 24_000
MAX_READ_LINES = 1000
MAX_READ_BYTES = MAX_TOOL_OUTPUT_CHARS  # Backward-compatible alias.
MAX_EXEC_OUTPUT_CHARS = MAX_TOOL_OUTPUT_CHARS
MAX_STREAMING_BUFFER_CHARS = 1_000_000
MAX_EXEC_ARTIFACT_BYTES = 100 * 1024 * 1024
MAX_PROCESS_LOG_BYTES = 100 * 1024 * 1024
MAX_PROCESS_LOG_TAIL_CHARS = MAX_TOOL_OUTPUT_CHARS
EXEC_READER_JOIN_TIMEOUT = 5
# How often streamed command output is flushed to the terminal, and how many
# lines a single raw block write may carry. Raw block writes keep the cost low
# even for high-volume output, and every line is streamed to the terminal (the
# full output also reaches the artifact file).
EXEC_DISPLAY_FLUSH_SECONDS = 0.1
EXEC_DISPLAY_FLUSH_LINES = 200
MAX_WEB_FETCH_CHARS = MAX_TOOL_OUTPUT_CHARS
MAX_WEB_FETCH_BYTES = 2 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024  # read_image payload cap (base64 expands ~1.37x)
MAX_WEB_REDIRECTS = 5
MAX_GLOB_RESULTS = 500
GLOB_TIMEOUT_SECONDS = 15
GREP_TIMEOUT_SECONDS = 30
MAX_GREP_MATCHES = 200

SKIP_DIRS = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".tox", "dist", "build", ".mypy_cache", ".pytest_cache",
})
IPV6_TRANSITION_NETWORKS = (
    ipaddress.ip_network("64:ff9b::/96"),
    ipaddress.ip_network("64:ff9b:1::/48"),
    ipaddress.ip_network("2001::/32"),  # Teredo
    ipaddress.ip_network("2002::/16"),  # 6to4
)

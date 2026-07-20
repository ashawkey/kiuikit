"""Limits and shared constants for built-in tools."""

import ipaddress

MAX_READ_LINES = 1000
MAX_READ_BYTES = 24_000
MAX_EXEC_OUTPUT_CHARS = 24_000
MAX_STREAMING_BUFFER_CHARS = 1_000_000
MAX_EXEC_ARTIFACT_BYTES = 100 * 1024 * 1024
MAX_PROCESS_LOG_BYTES = 100 * 1024 * 1024
EXEC_READER_JOIN_TIMEOUT = 5
MAX_WEB_FETCH_CHARS = 20_000
MAX_WEB_FETCH_BYTES = 2 * 1024 * 1024
MAX_WEB_REDIRECTS = 5
MAX_GLOB_RESULTS = 500
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

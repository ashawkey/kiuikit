"""Unified user-facing I/O for the kiui agent.

All console output, confirmation prompts, and log messages go through
AgentConsole so that theming, stderr redirection (pipe mode), and
interactive prompting are handled in one place.
"""

from __future__ import annotations

from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.theme import Theme


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")

AGENT_THEME = Theme({
    "debug": "dim cyan",
    "input": "bold yellow",
    "response": "bold green",
    "error": "bold red",
    "warning": "bold yellow",
    "system": "bold blue",
    "tool": "bold cyan",
    "tool_ok": "dim green",
    "tool_fail": "red",
})


class AgentConsole:
    """Thin wrapper around ``rich.Console`` with typed convenience methods."""

    def __init__(self):
        self._console = Console(theme=AGENT_THEME)

    # -- raw pass-through (for rich markup, tables, etc.) -------------------

    def print(self, *args, **kwargs):
        self._console.print(*args, **kwargs)

    def table(self, table: Table):
        self._console.print(table)

    # -- typed log helpers --------------------------------------------------

    def system(self, msg: str):
        self._console.print(f"[{_now()} SYSTEM] {msg}", style="system", markup=False)

    def debug(self, msg: str):
        self._console.print(f"[{_now()} DEBUG] {msg}", style="debug", markup=False)

    def error(self, msg: str):
        self._console.print(f"[{_now()} ERROR] {msg}", style="error", markup=False)

    def warn(self, msg: str, *, exc_info: bool = False):
        self._console.print(f"[{_now()} WARNING] {msg}", style="warning", markup=False)
        if exc_info:
            self._console.print_exception()

    def tool(self, msg: str):
        self._console.print(f"[{_now()} TOOL] {msg}", style="tool")

    def tool_result(self, msg: str, success: bool = True):
        style = "tool_ok" if success else "tool_fail"
        prefix = "  \u2713 " if success else "  \u2717 "
        lines = msg.splitlines()
        first = prefix + (lines[0] if lines else "")
        rest = "\n".join("    " + line for line in lines[1:])
        output = first + ("\n" + rest if rest else "")
        self._console.print(output, style=style, markup=False)

    def response(self, msg: str):
        self._console.print(f"[{_now()} RESP] {msg}", style="response", markup=False)

    # -- interactive prompts ------------------------------------------------

    def confirm(self, prompt: str, options: str = "[y]es / [n]o / [a]lways") -> str:
        """Display *prompt* with rich styling, then read a single-line answer."""
        self._console.print(prompt, highlight=False)
        return input(f"   {options}: ").strip().lower()

"""Lightweight permission control for tool execution.

Three modes:
  - auto:    all tools run without prompting (pipe/subagent mode)
  - default: risky tools prompt the user for confirmation
  - strict:  every tool prompts for confirmation
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from kiui.agent.ui import AgentConsole


class PermissionMode(str, Enum):
    AUTO = "auto"
    DEFAULT = "default"
    STRICT = "strict"


SAFE_TOOLS = frozenset({
    "read_file",
    "glob_files",
    "grep_files",
    "web_search",
    "web_fetch",
    "list_subagents",
})

RISKY_TOOLS = frozenset({
    "write_file",
    "edit_file",
    "exec_command",
    "remove_file",
    "spawn_subagent",
    "kill_subagent",
    "send_to_subagent",
})


def _summarize_call(name: str, args: dict[str, Any]) -> str:
    """Build a short human-readable summary of a tool call."""
    if name == "exec_command":
        return f"exec_command: {args.get('command', '?')}"
    if name in ("write_file", "edit_file", "read_file", "remove_file"):
        return f"{name}: {args.get('path', '?')}"
    if name == "spawn_subagent":
        task = args.get("task", "")
        return f"spawn_subagent: {task[:80]}"
    if name == "kill_subagent":
        return f"kill_subagent: {args.get('run_id', '?')}"
    if name == "send_to_subagent":
        return f"send_to_subagent -> {args.get('target', '?')}"
    return name


class PermissionController:
    """Gate tool execution behind user confirmation when required.

    Responses at the prompt:
      y  — allow this call
      n  — deny this call
      a  — allow this tool for the rest of the session
    """

    def __init__(
        self,
        mode: PermissionMode = PermissionMode.DEFAULT,
        console: AgentConsole | None = None,
    ):
        self.mode = mode
        self.console = console or AgentConsole()
        self._session_allowed: set[str] = set()

    def check(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        """Return True if the tool call is permitted, False to deny."""
        if self.mode == PermissionMode.AUTO:
            return True

        needs_prompt = self._needs_prompt(tool_name)
        if not needs_prompt:
            return True

        if tool_name in self._session_allowed:
            return True

        return self._prompt_user(tool_name, arguments)

    def _needs_prompt(self, tool_name: str) -> bool:
        if self.mode == PermissionMode.STRICT:
            return True
        # default mode: only risky tools
        return tool_name in RISKY_TOOLS

    def _prompt_user(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        summary = _summarize_call(tool_name, arguments)
        try:
            response = self.console.confirm(
                f"\n[bold yellow]⚠  Permission required[/bold yellow]"
                f"\n   [cyan]{summary}[/cyan]",
                options="Allow? [y]es / [n]o / [a]lways",
            )
        except (EOFError, KeyboardInterrupt):
            self.console.print("   [red]Denied (interrupted).[/red]")
            return False

        if response in ("a", "always"):
            self._session_allowed.add(tool_name)
            self.console.print(f"   [green]✓ {tool_name} allowed for this session.[/green]")
            return True
        if response in ("y", "yes", ""):
            return True

        self.console.print("   [red]✗ Denied.[/red]")
        return False

"""Lightweight permission control for tool execution.

Three modes:
  - auto:    all tools run without prompting (pipe/subagent mode)
  - default: risky tools prompt the user for confirmation
  - strict:  every tool prompts for confirmation

A hard safety layer (`SafetyGuard`) runs *before* mode-based checks and blocks
inherently dangerous operations regardless of mode:
  - Destructive shell commands (rm -rf /, mkfs, dd to devices, fork bombs, …)

Files outside the working directory trigger a warning prompt in interactive
modes (default/strict) so the user can choose to proceed.  In auto mode
they are still blocked since there is no user to ask.
"""

from __future__ import annotations

import os
import re
import sys
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

from kiui.agent.ui import AgentConsole


class PermissionMode(str, Enum):
    AUTO = "auto"
    DEFAULT = "default"
    STRICT = "strict"


SAFE_TOOLS = frozenset({
    "read_file",
    "ls",
    "glob_files",
    "grep_files",
    "web_search",
    "web_fetch",
    "load_skill",
})

RISKY_TOOLS = frozenset({
    "write_file",
    "edit_file",
    "multi_edit",
    "exec_command",
    "remove_file",
    "spawn_subagent",
})


# ---------------------------------------------------------------------------
# Safety guard
# ---------------------------------------------------------------------------

class SafetyGuard:
    """Hard safety layer that blocks dangerous operations regardless of mode.

    Enforced rules (hard-blocked, no override):
      - Shell commands must not match known destructive patterns
        (mkfs, dd to devices, rm -rf on critical paths, fork bombs, etc.)

    Path-containment checks are available via :meth:`check_path` and are
    *not* hard-blocked — the caller decides whether to override or prompt.
    """

    # -- Unix dangerous command patterns ------------------------------------

    _DANGEROUS_UNIX_PATTERNS: ClassVar[list[tuple[re.Pattern[str], str]]] = [
        (re.compile(r"(?:sudo\s+)?mkfs(?:\.\w+)?\s"),
         "filesystem format (mkfs)"),
        (re.compile(r"(?:sudo\s+)?dd\s+.*\bof=/dev/"),
         "dd write to block device"),
        (re.compile(r">\s*/dev/[sh]d"),
         "redirect to block device"),
        (re.compile(r":\(\)\s*\{"),
         "fork bomb"),
        (re.compile(r"(?:sudo\s+)?(?:shutdown|reboot|halt|poweroff)\b"),
         "system shutdown/reboot"),
    ]

    # -- Windows dangerous command patterns ---------------------------------

    _DANGEROUS_WIN_PATTERNS: ClassVar[list[tuple[re.Pattern[str], str]]] = [
        (re.compile(r"\bformat\s+[a-zA-Z]:", re.IGNORECASE),
         "format drive"),
        (re.compile(r"\bdiskpart\b", re.IGNORECASE),
         "diskpart command"),
    ]

    # Paths that should never be the target of recursive forced deletion.
    _CRITICAL_UNIX_PATHS = frozenset({
        "/", "/bin", "/boot", "/dev", "/etc", "/home", "/lib", "/lib64",
        "/media", "/mnt", "/opt", "/proc", "/root", "/run", "/sbin",
        "/snap", "/srv", "/sys", "/tmp", "/usr", "/var",
    })

    # Regex that splits on unquoted shell operators (; & | && ||)
    # while skipping over single- and double-quoted strings.
    _SHELL_SPLIT_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"""'[^']*'|"[^"]*"|([;&|]+)""", re.VERBOSE
    )

    def __init__(self, work_dir: str | Path | None = None):
        self._work_dir = Path(work_dir or os.getcwd()).resolve()

    @property
    def work_dir(self) -> Path:
        return self._work_dir

    @classmethod
    def _split_command(cls, command: str) -> list[str]:
        """Split a shell command on unquoted ; & | operators."""
        segments: list[str] = []
        last = 0
        for m in cls._SHELL_SPLIT_RE.finditer(command):
            if m.group(1) is not None:
                segments.append(command[last:m.start()])
                last = m.end()
        segments.append(command[last:])
        return segments

    def check(self, tool_name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
        """Return ``(allowed, reason)``.  *reason* is non-empty when blocked.

        Only performs *hard* checks that should never be skipped:
        destructive shell commands, critical rm/chmod patterns, etc.

        Path-containment checks (out-of-scope file edits) are handled
        separately via :meth:`check_path` so the user can override them.
        """
        if tool_name == "exec_command":
            return self._check_command(arguments.get("command", ""))
        return True, ""

    def check_path(self, path: str) -> tuple[bool, str]:
        """Check whether *path* is contained within the working directory.

        Returns ``(True, "")`` when safe, or ``(False, reason)`` when the
        path resolves outside the allowed work directory.

        Unlike the hard safety checks in :meth:`check`, denials from this
        method are *not* automatically fatal — the caller can surface them
        as a user-overridable prompt.
        """
        return self._check_path(path)

    # -- path containment ---------------------------------------------------

    def _check_path(self, path: str) -> tuple[bool, str]:
        if not path:
            return False, "Empty file path."
        try:
            p = Path(path).expanduser()
            if not p.is_absolute():
                p = self._work_dir / p
            resolved = p.resolve()
        except (OSError, ValueError) as e:
            return False, f"Invalid path: {e}"
        try:
            resolved.relative_to(self._work_dir)
        except ValueError:
            return False, (
                f"Path '{path}' resolves outside the allowed working "
                f"directory ({self._work_dir})."
            )
        return True, ""

    # -- command analysis ---------------------------------------------------

    def _check_command(self, command: str) -> tuple[bool, str]:
        patterns = (
            self._DANGEROUS_WIN_PATTERNS
            if sys.platform == "win32"
            else self._DANGEROUS_UNIX_PATTERNS
        )
        for pat, desc in patterns:
            if pat.search(command):
                return False, f"Blocked: {desc}."

        if sys.platform != "win32":
            reason = self._check_rm(command)
            if reason:
                return False, f"Blocked: {reason}."
            reason = self._check_recursive_perm(command)
            if reason:
                return False, f"Blocked: {reason}."

        return True, ""

    def _check_rm(self, command: str) -> str | None:
        """Detect dangerous ``rm -rf`` targeting critical system paths."""
        home = os.path.expanduser("~")
        for seg in self._split_command(command):
            tokens = seg.split()
            if not tokens:
                continue
            idx = 0
            while idx < len(tokens) and tokens[idx] in ("sudo", "doas"):
                idx += 1
            if idx >= len(tokens) or tokens[idx] != "rm":
                continue

            has_r = has_f = False
            targets: list[str] = []
            for tok in tokens[idx + 1:]:
                if tok.startswith("-"):
                    if tok == "--recursive":
                        has_r = True
                    elif tok == "--force":
                        has_f = True
                    elif tok == "--no-preserve-root":
                        return "rm --no-preserve-root"
                    elif not tok.startswith("--"):
                        if "r" in tok:
                            has_r = True
                        if "f" in tok:
                            has_f = True
                else:
                    targets.append(tok)

            if not (has_r and has_f):
                continue

            for target in targets:
                norm = os.path.normpath(os.path.expanduser(target))
                if norm in self._CRITICAL_UNIX_PATHS:
                    return f"rm -rf on critical system path ({norm})"
                if norm == home:
                    return "rm -rf on home directory"

        return None

    def _check_recursive_perm(self, command: str) -> str | None:
        """Detect recursive chmod/chown on the root filesystem."""
        for seg in self._split_command(command):
            tokens = seg.split()
            if not tokens:
                continue
            idx = 0
            while idx < len(tokens) and tokens[idx] in ("sudo", "doas"):
                idx += 1
            if idx >= len(tokens) or tokens[idx] not in ("chmod", "chown"):
                continue

            cmd_name = tokens[idx]
            has_R = False
            targets: list[str] = []
            for tok in tokens[idx + 1:]:
                if tok.startswith("-"):
                    if "R" in tok or tok == "--recursive":
                        has_R = True
                else:
                    targets.append(tok)

            if has_R:
                for t in targets:
                    norm = os.path.normpath(os.path.expanduser(t))
                    if norm == "/":
                        return f"recursive {cmd_name} on root directory"

        return None


# ---------------------------------------------------------------------------
# Permission helpers
# ---------------------------------------------------------------------------

def _summarize_call(name: str, args: dict[str, Any]) -> str:
    """Build a short human-readable summary of a tool call."""
    if name == "exec_command":
        return f"exec_command: {args.get('command', '?')}"
    if name in ("write_file", "edit_file", "multi_edit", "read_file", "remove_file", "ls"):
        return f"{name}: {args.get('file') or args.get('path', '?')}"
    if name == "spawn_subagent":
        return f"spawn_subagent: {args.get('task', '?')[:80]}"
    return name


class PermissionController:
    """Gate tool execution behind user confirmation when required.

    A hard ``SafetyGuard`` layer runs first and blocks inherently dangerous
    operations (destructive shell commands) regardless of mode.
    Out-of-scope file paths trigger a user prompt in interactive modes
    so the user can override the guard when needed.

    Responses at the prompt:
      y  — allow this call
      n  — deny this call
      a  — allow this tool for the rest of the session
    """

    def __init__(
        self,
        mode: PermissionMode = PermissionMode.DEFAULT,
        console: AgentConsole | None = None,
        work_dir: str | Path | None = None,
    ):
        self.mode = mode
        self.console = console or AgentConsole()
        self._session_allowed: set[str] = set()
        self.safety = SafetyGuard(work_dir=work_dir)

    @property
    def session_allowed_tools(self) -> frozenset[str]:
        """Return the set of tools allowed for this session (via 'always' prompt)."""
        return frozenset(self._session_allowed)

    def reset_session(self) -> None:
        """Clear all session-level tool allowances."""
        self._session_allowed.clear()

    def check(self, tool_name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
        """Return ``(allowed, reason)`` — *reason* is non-empty only on denial
        when the user provides feedback."""

        # Hard safety layer — always active, cannot be bypassed
        safe, reason = self.safety.check(tool_name, arguments)
        if not safe:
            self.console.print(
                f"[bold red]🛡  Safety guard:[/bold red] [red]{reason}[/red]"
            )
            return False, reason

        # Path-containment check — hard block in AUTO mode (no user),
        # user-overridable in DEFAULT/STRICT modes
        if tool_name in ("write_file", "edit_file", "multi_edit", "remove_file"):
            safe, reason = self.safety.check_path(arguments.get("file", ""))
            if not safe:
                if self.mode == PermissionMode.AUTO:
                    self.console.print(
                        f"[bold red]🛡  Safety guard:[/bold red] [red]{reason}[/red]"
                    )
                    return False, reason
                allowed, override_reason = self._prompt_path_override(
                    tool_name, arguments, reason
                )
                if not allowed:
                    return False, override_reason

        if self.mode == PermissionMode.AUTO:
            return True, ""

        needs_prompt = self._needs_prompt(tool_name)
        if not needs_prompt:
            return True, ""

        if tool_name in self._session_allowed:
            return True, ""

        return self._prompt_user(tool_name, arguments)

    def _needs_prompt(self, tool_name: str) -> bool:
        if self.mode == PermissionMode.STRICT:
            return True
        # default mode: only risky tools
        return tool_name in RISKY_TOOLS

    def _prompt_user(self, tool_name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
        summary = _summarize_call(tool_name, arguments)
        self.console.local(
            f"\n[bold yellow]•  Permission required[/bold yellow]"
            f"\n   [cyan]{summary}[/cyan]",
            highlight=False,
        )
        try:
            choices = ["Yes", "No", "Always allow this tool"]
            prompt = (
                f"Allow this call?\n{summary}"
                if self.console.prompt_broker is not None
                else "Allow this call?"
            )
            answer = self.console.select(
                prompt, choices=choices
            )
        except (EOFError, KeyboardInterrupt):
            self.console.print("   [red]Denied (interrupted).[/red]")
            return False, ""

        if answer is None:
            self.console.print("   [red]✗ Denied.[/red]")
            return False, ""

        response = answer.lower()

        if response in ("always allow this tool", "always"):
            self._session_allowed.add(tool_name)
            self.console.print(f"   [green]✓ {tool_name} allowed for this session.[/green]")
            return True, ""
        if response in ("yes",):
            return True, ""

        reason = self._ask_denial_reason()
        if reason:
            self.console.print(f"   [red]✗ Denied:[/red] {reason}")
        else:
            self.console.print("   [red]✗ Denied.[/red]")
        return False, reason

    def _prompt_path_override(
        self, tool_name: str, arguments: dict[str, Any], safety_reason: str
    ) -> tuple[bool, str]:
        """Prompt the user when a file path is outside the working directory.

        Returns ``(True, "")`` if the user chooses to proceed despite the
        warning, or ``(False, reason)`` if they decline.
        """
        summary = _summarize_call(tool_name, arguments)
        self.console.local(
            f"\n[bold yellow]⚠  Out-of-scope file access[/bold yellow]"
            f"\n   [cyan]{summary}[/cyan]"
            f"\n   [yellow]{safety_reason}[/yellow]",
            highlight=False,
        )
        try:
            choices = ["Allow this call", "Deny"]
            prompt = (
                f"File is outside working directory. Allow?\n{safety_reason}"
                if self.console.prompt_broker is not None
                else "File is outside working directory. Allow?"
            )
            answer = self.console.select(prompt, choices=choices)
        except (EOFError, KeyboardInterrupt):
            self.console.print("   [red]Denied (interrupted).[/red]")
            return False, ""

        if answer is None or answer.lower() not in ("allow this call", "allow", "yes"):
            self.console.print("   [red]✗ Denied — path outside working directory.[/red]")
            return False, safety_reason

        self.console.print("   [yellow]⚠ Allowed — proceeding with out-of-scope file access.[/yellow]")
        return True, ""

    def _ask_denial_reason(self) -> str:
        """Prompt the user for an optional reason after denying a tool call."""
        try:
            reason = self.console.ask_text(
                "Reason for denying (optional; Enter to skip): ", default=""
            )
        except (EOFError, KeyboardInterrupt):
            reason = ""
        return (reason or "").strip()

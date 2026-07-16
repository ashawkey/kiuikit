"""Lightweight permission control for tool execution.

Three modes:
  - auto:    all tools run without prompting (pipe/subagent mode)
  - default: risky tools prompt the user for confirmation
  - strict:  every tool prompts for confirmation

A safety layer (`SafetyGuard`) runs *before* mode-based checks. Its shell
patterns reject common destructive commands (rm -rf /, mkfs, device writes,
fork bombs, …) as defense-in-depth, but they are not a sandbox and cannot
recognize every equivalent shell expression.
"""

from __future__ import annotations

import glob
import os
import re
import shlex
import stat
import sys
from enum import Enum
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
    """Reject recognized dangerous operations regardless of permission mode.

    Shell checks are heuristic defense-in-depth, not a security boundary: a
    full shell can express equivalent commands in ways static pattern matching
    cannot recognize. Use OS-level sandboxing when untrusted commands require
    containment.

    """

    _DESTRUCTIVE_COMMANDS: ClassVar[dict[str, str]] = {
        "blkdiscard": "discard block device",
        "cfdisk": "partition table editor",
        "fdisk": "partition table editor",
        "halt": "system shutdown/reboot",
        "mke2fs": "filesystem format",
        "mkdosfs": "filesystem format",
        "mkfs": "filesystem format",
        "mkfs.btrfs": "filesystem format",
        "mkfs.ext2": "filesystem format",
        "mkfs.ext3": "filesystem format",
        "mkfs.ext4": "filesystem format",
        "mkfs.fat": "filesystem format",
        "mkfs.xfs": "filesystem format",
        "lvremove": "remove logical volume",
        "mkswap": "swap format",
        "newfs": "filesystem format",
        "pvremove": "remove physical volume",
        "parted": "partition table editor",
        "poweroff": "system shutdown/reboot",
        "reboot": "system shutdown/reboot",
        "sfdisk": "partition table editor",
        "shutdown": "system shutdown/reboot",
        "vgremove": "remove volume group",
        "wipefs": "erase filesystem signatures",
    }
    _DANGEROUS_CODE_PATTERNS: ClassVar[list[tuple[re.Pattern[str], str]]] = [
        (re.compile(
            r"\brm\b(?=[^\r\n;]*?(?:--recursive|-[^\s]*[rR]))"
            r"[^\r\n;]*?\s(?:/(?:bin|boot|dev|etc|lib|lib64|opt|proc|root|run|"
            r"sbin|snap|sys|usr)(?:/[^\s'\"]*)?|/(?=[\s'\")};]|$)|~(?:/[^\s'\"]*)?)",
        ), "recursive rm on critical path"),
        (re.compile(
            r"(?<![\w.-])(?:mkfs(?:\.\w+)?|mke2fs|mkdosfs|newfs|mkswap|"
            r"wipefs|blkdiscard)\b",
        ), "filesystem or block-device destruction"),
        (re.compile(r"\bof=/dev/\S+"), "write to block device"),
    ]
    _WRAPPER_COMMANDS = frozenset({
        "command", "doas", "env", "ionice", "nice", "nohup", "setsid",
        "stdbuf", "sudo", "time",
    })
    _WRAPPER_VALUE_OPTIONS = {
        "sudo": frozenset({
            "-C", "--close-from", "-D", "--chdir", "-g", "--group",
            "-h", "--host", "-p", "--prompt", "-R", "--chroot",
            "-T", "--command-timeout", "-u", "--user",
        }),
        "doas": frozenset({"-C", "-u"}),
        "env": frozenset({"-C", "--chdir", "-S", "--split-string", "-u", "--unset"}),
        "ionice": frozenset({"-c", "--class", "-n", "--classdata", "-p", "--pid", "-P", "--pgid", "-u", "--uid"}),
        "nice": frozenset({"-n", "--adjustment"}),
        "stdbuf": frozenset({"-i", "--input", "-o", "--output", "-e", "--error"}),
        "time": frozenset({"-f", "--format", "-o", "--output"}),
    }
    _CONTROL_TOKENS = frozenset({
        ";", "&", "&&", "|", "||", "(", ")", "{", "}", "\n",
    })
    _SHELL_PREFIXES = frozenset({
        "!", "do", "elif", "else", "if", "then", "time", "until", "while",
    })
    _REDIRECT_TOKENS = frozenset({"<", ">", "<<", ">>", "<<<", "<>"})
    _DEVICE_PATH_RE = re.compile(
        r"^/dev/(?:"
        r"(?:sd|hd|vd|xvd)[a-z](?:\d+)?|"
        r"nvme\d+n\d+(?:p\d+)?|"
        r"mmcblk\d+(?:p\d+)?|"
        r"loop\d+|md\d+|dm-\d+|mapper/.+|disk/.+"
        r")$"
    )
    _ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

    _DANGEROUS_WIN_PATTERNS: ClassVar[list[tuple[re.Pattern[str], str]]] = [
        (re.compile(r"(?:^|[;&|\r\n]\s*)(?:format|format\.com)\s+[a-z]:", re.IGNORECASE),
         "format drive"),
        (re.compile(r"(?:^|[;&|\r\n]\s*)diskpart\b", re.IGNORECASE),
         "diskpart command"),
        (re.compile(r"(?:^|[;&|\r\n]\s*)(?:clear|initialize)-disk\b", re.IGNORECASE),
         "disk erase/initialization"),
        (re.compile(r"(?:^|[;&|\r\n]\s*)(?:stop|restart)-computer\b", re.IGNORECASE),
         "system shutdown/reboot"),
        (re.compile(
            r"(?:^|[;&|\r\n]\s*)(?:remove-item|rd|rmdir)\b[^\r\n;&|]*"
            r"(?:[a-z]:\\(?:\s|$)|[a-z]:\\\*).*"
            r"(?:-recurse|-r\b|/s\b)",
            re.IGNORECASE,
        ), "recursive deletion of drive root"),
    ]

    # Paths that should never be recursively mutated or directly removed.
    _CRITICAL_UNIX_PATHS = frozenset({
        "/", "/bin", "/boot", "/dev", "/etc", "/home", "/lib", "/lib64",
        "/media", "/mnt", "/opt", "/proc", "/root", "/run", "/sbin",
        "/snap", "/srv", "/sys", "/tmp", "/usr", "/var",
    })
    _PROTECTED_UNIX_TREES = frozenset({
        "/bin", "/boot", "/dev", "/etc", "/lib", "/lib64", "/opt",
        "/proc", "/root", "/run", "/sbin", "/snap", "/sys", "/usr",
    })

    def __init__(self, work_dir: str | None = None):
        self._work_dir = os.path.abspath(work_dir or os.getcwd())

    def check(self, tool_name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
        """Return ``(allowed, reason)``.  *reason* is non-empty when blocked.

        Only performs *hard* checks that should never be skipped:
        destructive shell commands, critical rm/chmod patterns, etc.
        """
        if tool_name == "exec_command":
            return self._check_command(
                arguments.get("command", ""), arguments.get("cwd")
            )
        if tool_name in ("write_file", "edit_file", "multi_edit"):
            path = arguments.get("file", "")
            if path and self._is_device_path(path):
                return False, f"Blocked: write to block device ({path})."
        if tool_name == "remove_file":
            return self._check_remove(arguments.get("file", ""))
        return True, ""

    # -- command analysis ---------------------------------------------------

    def _check_command(self, command: str, cwd: str | None = None) -> tuple[bool, str]:
        if sys.platform == "win32":
            for pattern, desc in self._DANGEROUS_WIN_PATTERNS:
                if pattern.search(command):
                    return False, f"Blocked: {desc}."
            return True, ""

        if re.search(r":\s*\(\s*\)\s*\{", command):
            return False, "Blocked: fork bomb."
        for pattern, desc in self._DANGEROUS_CODE_PATTERNS:
            if pattern.search(command):
                return False, f"Blocked: {desc}."

        try:
            tokens = self._tokenize(command)
        except ValueError as e:
            return False, f"Blocked: cannot safely parse shell command ({e})."

        for idx, token in enumerate(tokens[:-1]):
            if ">" in token and self._is_device_path(tokens[idx + 1]):
                return False, "Blocked: redirect to block device."

        segments = self._command_segments(tokens)
        for segment in segments:
            reason = self._check_segment(segment, cwd)
            if reason:
                return False, f"Blocked: {reason}."

        return True, ""

    @staticmethod
    def _tokenize(command: str) -> list[str]:
        lexer = shlex.shlex(
            command, posix=True, punctuation_chars=";&|()<>{}\n"
        )
        lexer.whitespace = " \t\r"
        lexer.whitespace_split = True
        lexer.commenters = ""
        return list(lexer)

    @classmethod
    def _command_segments(cls, tokens: list[str]) -> list[list[str]]:
        segments: list[list[str]] = []
        current: list[str] = []
        for token in tokens:
            if token in cls._CONTROL_TOKENS:
                if current:
                    segments.append(current)
                    current = []
            else:
                current.append(token)
        if current:
            segments.append(current)
        return segments

    def _check_segment(self, tokens: list[str], cwd: str | None) -> str | None:
        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            name = os.path.basename(token).lower()

            if token in self._SHELL_PREFIXES or self._ASSIGNMENT_RE.match(token):
                idx += 1
                continue
            if token.isdigit() and idx + 1 < len(tokens) and tokens[idx + 1] in self._REDIRECT_TOKENS:
                idx += 1
                continue
            if token in self._REDIRECT_TOKENS:
                idx += 2
                continue

            if name in self._WRAPPER_COMMANDS:
                idx = self._skip_wrapper(tokens, idx, name)
                continue

            args = tokens[idx + 1:]
            reason = self._check_invocation(name, args, cwd)
            if reason:
                return reason
            return None
        return None

    def _skip_wrapper(self, tokens: list[str], idx: int, name: str) -> int:
        idx += 1
        value_options = self._WRAPPER_VALUE_OPTIONS.get(name, frozenset())
        while idx < len(tokens):
            token = tokens[idx]
            if token == "--":
                return idx + 1
            if name == "env" and self._ASSIGNMENT_RE.match(token):
                idx += 1
                continue
            option = token.split("=", 1)[0]
            if option in value_options:
                idx += 1 if "=" in token else 2
                continue
            if token.startswith("-"):
                idx += 1
                continue
            return idx
        return idx

    def _check_invocation(
        self, name: str, args: list[str], cwd: str | None
    ) -> str | None:
        if name in self._DESTRUCTIVE_COMMANDS:
            return self._DESTRUCTIVE_COMMANDS[name]
        if name.startswith("mkfs."):
            return "filesystem format"

        if name == "rm":
            return self._check_rm(args, cwd)
        if name == "git":
            return self._check_git(args)
        if name in ("chmod", "chown"):
            return self._check_recursive_perm(name, args, cwd)
        if name in ("busybox", "toybox") and args:
            return self._check_invocation(os.path.basename(args[0]), args[1:], cwd)
        if name == "find":
            if "-delete" in args:
                for target in self._find_targets(args):
                    if self._is_critical_path(target, cwd):
                        return f"find -delete on critical path ({target})"
        if name in ("systemctl", "loginctl") and any(
            arg in ("halt", "poweroff", "reboot") for arg in args
        ):
            return "system shutdown/reboot"
        if name == "cryptsetup" and any("luksformat" in arg.lower() for arg in args):
            return "encrypted volume format"
        if name == "zpool" and "destroy" in args:
            return "destroy storage pool"
        if name == "zfs" and "destroy" in args:
            return "destroy filesystem"

        for token in args:
            if token.startswith("of=") and self._is_device_path(token[3:]):
                return "write to block device"
        if name in ("cp", "mv", "tee", "truncate", "shred"):
            if any(self._is_device_path(token) for token in args):
                return "write to block device"

        return None

    def _check_rm(self, args: list[str], cwd: str | None) -> str | None:
        recursive = False
        targets: list[str] = []
        options_done = False
        for token in args:
            if not options_done and token == "--":
                options_done = True
            elif not options_done and token.startswith("-"):
                if token == "--no-preserve-root":
                    return "rm --no-preserve-root"
                if token == "--recursive" or (
                    not token.startswith("--") and "r" in token.lower()
                ):
                    recursive = True
            else:
                targets.append(token)

        if recursive:
            for target in targets:
                if self._is_critical_path(target, cwd):
                    return f"recursive rm on critical path ({target})"
        return None

    @staticmethod
    def _check_git(args: list[str]) -> str | None:
        if "clean" in args and any(
            token.startswith("-") and "f" in token[1:] for token in args
        ):
            return "git clean can destroy untracked files"
        if "reset" in args and "--hard" in args:
            return "git reset --hard can destroy uncommitted changes"
        if any(command in args for command in ("checkout", "restore")) and any(
            target in (".", ":/") for target in args
        ):
            return "git operation can overwrite the working tree"
        return None

    def _check_recursive_perm(
        self, name: str, args: list[str], cwd: str | None
    ) -> str | None:
        recursive = any(
            token == "--recursive"
            or (token.startswith("-") and not token.startswith("--") and "R" in token)
            for token in args
        )
        if recursive and any(self._is_critical_path(token, cwd) for token in args):
            return f"recursive {name} on critical path"
        return None

    @staticmethod
    def _find_targets(args: list[str]) -> list[str]:
        targets: list[str] = []
        for token in args:
            if token.startswith("-") or token in ("!", "(", ")"):
                break
            targets.append(token)
        return targets or ["."]

    def _is_critical_path(self, path: str, cwd: str | None = None) -> bool:
        pattern = os.path.expandvars(os.path.expanduser(path))
        if not os.path.isabs(pattern):
            pattern = os.path.join(cwd or self._work_dir, pattern)
        literal = pattern.rstrip("*?[]{}") or "/"
        norm = os.path.normpath(os.path.realpath(os.path.abspath(literal)))
        home = os.path.normpath(os.path.realpath(os.path.expanduser("~")))
        work_dir = os.path.normpath(os.path.realpath(self._work_dir))
        if work_dir == norm or work_dir.startswith(norm + os.sep):
            return True
        if self._is_resolved_critical(norm, home):
            return True
        return any(
            self._is_resolved_critical(os.path.normpath(os.path.realpath(match)), home)
            for match in glob.glob(pattern)
        )

    def _is_resolved_critical(self, path: str, home: str) -> bool:
        return (
            path in self._CRITICAL_UNIX_PATHS
            or path == home
            or os.path.ismount(path)
            or any(path.startswith(root + os.sep) for root in self._PROTECTED_UNIX_TREES)
        )

    def _is_device_path(self, path: str) -> bool:
        path = os.path.expandvars(os.path.expanduser(path))
        if not os.path.isabs(path):
            path = os.path.join(self._work_dir, path)
        path = os.path.normpath(path)
        if self._DEVICE_PATH_RE.match(path):
            return True
        try:
            return stat.S_ISBLK(os.stat(path).st_mode)
        except OSError:
            return False

    def _check_remove(self, path: str) -> tuple[bool, str]:
        if path and self._is_critical_path(path):
            return False, f"Blocked: removal of critical path ({path})."
        return True, ""


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

    ``SafetyGuard`` first rejects recognized destructive shell patterns as a
    defense-in-depth measure regardless of mode. It is not a shell sandbox.

    Responses at the prompt:
      y  — allow this call
      n  — deny this call
      a  — allow this tool for the rest of the session
    """

    def __init__(
        self,
        mode: PermissionMode = PermissionMode.DEFAULT,
        console: AgentConsole | None = None,
        work_dir: str | None = None,
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

        safe, reason = self.check_safety(tool_name, arguments)
        if not safe:
            return False, reason

        if self.mode == PermissionMode.AUTO:
            return True, ""

        needs_prompt = self._needs_prompt(tool_name)
        if not needs_prompt:
            return True, ""

        if tool_name in self._session_allowed:
            return True, ""

        return self._prompt_user(tool_name, arguments)

    def check_safety(self, tool_name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
        """Run safety checks without applying mode-based confirmation policy."""
        safe, reason = self.safety.check(tool_name, arguments)
        if not safe:
            self.console.print(
                f"[bold red]🛡  Safety guard:[/bold red] [red]{reason}[/red]"
            )
        return safe, reason

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

    def _ask_denial_reason(self) -> str:
        """Prompt the user for an optional reason after denying a tool call."""
        try:
            reason = self.console.ask_text(
                "Reason for denying (optional; Enter to skip): ", default=""
            )
        except (EOFError, KeyboardInterrupt):
            reason = ""
        return (reason or "").strip()

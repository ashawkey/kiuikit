"""Unified user-facing I/O for the kiui agent.

All console output, confirmation prompts, and log messages go through
AgentConsole so that theming, stderr redirection (pipe mode), and
interactive prompting are handled in one place.
"""

from __future__ import annotations

import threading
import time
import questionary
from rich.console import Console
from rich.status import Status
from rich.table import Table
from rich.theme import Theme

# Custom questionary style that blends with the CLI aesthetic
_QS_STYLE = questionary.Style([
    ("qmark", "fg:ansiblue bold"),
    ("question", "fg:ansiyellow bold"),
    ("answer", "fg:ansiyellow bold"),
    ("pointer", "fg:ansicyan bold"),
    ("highlighted", "fg:ansicyan bold"),
    ("selected", "fg:ansigreen"),
    ("separator", "fg:ansiblack"),
    ("instruction", "fg:ansibrightblack"),
    ("text", "fg:ansibrightblack"),
    ("disabled", "fg:ansibrightblack italic"),
])

# Marker glyphs
_DOT = "\u2022"      # bullet •
_CHECK = "\u2713"    # ✓
_CROSS = "\u2717"    # ✗

AGENT_THEME = Theme({
    "debug": "dim cyan",
    "input": "bold yellow",
    "response": "white",
    "error": "bold red",
    "warning": "bold yellow",
    "system": "bold blue",
    "tool": "color(244)",
    "tool_ok": "dim green",
    "tool_fail": "dim red",
})


class ThinkingIndicator:
    """Context manager that shows an animated "Working... (Xs)" line
    while the model is generating a response.

    Uses Rich's ``Status`` with a bouncing-ball spinner in a background
    thread to update the elapsed-time counter every second.
    """

    def __init__(self, console: Console):
        self._console = console
        self._status: Status | None = None
        self._start_time: float = 0
        self._running = False
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "ThinkingIndicator":
        self._start_time = time.monotonic()
        self._running = True
        self._status = Status(
            "Working...",
            spinner="dots",
            console=self._console,
            speed=1.5,
        )
        self._status.start()
        self._thread = threading.Thread(target=self._tick, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._status is not None:
            self._status.stop()

    def _tick(self) -> None:
        """Background loop that updates the elapsed-time label."""
        while self._running:
            time.sleep(0.5)
            if self._running and self._status is not None:
                elapsed = time.monotonic() - self._start_time
                try:
                    self._status.update(f"Working... ({elapsed:.0f}s)")
                except Exception:
                    pass  # best-effort; don't crash the agent on a display glitch


class AgentConsole:
    """Thin wrapper around ``rich.Console`` with typed convenience methods."""

    def __init__(self):
        self._console = Console(theme=AGENT_THEME)

    def thinking(self) -> ThinkingIndicator:
        """Return a context manager that displays an animated thinking indicator.

        Usage::

            with console.thinking():
                response = client.chat.completions.create(...)
        """
        return ThinkingIndicator(self._console)

    # -- raw pass-through (for rich markup, tables, etc.) -------------------

    def print(self, *args, **kwargs):
        self._console.print(*args, **kwargs)

    def table(self, table: Table):
        self._console.print(table)

    # -- block helper -------------------------------------------------------

    def _block(self, msg: str, style: str, *, prefix: str, markup: bool = False):
        """Print a block: first line with *prefix*, continuation lines indented."""
        lines = msg.splitlines()
        indent = " " * len(prefix)
        if not lines:
            self._console.print(prefix.rstrip(), style=style)
            return
        out = [f"{prefix}{lines[0]}"] + [f"{indent}{line}" for line in lines[1:]]
        self._console.print("\n".join(out), style=style, markup=markup)

    # -- typed log helpers --------------------------------------------------

    def system(self, msg: str):
        self._block(msg, style="system", prefix=f"{_DOT} ")

    def debug(self, msg: str):
        self._block(msg, style="debug", prefix=f"{_DOT} ")

    def error(self, msg: str):
        self._block(msg, style="error", prefix=f"{_DOT} ")

    def warn(self, msg: str, *, exc_info: bool = False):
        self._block(msg, style="warning", prefix=f"{_DOT} ")
        if exc_info:
            self._console.print_exception()

    def tool(self, msg: str):
        self._block(msg, style="tool", prefix=f"{_DOT} ")

    def tool_result(self, msg: str, success: bool = True):
        style = "tool_ok" if success else "tool_fail"
        icon = _CHECK if success else _CROSS
        prefix = f"{_DOT} {icon} "
        lines = msg.splitlines()
        indent = " " * len(prefix)
        first = prefix + (lines[0] if lines else "")
        rest = "\n".join(indent + line for line in lines[1:])
        output = first + ("\n" + rest if rest else "")
        self._console.print(output, style=style, markup=False)

    def _highlight_line(self, code: str, language: str) -> "Text":
        """Return a ``rich.text.Text`` with syntax-highlighted *code*.

        Falls back to plain white text when highlighting isn't available.
        """
        from rich.syntax import Syntax
        from rich.text import Text as RichText

        try:
            syntax = Syntax(
                code, language, theme="monokai",
                line_numbers=False, background_color=None,
            )
            segments = list(self._console.render(syntax))
            # Syntax renderer may append a trailing newline; drop it.
            if segments and segments[-1].text == "\n":
                segments.pop()
            text = RichText()
            for segment in segments:
                text.append(segment.text, style=segment.style)
            return text
        except Exception:
            return RichText(code, style="white")

    def _guess_language(self, path: str) -> str:
        """Guess a Pygments lexer name from a file extension."""
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        _MAP = {
            "py": "python", "pyi": "python", "pyx": "python",
            "js": "javascript", "ts": "typescript", "tsx": "typescript",
            "jsx": "javascript", "json": "json", "yaml": "yaml", "yml": "yaml",
            "toml": "toml", "cfg": "ini", "ini": "ini",
            "html": "html", "css": "css", "scss": "scss",
            "md": "markdown", "rst": "rst",
            "sh": "bash", "bash": "bash", "zsh": "bash",
            "ps1": "powershell", "psm1": "powershell",
            "sql": "sql", "rs": "rust", "go": "go",
            "java": "java", "kt": "kotlin", "swift": "swift",
            "c": "c", "h": "c", "cpp": "cpp", "hpp": "cpp",
            "rb": "ruby", "php": "php", "lua": "lua",
            "vim": "vim", "dockerfile": "dockerfile",
        }
        return _MAP.get(ext, "text")

    def diff_edit(
        self,
        path: str,
        old_text: str | None,
        new_text: str | None,
        line_num: int | None = None,
        count: int = 1,
        success: bool = True,
    ):
        """Render a file diff with +/- markers, muted backgrounds, and
        syntax highlighting.

        - Dark red background for removed lines (prefixed ``-``)
        - Dark green background for added lines (prefixed ``+``)
        - Full terminal-width backgrounds so the diff stands out as a block.
        - When *old_text* is empty or ``None`` (e.g. write_file), only ``+`` lines are shown.
        """
        from rich.text import Text as RichText

        # Muted, dark backgrounds — visible but not jarring
        BG_RED = "#3a1a1a"
        BG_GREEN = "#1a3a1a"
        SIGN_RED = "#f87171"     # soft red for the "-" sign
        SIGN_GREEN = "#4ade80"   # soft green for the "+" sign

        icon = _CHECK if success else _CROSS
        style = "tool_ok" if success else "tool_fail"

        # Build header line
        if count > 1:
            header = f"{path}  ({count} occurrences)"
        elif line_num is not None:
            header = f"{path}  line {line_num}"
        else:
            header = f"{path}"

        prefix = f"{_DOT} {icon} "
        self._console.print(f"{prefix}{header}", style=style)

        if not success:
            return

        indent = " " * len(prefix)
        pad_to = self._console.width
        language = self._guess_language(path)
        has_old = bool(old_text)
        start_line = line_num if line_num is not None else 1
        LN = "dim color(240)"  # line-number style: dim grey, unobtrusive

        if has_old:
            for i, line in enumerate(old_text.splitlines()):
                ln = f"{start_line + i:>4} "
                t = RichText(f"{indent}", style="")
                t.append(ln, style=LN)
                t.append("- ", style=f"bold {SIGN_RED}")
                t.append(self._highlight_line(line, language))
                t.append(" " * max(0, pad_to - len(t)))
                t.stylize(f"on {BG_RED}")
                self._console.print(t)

        if new_text:
            for i, line in enumerate(new_text.splitlines()):
                ln = f"{start_line + i:>4} "
                t = RichText(f"{indent}", style="")
                t.append(ln, style=LN)
                t.append("+ ", style=f"bold {SIGN_GREEN}")
                t.append(self._highlight_line(line, language))
                t.append(" " * max(0, pad_to - len(t)))
                t.stylize(f"on {BG_GREEN}")
                self._console.print(t)

    def response(self, msg: str):
        from rich.markdown import Markdown
        self._console.print(Markdown(f"{_DOT} {msg}"))

    def user_input(self, msg: str):
        """Print user input with a horizontal rule for visual separation."""
        self._console.rule(style="dim color(240)")
        lines = msg.splitlines()
        if not lines:
            return
        prefix = "> "
        indent = " " * len(prefix)
        out_lines = []
        for i, line in enumerate(lines):
            content = prefix + line if i == 0 else indent + line
            out_lines.append(content)
        self._console.print("\n".join(out_lines), style="input", markup=False)

    # -- interactive prompts ------------------------------------------------

    def confirm(self, prompt: str, choices: list[str] | None = None) -> str:
        """Display *prompt* with rich styling, then use questionary for interactive selection.

        Returns one of the choice strings (lowercased), or "" if cancelled.
        """
        self._console.print(f"{_DOT} {prompt}", style="warning", highlight=False)
        if choices is None:
            choices = ["Yes", "No", "Always allow this tool"]
        try:
            answer = questionary.select(
                "",
                choices=choices,
                qmark="",
                style=_QS_STYLE,
                use_indicator=True,
            ).unsafe_ask()
        except KeyboardInterrupt:
            return ""
        if answer is None:
            return ""
        return answer.lower()

    def select(
        self,
        message: str,
        choices: list[str],
        *,
        default: str | None = None,
        use_indicator: bool = True,
    ) -> str | None:
        """Present a scrollable single-select list via questionary.

        Returns the chosen string, or None if cancelled.
        """
        try:
            return questionary.select(
                message,
                choices=choices,
                default=default,
                style=_QS_STYLE,
                use_indicator=use_indicator,
            ).unsafe_ask()
        except KeyboardInterrupt:
            return None

    def ask_text(
        self,
        message: str,
        *,
        default: str = "",
        multiline: bool = False,
        validate: questionary.Validator | None = None,
    ) -> str | None:
        """Prompt for free-text input via questionary.

        Returns the input string, or None if cancelled.
        """
        try:
            return questionary.text(
                message,
                default=default,
                multiline=multiline,
                validate=validate,
                style=_QS_STYLE,
            ).unsafe_ask()
        except KeyboardInterrupt:
            return None

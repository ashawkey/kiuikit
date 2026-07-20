"""Unified user-facing I/O for the kiui agent.

All console output, confirmation prompts, and log messages go through
AgentConsole so that theming, stderr redirection (pipe mode), and
interactive prompting are handled in one place.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.status import Status
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

if TYPE_CHECKING:
    from kiui.agent.io import EventHub, PromptBroker

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

_AGENT_LOGO = (
    "   ▄   \n"
    " ▄█▀█▄ \n"
    "▀█▄▀▄█▀\n"
    "  ▀█▀  \n"
)

# Diff rendering: if old + new lines exceed this, show a summary instead
DIFF_MAX_LINES = 200
DIFF_PREVIEW_LINES = 10  # how many lines to show from each side in summary mode

AGENT_THEME = Theme({
    "debug": "dim cyan",
    "input": "bold yellow",
    "response": "white",
    "error": "bold red",
    "warning": "bold yellow",
    "system": "bold blue",
    "thinking": "dim italic color(245)",
    "tool": "color(244)",
    "tool_ok": "dim green",
    "tool_fail": "dim red",
})


def _compact_tokens(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return str(value)


def _print_markdown_response(console: Console, content: str) -> None:
    from rich.markdown import Markdown

    row = Table.grid(padding=0, expand=True)
    row.add_column(width=2, no_wrap=True)
    row.add_column(ratio=1)
    row.add_row(Text(f"{_DOT} "), Markdown(content))
    console.print(row)


@dataclass(frozen=True)
class ContextStatus:
    """Context-window usage displayed alongside the thinking spinner."""

    tokens: int
    limit: int
    total_tokens_used: int

    @property
    def fraction(self) -> float:
        if self.limit <= 0:
            return 0.0
        return min(max(self.tokens / self.limit, 0.0), 1.0)

    def plain(self) -> str:
        if self.limit <= 0:
            return f"~{_compact_tokens(self.tokens)} · {_compact_tokens(self.total_tokens_used)} used"
        return f"{self.fraction:.0%} · {_compact_tokens(self.total_tokens_used)} used"

    def render(self) -> Table | Text:
        if self.limit <= 0:
            return Text(self.plain(), style="dim")

        color = "red" if self.fraction >= 0.9 else "yellow" if self.fraction >= 0.75 else "cyan"
        row = Table.grid(padding=(0, 1))
        row.add_row(
            ProgressBar(
                total=self.limit,
                completed=self.fraction * self.limit,
                width=14,
                style="color(238)",
                complete_style=color,
                finished_style=color,
            ),
            Text(f"{self.fraction:.0%}", style=color),
            Text(f"· {_compact_tokens(self.total_tokens_used)} used", style="dim"),
        )
        return row


class ThinkingIndicator:
    """Animated model activity or indeterminate operation progress."""

    def __init__(
        self,
        console: Console,
        events: EventHub | None = None,
        status_suffix: str | ContextStatus = "",
        label: str = "Working",
        progress: bool = False,
    ):
        self._console = console
        self._events = events
        self._status_suffix = status_suffix
        self._label_text = label
        self._progress = progress
        self._status: Status | None = None
        self._start_time: float = 0
        self._started_at: float = 0
        self._running = False
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "ThinkingIndicator":
        self._start_time = time.monotonic()
        self._started_at = time.time()
        self._running = True
        self._status = Status(
            self._label(0.0),
            spinner="dots",
            console=self._console,
            speed=1.5,
        )
        self._status.start()
        self._thread = threading.Thread(target=self._tick, daemon=True)
        self._thread.start()
        if self._events is not None:
            if isinstance(self._status_suffix, ContextStatus):
                self._events.publish(
                    "thinking_start",
                    suffix=self._status_suffix.plain(),
                    context_tokens=self._status_suffix.tokens,
                    context_limit=self._status_suffix.limit,
                    total_tokens_used=self._status_suffix.total_tokens_used,
                    started_at=self._started_at,
                    label=self._label_text,
                    progress=self._progress,
                )
            else:
                self._events.publish(
                    "thinking_start",
                    suffix=self._status_suffix,
                    started_at=self._started_at,
                    label=self._label_text,
                    progress=self._progress,
                )
        return self

    def __exit__(self, *args) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._status is not None:
            self._status.stop()
        if self._events is not None:
            self._events.publish("thinking_stop")

    def _label(self, elapsed: float) -> str | Table:
        base = (
            f"{self._label_text}... ({elapsed:.0f}s)"
            if elapsed
            else f"{self._label_text}..."
        )
        if self._progress:
            row = Table.grid(padding=(0, 1))
            cells = [
                Text(base),
                ProgressBar(total=None, pulse=True, width=14, pulse_style="cyan"),
            ]
            if self._status_suffix:
                cells.extend([
                    Text("·", style="dim"),
                    Text(str(self._status_suffix), style="dim"),
                ])
            row.add_row(*cells)
            return row
        if not self._status_suffix:
            return base
        if isinstance(self._status_suffix, str):
            return f"{base} · {self._status_suffix}"

        row = Table.grid(padding=(0, 1))
        row.add_row(Text(base), Text("·", style="dim"), self._status_suffix.render())
        return row

    def _tick(self) -> None:
        """Background loop that updates the elapsed-time label."""
        while self._running:
            time.sleep(0.5)
            if self._running and self._status is not None:
                elapsed = time.monotonic() - self._start_time
                try:
                    self._status.update(self._label(elapsed))
                except Exception:
                    pass  # best-effort; don't crash the agent on a display glitch


class ResponseStream:
    """Buffered terminal sink for a streamed assistant turn.

    Rich Live cannot safely update content taller than the terminal: old frames
    enter scrollback and cannot be erased, repeating the response. Terminal
    output is therefore buffered and rendered once on close. Web clients still
    receive fragment events in real time, followed by consolidated events for
    late-joining clients.
    """

    def __init__(
        self,
        console: Console,
        events: "EventHub | None",
        *,
        show_thinking: bool = False,
    ):
        self._console = console
        self._events = events
        self._show_thinking = show_thinking
        self._content = ""
        self._thinking = ""
        self._content_visible = False
        self._closed = False
        self._lock = threading.Lock()

    def __enter__(self) -> "ResponseStream":
        return self

    def __exit__(self, exc_type, *args) -> None:
        self.close(render_terminal=exc_type is None)

    def on_content(self, text: str) -> None:
        with self._lock:
            if self._closed:
                return
            self._content += text
            if not self._content_visible:
                if not self._content.strip():
                    return
                self._content_visible = True
                text = self._content
            if self._events is not None:
                self._events.publish("assistant_delta", text=text)

    def on_thinking(self, text: str) -> None:
        # show_thinking gates reasoning output uniformly across terminal and
        # web, so the two surfaces never disagree on whether a turn had a
        # thinking block.
        if not self._show_thinking:
            return
        with self._lock:
            if self._closed:
                return
            self._thinking += text
            if self._events is not None:
                self._events.publish("thinking_delta", text=text)

    def close(self, *, render_terminal: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            thinking = self._thinking
            content = self._content if self._content_visible else ""
        if render_terminal:
            if thinking:
                self._console.print(Text(thinking, style="thinking"))
            if content:
                _print_markdown_response(self._console, content)
        if self._events is not None:
            if thinking:
                self._events.publish("thinking", text=thinking)
            if content:
                self._events.publish("assistant_message", text=content)


class AgentConsole:
    """Thin wrapper around ``rich.Console`` with typed convenience methods."""

    def __init__(self, events: EventHub | None = None):
        self._console = Console(theme=AGENT_THEME)
        self.events = events
        self.prompt_broker: PromptBroker | None = None

    def _emit(self, event_type: str, **data) -> None:
        if self.events is not None:
            self.events.publish(event_type, **data)

    def _render_plain(self, *objects, markup: bool = True) -> str:
        """Render rich objects/markup to plain text for web clients."""
        capture_console = Console(width=120, theme=AGENT_THEME, no_color=True)
        with capture_console.capture() as capture:
            capture_console.print(*objects, markup=markup)
        return capture.get().rstrip("\n")

    def thinking(
        self,
        *,
        status_suffix: str | ContextStatus = "",
        label: str = "Working",
        progress: bool = False,
    ) -> ThinkingIndicator:
        """Return a context manager that displays an animated thinking indicator.

        ``status_suffix`` is appended after the elapsed-time counter. Set
        ``progress`` for an indeterminate bar. A :class:`ContextStatus` renders
        context usage as a progress bar in the terminal and plain text on web.

        Usage::

            with console.thinking():
                response = client.chat.completions.create(...)
        """
        return ThinkingIndicator(
            self._console,
            self.events,
            status_suffix=status_suffix,
            label=label,
            progress=progress,
        )

    def stream_response(self, *, show_thinking: bool = False) -> ResponseStream:
        """Return a buffered terminal sink for a streamed assistant turn.

        Usage::

            with console.stream_response() as sink:
                message, usage = consume_stream(stream, on_content=sink.on_content, ...)
        """
        return ResponseStream(self._console, self.events, show_thinking=show_thinking)

    # -- raw pass-through (for rich markup, tables, etc.) -------------------

    def print(self, *args, **kwargs):
        self._console.print(*args, **kwargs)
        if args and self.events is not None:
            self._emit(
                "output",
                text=self._render_plain(*args, markup=kwargs.get("markup", True)),
            )

    def local(self, *args, **kwargs):
        """Print terminal-only information such as authentication secrets."""
        self._console.print(*args, **kwargs)

    def table(self, table: Table):
        self._console.print(table)
        if self.events is not None:
            self._emit("output", text=self._render_plain(table))

    def startup_panel(
        self,
        model: str,
        context: str,
        reasoning: str,
        permission: str,
        persona: str,
        skills: str,
        workspace: str,
    ) -> None:
        """Render the interactive agent's startup summary."""
        logo = Text(_AGENT_LOGO, style="bold green")

        details = Table.grid(padding=(0, 1), expand=True)
        details.add_column(width=12, no_wrap=True, style="dim green")
        details.add_column(ratio=1, style="white", overflow="fold")
        details.add_row("Model", Text(f"{model} ({context}) · {reasoning}", style="bold"))

        details.add_row("Permission", Text(permission, style="bold"))
        details.add_row("Persona", Text(persona, style="bold"))
        details.add_row("Skills", Text(skills, style="bold"))
        details.add_row("Workspace", Text(workspace, style="bold"))

        body = Table.grid(expand=True)
        logo_width = max(len(line) for line in _AGENT_LOGO.splitlines())
        body.add_column(width=logo_width + 4, no_wrap=True)
        body.add_column(ratio=1)
        body.add_row(logo, details)
        panel = Panel(body, border_style="green", padding=(0, 2))
        self.print(panel)

    def rule(self):
        self._console.rule(style="dim color(240)")
        self._emit("rule")

    def reset_timeline(self):
        """Clear rendered history before replaying authoritative context."""
        if self._console.is_terminal:
            print("\033[3J\033[2J\033[H", end="", file=self._console.file, flush=True)
        self._emit("timeline_reset")

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
        self._emit("system", text=msg)

    def debug(self, msg: str):
        self._block(msg, style="debug", prefix=f"{_DOT} ")
        self._emit("debug", text=msg)

    def error(self, msg: str):
        self._block(msg, style="error", prefix=f"{_DOT} ")
        self._emit("error", text=msg)

    def warn(self, msg: str, *, exc_info: bool = False):
        self._block(msg, style="warning", prefix=f"{_DOT} ")
        self._emit("warning", text=msg)
        if exc_info:
            self._console.print_exception()

    def tool(self, msg: str):
        self._block(msg, style="tool", prefix=f"{_DOT} ")
        self._emit("tool_start", text=msg)

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
        self._emit("tool_result", text=msg, success=success)

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

    def _render_diff_block(
        self,
        lines: list[str],
        sign: str,
        sign_style: str,
        bg_color: str,
        indent: str,
        pad_to: int,
        language: str,
        start_line: int,
        limit: int | None = None,
    ):
        """Render a block of diff lines (either removals or additions).

        When *limit* is set, only the first *limit* lines are rendered,
        and the count of omitted lines is returned.
        """
        from rich.text import Text as RichText

        LN = "color(245)"
        rendered = 0
        for i, line in enumerate(lines):
            if limit is not None and rendered >= limit:
                break
            ln = f"{start_line + i:>4} "
            t = RichText(f"{indent}", style="")
            t.append(ln, style=LN)
            t.append(f"{sign} ", style=sign_style)
            t.append(self._highlight_line(line, language))
            padding = max(0, pad_to - t.cell_len)
            if padding:
                t.append(" " * padding)
            t.stylize(f"on {bg_color}")
            self._console.print(t)
            rendered += 1
        return len(lines) - rendered  # omitted count

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
        - Large diffs (> DIFF_MAX_LINES total) show a summary with preview instead of
          the full diff to avoid flooding the terminal.
        """
        # Muted, dark backgrounds — visible but not jarring
        BG_RED = "#3a1a1a"
        BG_GREEN = "#1a3a1a"
        SIGN_RED = "#f87171"
        SIGN_GREEN = "#4ade80"

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
        self._emit(
            "diff",
            path=path,
            old_text=old_text or "",
            new_text=new_text or "",
            line_num=line_num,
            count=count,
            success=success,
        )

        if not success:
            return

        indent = " " * len(prefix)
        pad_to = min(self._console.width, 160)
        language = self._guess_language(path)
        has_old = bool(old_text)
        start_line = line_num if line_num is not None else 1

        old_lines = old_text.splitlines() if old_text else []
        new_lines = new_text.splitlines() if new_text else []
        total_lines = len(old_lines) + len(new_lines)

        # ── summary mode for large diffs ──
        if total_lines > DIFF_MAX_LINES:
            removed = len(old_lines)
            added = len(new_lines)
            parts = []
            if removed:
                parts.append(f"{removed} removed")
            if added:
                parts.append(f"{added} added")
            stat = ", ".join(parts)
            self._console.print(
                f"{indent}{stat}  (diff too large — showing first {DIFF_PREVIEW_LINES} lines per side)",
                style="dim",
            )

            preview = DIFF_PREVIEW_LINES
            if has_old:
                omitted = self._render_diff_block(
                    old_lines, "-", f"bold {SIGN_RED}", BG_RED,
                    indent, pad_to, language, start_line, limit=preview,
                )
                if omitted:
                    self._console.print(
                        f"{indent}...  ({omitted} more removed lines)", style="dim"
                    )
            if new_lines:
                omitted = self._render_diff_block(
                    new_lines, "+", f"bold {SIGN_GREEN}", BG_GREEN,
                    indent, pad_to, language, start_line, limit=preview,
                )
                if omitted:
                    self._console.print(
                        f"{indent}...  ({omitted} more added lines)", style="dim"
                    )
            return

        # ── full diff ──
        if has_old:
            self._render_diff_block(
                old_lines, "-", f"bold {SIGN_RED}", BG_RED,
                indent, pad_to, language, start_line,
            )

        if new_lines:
            self._render_diff_block(
                new_lines, "+", f"bold {SIGN_GREEN}", BG_GREEN,
                indent, pad_to, language, start_line,
            )

    def response(self, msg: str):
        _print_markdown_response(self._console, msg)
        self._emit("assistant_message", text=msg)

    def user_input(
        self,
        msg: str,
        *,
        source: str = "replay",
        submission_id: str | None = None,
        with_rule: bool = True,
    ):
        """Print user input with a horizontal rule for visual separation.

        Single emission point for ``user_message`` events — callers must not
        publish the event themselves.
        """
        if with_rule:
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
        data = {"text": msg, "source": source}
        if submission_id is not None:
            data["submission_id"] = submission_id
        self._emit("user_message", **data)

    # -- interactive prompts ------------------------------------------------

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
        if self.prompt_broker is not None:
            return self.prompt_broker.ask(
                "select", message, choices=choices, default=default or ""
            )
        return self.select_terminal(
            message,
            choices,
            default=default,
            use_indicator=use_indicator,
        )

    def select_terminal(
        self,
        message: str,
        choices: list[str],
        *,
        default: str | None = None,
        use_indicator: bool = True,
    ) -> str | None:
        """Render a select prompt only on the local terminal."""
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

    async def select_terminal_async(
        self,
        message: str,
        choices: list[str],
        *,
        default: str | None = None,
        use_indicator: bool = True,
    ) -> str | None:
        """Render a local select while allowing web input to race it."""
        question = questionary.select(
            message,
            choices=choices,
            default=default,
            style=_QS_STYLE,
            use_indicator=use_indicator,
        )
        try:
            return await question.unsafe_ask_async()
        except (KeyboardInterrupt, EOFError):
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
        if self.prompt_broker is not None:
            return self.prompt_broker.ask(
                "text", message, default=default
            )
        return self.ask_text_terminal(
            message, default=default, multiline=multiline, validate=validate
        )

    def ask_text_terminal(
        self,
        message: str,
        *,
        default: str = "",
        multiline: bool = False,
        validate: questionary.Validator | None = None,
    ) -> str | None:
        """Render a text prompt only on the local terminal."""
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

    async def ask_text_terminal_async(
        self,
        message: str,
        *,
        default: str = "",
        multiline: bool = False,
        validate: questionary.Validator | None = None,
    ) -> str | None:
        """Render a local text prompt while allowing web input to race it."""
        question = questionary.text(
            message,
            default=default,
            multiline=multiline,
            validate=validate,
            style=_QS_STYLE,
        )
        try:
            return await question.unsafe_ask_async()
        except (KeyboardInterrupt, EOFError):
            return None

"""Unified user-facing I/O for the kiui agent.

All console output, confirmation prompts, and log messages go through
AgentConsole so that theming, stderr redirection (pipe mode), and
interactive prompting are handled in one place.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import questionary
from rich.console import Console
from rich.live import Live
from rich.status import Status
from rich.table import Table
from rich.theme import Theme

if TYPE_CHECKING:
    from rich.text import Text
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


class ThinkingIndicator:
    """Context manager that shows an animated "Working... (Xs)" line
    while the model is generating a response.

    Uses Rich's ``Status`` with a bouncing-ball spinner in a background
    thread to update the elapsed-time counter every second.
    """

    def __init__(self, console: Console, events: EventHub | None = None, status_suffix: str = ""):
        self._console = console
        self._events = events
        self._status_suffix = status_suffix
        self._status: Status | None = None
        self._start_time: float = 0
        self._running = False
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "ThinkingIndicator":
        self._start_time = time.monotonic()
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
            self._events.publish("thinking_start", suffix=self._status_suffix)
        return self

    def __exit__(self, *args) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._status is not None:
            self._status.stop()
        if self._events is not None:
            self._events.publish("thinking_stop")

    def _label(self, elapsed: float) -> str:
        base = f"Working... ({elapsed:.0f}s)" if elapsed else "Working..."
        return f"{base} · {self._status_suffix}" if self._status_suffix else base

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
    """Live-rendering sink for a streamed assistant turn.

    Terminal: accumulated visible content is re-rendered as Markdown in a
    ``rich.Live`` region on each token; reasoning ("thinking") text streams
    above it in a dim block when ``show_thinking`` is set. The Live region is
    non-transient, so ``close()`` simply stops it and its final frame stays in
    scrollback. We do NOT reprint the content statically: combining a transient
    Live with ``vertical_overflow="visible"`` cannot reliably erase a response
    taller than the terminal, so a reprint duplicated the overflowed lines.

    Web: every fragment is published as ``assistant_delta`` / ``thinking_delta``
    events; on close the full reasoning and message are emitted once more as
    ``thinking`` / ``assistant_message`` so late-joining clients get the whole
    turn.
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
        self._live: Live | None = None
        self._closed = False

    def __enter__(self) -> "ResponseStream":
        # Non-transient: the final frame is left in scrollback on stop(), so no
        # static reprint is needed (and none happens) on close().
        self._live = Live(
            "",
            console=self._console,
            refresh_per_second=12,
            transient=False,
            vertical_overflow="visible",
        )
        self._live.start()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _renderable(self):
        from rich.console import Group
        from rich.markdown import Markdown
        from rich.text import Text as RichText

        parts = []
        if self._thinking:
            parts.append(RichText(self._thinking, style="thinking"))
        if self._content_visible:
            parts.append(Markdown(f"{_DOT} {self._content}"))
        return Group(*parts) if parts else RichText("")

    def _refresh(self) -> None:
        if self._live is not None:
            try:
                self._live.update(self._renderable())
            except Exception:
                pass  # display glitches must never abort the response

    def on_content(self, text: str) -> None:
        self._content += text
        if not self._content_visible:
            if not self._content.strip():
                return
            self._content_visible = True
            text = self._content
        self._refresh()
        if self._events is not None:
            self._events.publish("assistant_delta", text=text)

    def on_thinking(self, text: str) -> None:
        # show_thinking gates reasoning output uniformly across terminal and
        # web, so the two surfaces never disagree on whether a turn had a
        # thinking block.
        if not self._show_thinking:
            return
        self._thinking += text
        self._refresh()
        if self._events is not None:
            self._events.publish("thinking_delta", text=text)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._live is not None:
            self._live.stop()
            self._live = None
        # The non-transient Live leaves its final frame in scrollback, so the
        # full turn survives without a static reprint. Only web clients need
        # the consolidated events below.
        if self._events is not None:
            if self._thinking:
                self._events.publish("thinking", text=self._thinking)
            if self._content_visible:
                self._events.publish("assistant_message", text=self._content)


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

    def thinking(self, *, status_suffix: str = "") -> ThinkingIndicator:
        """Return a context manager that displays an animated thinking indicator.

        ``status_suffix`` is appended after the elapsed-time counter (e.g. a
        token/context summary) in both the terminal status bar and the web UI.

        Usage::

            with console.thinking():
                response = client.chat.completions.create(...)
        """
        return ThinkingIndicator(self._console, self.events, status_suffix=status_suffix)

    def stream_response(self, *, show_thinking: bool = False) -> ResponseStream:
        """Return a live-rendering sink for a streamed assistant turn.

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

    def rule(self):
        self._console.rule(style="dim color(240)")
        self._emit("rule")

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
        from rich.markdown import Markdown
        self._console.print(Markdown(f"{_DOT} {msg}"))
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

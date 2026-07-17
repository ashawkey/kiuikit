"""Tests for Rich terminal UI helpers."""

from io import StringIO

from rich.console import Console
from rich.progress_bar import ProgressBar
from rich.table import Table

from kiui.agent.io import EventHub
from kiui.agent.ui import AgentConsole, ContextStatus, ThinkingIndicator


def test_context_status_renders_progress_bar():
    status = ContextStatus(tokens=58_784, limit=1_000_000, total_tokens_used=418_431)

    assert status.fraction == 0.058784
    assert status.plain() == "6% · 418K used"

    rendered = status.render()
    assert isinstance(rendered, Table)
    assert any(isinstance(renderable, ProgressBar) for renderable in rendered.columns[0]._cells)


def test_thinking_indicator_publishes_structured_context_status():
    events = EventHub()
    status = ContextStatus(tokens=750, limit=1_000, total_tokens_used=2_000)
    indicator = ThinkingIndicator(Console(), events, status_suffix=status)

    with indicator:
        pass

    event = next(event for event in events.after(0) if event.type == "thinking_start")
    assert event.data == {
        "suffix": "75% · 2K used",
        "context_tokens": 750,
        "context_limit": 1_000,
        "total_tokens_used": 2_000,
        "label": "Working",
        "progress": False,
    }


def test_thinking_indicator_renders_indeterminate_progress():
    console = Console()
    indicator = ThinkingIndicator(
        console,
        status_suffix="436 messages, ~305,603 tokens",
        label="Compacting",
        progress=True,
    )

    label = indicator._label(2)

    assert isinstance(label, Table)
    assert any(isinstance(renderable, ProgressBar) for renderable in label.columns[1]._cells)
    with console.capture() as capture:
        console.print(label)
    assert "Compacting... (2s)" in capture.get()
    assert "436 messages, ~305,603 tokens" in capture.get()


def test_console_reset_timeline_emits_web_reset():
    events = EventHub()

    AgentConsole(events=events).reset_timeline()

    assert events.after(0)[0].type == "timeline_reset"


def test_response_renders_markdown_from_first_line():
    output = StringIO()
    console = AgentConsole()
    console._console = Console(file=output, width=60, force_terminal=True)

    console.response("```python\nprint(1)\n```")

    rendered = output.getvalue()
    assert "```python" not in rendered
    assert "print" in rendered


def test_thinking_indicator_includes_context_progress():
    console = Console()
    status = ContextStatus(tokens=750, limit=1_000, total_tokens_used=2_000)

    label = ThinkingIndicator(console, status_suffix=status)._label(2)

    assert isinstance(label, Table)
    with console.capture() as capture:
        console.print(label)
    text = capture.get()
    assert "Working... (2s)" in text
    assert "ctx" not in text
    assert "75%" in text
    assert "2K used" in text

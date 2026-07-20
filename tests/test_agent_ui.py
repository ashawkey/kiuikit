"""Tests for Rich terminal UI helpers."""

from io import StringIO

from rich.console import Console
from rich.progress_bar import ProgressBar
from rich.table import Table

from kiui.agent.utils.io import EventHub
from kiui.agent.ui import _AGENT_LOGO, AgentConsole, ContextStatus, ThinkingIndicator


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
    started_at = event.data["started_at"]
    assert isinstance(started_at, float)
    assert event.data == {
        "suffix": "75% · 2K used",
        "context_tokens": 750,
        "context_limit": 1_000,
        "total_tokens_used": 2_000,
        "started_at": started_at,
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


def test_startup_panel_renders_agent_details():
    output = StringIO()
    console = AgentConsole()
    console._console = Console(file=output, width=100, force_terminal=False)

    console.startup_panel(
        model="openai/gpt-5",
        context="258,000 tokens",
        reasoning="openai · high effort",
        permission="auto",
        persona="coder",
        skills="4 available · ~468 tokens (11.2% of prompt)",
        workspace="/home/user/project",
    )

    rendered = output.getvalue()
    rendered_lines = rendered.splitlines()
    logo_lines = _AGENT_LOGO.splitlines()
    rendered_logo = [
        line[3:3 + len(logo_lines[0])]
        for line in rendered_lines[1:1 + len(logo_lines)]
    ]
    assert rendered_logo == logo_lines
    assert "Model         openai/gpt-5 (258,000 tokens) · openai · high effort" in rendered_lines[1]
    assert "Permission    auto" in rendered_lines[2]
    assert "Persona       coder" in rendered_lines[3]
    assert "kia" not in rendered_lines[0]
    assert "terminal AI agent" not in rendered_lines[-1]
    for value in (
        "openai/gpt-5",
        "258,000 tokens",
        "openai · high effort",
        "auto",
        "coder",
        "4 available",
        "/home/user/project",
    ):
        assert value in rendered


def test_session_end_panel_renders_usage_and_resume():
    output = StringIO()
    console = AgentConsole()
    console._console = Console(file=output, width=120, force_terminal=False)

    console.session_end_panel(
        total=1234,
        prompt=900,
        cached_prompt=400,
        completion=300,
        reasoning=34,
        resume="20260719_193251",
    )

    rendered = output.getvalue()
    assert "1,234 total · 900 input · 400 cached input · 300 output · 34 reasoning" in rendered
    assert "kia --resume 20260719_193251" in rendered
    assert _AGENT_LOGO.splitlines()[0].strip() not in rendered
    assert "compaction" not in rendered.lower()
    assert "Skills" not in rendered


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

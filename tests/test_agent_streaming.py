"""Tests for streamed chat-completion accumulation."""

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from rich.markdown import Markdown
from rich.table import Table

from kiui.agent.utils.interrupt import RequestInterrupted
from kiui.agent.utils.io import EventHub
from kiui.agent.utils.streaming import consume_stream
from kiui.agent.ui import AgentConsole


def _delta(content=None, reasoning=None, reasoning_content=None, tool_calls=None, role=None):
    return NS(
        role=role,
        content=content,
        reasoning=reasoning,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls,
        model_extra={},
    )


def _chunk(delta=None, usage=None, finish_reason=None):
    choices = [] if delta is None else [NS(delta=delta, finish_reason=finish_reason)]
    return NS(choices=choices, usage=usage)


def _tc(index, id=None, name=None, args=None):
    return NS(index=index, id=id, function=NS(name=name, arguments=args))


def _usage(**kw):
    kw.setdefault("prompt_tokens_details", None)
    kw.setdefault("completion_tokens_details", None)
    return NS(**kw)


def test_consume_stream_accumulates_content_and_calls_back():
    parts = []
    stream = [
        _chunk(_delta(role="assistant", content="Hel")),
        _chunk(_delta(content="lo")),
        _chunk(delta=None, usage=_usage(total_tokens=3, prompt_tokens=1, completion_tokens=2)),
    ]
    message, usage, finish_reason = consume_stream(stream, on_content=parts.append)
    assert message["content"] == "Hello"
    assert "tool_calls" not in message
    assert parts == ["Hel", "lo"]
    assert usage.total_tokens == 3
    assert finish_reason is None


def test_consume_stream_reassembles_tool_call_fragments():
    stream = [
        _chunk(_delta(tool_calls=[_tc(0, id="call_1", name="read_file", args='{"file":')])),
        _chunk(_delta(tool_calls=[_tc(0, args=' "a.py"}')])),
        _chunk(delta=None, usage=_usage(total_tokens=5, prompt_tokens=3, completion_tokens=2)),
    ]
    message, _, _ = consume_stream(stream)
    assert message["content"] is None
    assert len(message["tool_calls"]) == 1
    tc = message["tool_calls"][0]
    assert tc["id"] == "call_1"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "read_file"
    assert tc["function"]["arguments"] == '{"file": "a.py"}'


def test_consume_stream_captures_finish_reason():
    stream = [_chunk(_delta(content="partial"), finish_reason="length")]
    message, _, finish_reason = consume_stream(stream)
    assert message["content"] == "partial"
    assert finish_reason == "length"


def test_consume_stream_captures_reasoning_variants():
    thoughts = []
    stream = [
        _chunk(_delta(reasoning_content="think ")),   # deepseek-style
        _chunk(_delta(reasoning="more")),             # openai/gemini-style
        _chunk(_delta(content="answer")),
    ]
    message, _, _ = consume_stream(stream, on_thinking=thoughts.append)
    assert thoughts == ["think ", "more"]
    assert message["reasoning_content"] == "think more"
    assert message["content"] == "answer"


def test_consume_stream_stops_early_on_should_stop():
    parts = []
    seen = {"n": 0}

    def gate():
        seen["n"] += 1
        return seen["n"] > 1  # stop after first check

    stream = [
        _chunk(_delta(content="a")),
        _chunk(_delta(content="b")),
        _chunk(_delta(content="c")),
    ]
    with pytest.raises(RequestInterrupted):
        consume_stream(stream, on_content=parts.append, should_stop=gate)
    assert parts == ["a"]


def _stream_events(hub):
    return [(e.type, e.data.get("text")) for e in hub.after(0)]


def test_response_stream_emits_reasoning_events_only_when_shown():
    """show_thinking gates reasoning uniformly: no thinking events when off."""
    hub = EventHub()
    console = AgentConsole(events=hub)
    with console.stream_response(show_thinking=False) as sink:
        sink.on_thinking("secret reasoning")
        sink.on_content("answer")

    events = _stream_events(hub)
    assert ("thinking_delta", "secret reasoning") not in events
    assert all(etype != "thinking" for etype, _ in events)
    assert ("assistant_delta", "answer") in events
    assert ("assistant_message", "answer") in events


def test_response_stream_finalizes_reasoning_when_shown():
    hub = EventHub()
    console = AgentConsole(events=hub)
    with console.stream_response(show_thinking=True) as sink:
        sink.on_thinking("let me think")
        sink.on_content("answer")

    events = _stream_events(hub)
    assert ("thinking_delta", "let me think") in events
    # A finalizing thinking event lets late-joining clients rebuild the block.
    assert ("thinking", "let me think") in events
    assert ("assistant_message", "answer") in events


def test_response_stream_renders_terminal_markdown_once_on_close():
    console = Mock()
    sink = AgentConsole().stream_response()
    sink._console = console

    with sink:
        sink.on_content("# Head")
        sink.on_content("ing")
        console.print.assert_not_called()

    console.print.assert_called_once()
    rendered = console.print.call_args.args[0]
    assert isinstance(rendered, Table)
    assert rendered.columns[0]._cells[0].plain == "• "
    markdown = rendered.columns[1]._cells[0]
    assert isinstance(markdown, Markdown)
    assert markdown.markup == "# Heading"

    sink.close()
    sink.on_content(" ignored")
    console.print.assert_called_once()


def test_response_stream_does_not_render_failed_attempt():
    console = Mock()
    sink = AgentConsole().stream_response()
    sink._console = console

    try:
        with sink:
            sink.on_content("partial response")
            raise RuntimeError("stream failed")
    except RuntimeError:
        pass

    console.print.assert_not_called()

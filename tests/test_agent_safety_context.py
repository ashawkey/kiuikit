"""Regression tests for agent safety routing, compaction, and stream cleanup."""

from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest

from kiui.agent.backend import LLMAgent
from kiui.agent.context import compact_context
from kiui.agent.interrupt import RequestInterrupted
from kiui.agent.permissions import PermissionController, PermissionMode


class _Console:
    prompt_broker = None

    def __init__(self):
        self.results = []

    def print(self, *args, **kwargs):
        pass

    def tool(self, *args, **kwargs):
        pass

    def tool_result(self, message, success):
        self.results.append((message, success))

    def thinking(self, **kwargs):
        return nullcontext()

    def stream_response(self, **kwargs):
        sink = NS(on_content=lambda text: None, on_thinking=lambda text: None)
        return nullcontext(sink)

    def warn(self, *args, **kwargs):
        pass


def test_direct_bash_routes_through_safety_guard(tmp_path):
    console = _Console()
    executed = []
    agent = NS(
        console=console,
        permissions=PermissionController(
            mode=PermissionMode.AUTO, console=console, work_dir=tmp_path
        ),
        tool_executor=NS(execute=lambda *args: executed.append(args)),
        _operation=lambda label: nullcontext(),
    )

    LLMAgent._run_bash_command(agent, "mkfs.ext4 /dev/example")

    assert executed == []
    assert console.results and console.results[-1][1] is False


def test_failed_compaction_preserves_original_messages():
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
    ]
    client = NS(
        chat=NS(completions=NS(create=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("offline"))))
    )

    result = compact_context(messages, client, "model")

    assert result == messages
    assert result is not messages


def test_stream_is_closed_when_consumption_is_cancelled(monkeypatch):
    stream = NS(close_calls=0)

    def close():
        stream.close_calls += 1

    stream.close = close
    calls = 0

    def interruptible(fn, cancellation):
        nonlocal calls
        calls += 1
        if calls == 1:
            return stream
        raise RequestInterrupted()

    monkeypatch.setattr("kiui.agent.backend.run_interruptible", interruptible)
    agent = NS(
        console=_Console(),
        client=NS(chat=NS(completions=NS(create=lambda **kwargs: stream))),
        cancellation=None,
        show_thinking=False,
        _status_suffix=lambda: "",
    )

    with pytest.raises(RequestInterrupted):
        LLMAgent._stream_completion(agent, {})

    assert stream.close_calls == 1

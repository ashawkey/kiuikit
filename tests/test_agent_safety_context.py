"""Regression tests for agent safety routing, compaction, and stream cleanup."""

import json
import os
import sys
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest

import kiui.agent.backend as backend
import kiui.agent.tools.results as tool_results
from kiui.agent.backend import LLMAgent
from kiui.agent.context import (
    ContextManager,
    ToolResultEnvelope,
    compact_context,
    compact_tool_result_envelope,
    estimate_context_chars,
    tool_result_char_budget,
)
from kiui.agent.utils.interrupt import RequestInterrupted
from kiui.agent.permissions import PermissionController, PermissionMode
from kiui.agent.tools import ToolExecutor


def _compact_text(
    text, tool_name, context_length=16_000, artifact_path=None, arguments=None
):
    result = compact_tool_result_envelope(
        ToolResultEnvelope(tool_name, arguments or {"command": ""}, {}, text),
        context_length,
        artifact_path=artifact_path,
    )
    return result.text, result.compacted


class _Console:
    prompt_broker = None

    def __init__(self):
        self.results = []
        self.thinking_calls = []

    def print(self, *args, **kwargs):
        pass

    def tool(self, *args, **kwargs):
        pass

    def tool_result(self, message, success):
        self.results.append((message, success))

    def thinking(self, **kwargs):
        self.thinking_calls.append(kwargs)
        return nullcontext()

    def stream_response(self, **kwargs):
        sink = NS(on_content=lambda text: None, on_thinking=lambda text: None)
        return nullcontext(sink)

    def warn(self, *args, **kwargs):
        pass


@pytest.mark.skipif(sys.platform == "win32", reason="Unix shell command semantics")
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


def test_actual_failed_exec_capture_is_compacted_and_persisted(tmp_path):
    console = _Console()
    console.system = lambda *args, **kwargs: None
    executor = ToolExecutor(console=console, work_dir=str(tmp_path))
    command = (
        "python -c \"import sys; print('HEAD'); print('x' * 5000); "
        "print('brief stderr', file=sys.stderr); sys.exit(1)\""
    )
    tool_call = {"id": "call-real-failed", "type": "function", "function": {"name": "exec_command", "arguments": json.dumps({"command": command})}}
    agent = NS(
        verbose=False,
        console=console,
        permissions=NS(check=lambda *args: (True, "")),
        tool_executor=executor,
        context_length=16_000,
        token_estimator=NS(chars_per_token=3.3),
        context=NS(messages=[], add=lambda message: agent.context.messages.append(message)),
        cancellation=None,
        work_dir=str(tmp_path),
        _session_id="test",
        round_id=3,
        tool_compaction_totals={"calls": 0, "original_chars": 0, "retained_chars": 0},
    )

    LLMAgent.execute_tool_calls(agent, [tool_call])

    stored = agent.context.messages[-1]["content"]
    assert "Large exec_command result compacted" in stored
    assert "brief stderr" in stored
    artifact = (
        tmp_path / ".kia" / "tool-results" / "test"
        / "r3-call-real-failed-exec_command.txt"
    )
    captured = artifact.read_text()
    assert "HEAD" in captured and "brief stderr" in captured and len(captured) > 5_000


def test_large_tool_result_is_persisted_before_context(tmp_path):
    console = _Console()
    console.system = lambda *args, **kwargs: None
    result_text = "first\n" + "noise\n" * 3000 + "ERROR final\n"
    tool_call = {"id": "call-1", "type": "function", "function": {"name": "exec_command", "arguments": '{"command": "noisy"}'}}
    agent = NS(
        verbose=False,
        console=console,
        permissions=NS(check=lambda *args: (True, "")),
        tool_executor=NS(execute=lambda *args: {"stdout": result_text, "exit_code": 0, "success": True}),
        context_length=16_000,
        token_estimator=NS(chars_per_token=3.3),
        context=NS(messages=[], add=lambda message: agent.context.messages.append(message)),
        cancellation=None,
        work_dir=str(tmp_path),
        _session_id="test",
        round_id=2,
        tool_compaction_totals={"calls": 0, "original_chars": 0, "retained_chars": 0},
    )

    interrupted = LLMAgent.execute_tool_calls(agent, [tool_call])

    assert not interrupted
    stored = agent.context.messages[-1]["content"]
    assert len(stored) <= tool_result_char_budget(16_000, tool_name="exec_command")
    assert "Large exec_command result compacted" in stored
    artifact = tmp_path / ".kia" / "tool-results" / "test" / "r2-call-1-exec_command.txt"
    assert artifact.read_text() == result_text
    if os.name == "posix":  # Windows has no Unix permission bits
        assert artifact.stat().st_mode & 0o077 == 0


def test_call_api_sets_output_limit_and_rejects_truncated_response():
    kwargs_seen = {}
    added = []
    max_output_tokens = 20_000
    usage = NS(prompt_tokens=1, completion_tokens=max_output_tokens, total_tokens=max_output_tokens + 1)

    def completion(kwargs):
        kwargs_seen.update(kwargs)
        return {"role": "assistant", "content": "partial"}, usage, "length"

    agent = NS(
        context_length=0,
        model="test-model",
        max_output_tokens=max_output_tokens,
        INITIAL_BACKOFF=1.0,
        MAX_BACKOFF=64.0,
        verbose=False,
        round_id=1,
        _pending_images=[],
        _messages_with_pending_images=lambda: [],
        stream=False,
        tools=[],
        profile=NS(reasoning=None),
        reasoning_effort="high",
        _blocking_completion=completion,
        cancellation=None,
        _accumulate_usage=lambda value: None,
        token_estimator=NS(calibrate=lambda *args: None),
        context=NS(add=added.append),
    )

    with pytest.raises(RuntimeError, match="finish_reason='length'"):
        LLMAgent.call_api(agent)

    assert kwargs_seen["max_tokens"] == max_output_tokens
    assert added == []


def test_stream_is_closed_when_consumption_is_cancelled(monkeypatch):
    class Stream(list):
        close_calls = 0

        def close(self):
            self.close_calls += 1

    stream = Stream()
    client = NS(
        close=lambda: None,
        chat=NS(completions=NS(create=lambda **kwargs: stream)),
    )

    def interruptible(fn, cancellation, on_cancel=None):
        fn()
        raise RequestInterrupted()

    monkeypatch.setattr("kiui.agent.backend.run_interruptible", interruptible)
    agent = NS(
        console=_Console(),
        cancellation=None,
        show_thinking=False,
        _status_suffix=lambda: "",
        _request_client=lambda: client,
    )

    with pytest.raises(RequestInterrupted):
        LLMAgent._stream_completion(agent, {})

    assert stream.close_calls == 1


def _tool_call_msg(name, args_json):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": name, "arguments": args_json}}
        ],
    }

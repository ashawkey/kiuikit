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
from kiui.agent.providers import CompletionResult, ProviderUsage
from kiui.agent.context import (
    ContextManager,
    ToolResultEnvelope,
    compact_context,
    compact_tool_result_envelope,
    estimate_context_chars,
    tool_result_char_budget,
)
from kiui.agent.utils.interrupt import RequestInterrupted
from kiui.agent.utils.io import EventHub, InputBroker
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


def test_compaction_uses_provider_neutral_summarizer():
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
        {"role": "assistant", "content": "fourth"},
    ]
    prompts = []

    def summarize(prompt):
        prompts.append(prompt)
        return "condensed history"

    result = compact_context(messages, summarize)

    assert prompts and "first" in prompts[0]
    assert result[0]["content"] == "[Previous conversation summary]\ncondensed history"


def test_failed_compaction_preserves_original_messages():
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
    ]
    def fail(_prompt):
        raise RuntimeError("offline")

    result = compact_context(messages, fail)

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


def test_pending_message_steers_next_agentic_iteration():
    context = ContextManager("system")
    context.add({"role": "user", "content": "start"})
    broker = InputBroker(EventHub())
    user_inputs = []
    request_contexts = []
    tool_call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "read_file", "arguments": '{"file": "a.py"}'},
    }
    responses = [
        {"role": "assistant", "content": None, "tool_calls": [tool_call]},
        {"role": "assistant", "content": "done"},
    ]

    def call_api():
        request_contexts.append(list(context.messages))
        message = responses.pop(0)
        context.add(message)
        return message

    def execute_tool_calls(_tool_calls):
        context.add({
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "file contents",
        })
        broker.submit("use @config.py instead", source="web")
        return False

    console = NS(
        response=lambda text: None,
        user_input=lambda text, **kwargs: user_inputs.append((text, kwargs)),
    )
    agent = NS(
        verbose=False,
        stream=False,
        context=context,
        input_broker=broker,
        console=console,
        call_api=call_api,
        execute_tool_calls=execute_tool_calls,
        _last_interrupted=False,
        round_id=4,
    )
    agent._inject_pending_steer = lambda: LLMAgent._inject_pending_steer(agent)

    assert LLMAgent.get_response(agent) == "done"
    assert [message["role"] for message in request_contexts[1]] == [
        "user", "assistant", "tool", "user",
    ]
    assert request_contexts[1][-1]["content"] == "use config.py instead"
    assert broker.submission is None
    assert user_inputs[0][0] == "use config.py instead"
    assert user_inputs[0][1]["source"] == "web"
    assert agent.round_id == 4


@pytest.mark.parametrize("query", ["/help", "!git status", "exit", "quit"])
def test_pending_local_query_waits_for_round_completion(query):
    broker = InputBroker(EventHub())
    submission = broker.submit(query)
    context = ContextManager("system")
    agent = NS(
        input_broker=broker,
        context=context,
        console=NS(user_input=lambda *args, **kwargs: None),
    )

    assert not LLMAgent._inject_pending_steer(agent)
    assert broker.submission == submission
    assert context.messages == []


def test_interrupted_tool_iteration_does_not_consume_pending_message():
    context = ContextManager("system")
    context.add({"role": "user", "content": "start"})
    broker = InputBroker(EventHub())
    pending = None
    tool_call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "read_file", "arguments": '{"file": "a.py"}'},
    }

    def call_api():
        message = {"role": "assistant", "content": None, "tool_calls": [tool_call]}
        context.add(message)
        return message

    def execute_tool_calls(_tool_calls):
        nonlocal pending
        pending = broker.submit("steer later")
        return True

    agent = NS(
        verbose=False,
        stream=False,
        context=context,
        input_broker=broker,
        console=NS(response=lambda text: None, system=lambda text: None),
        call_api=call_api,
        execute_tool_calls=execute_tool_calls,
        _pending_images=[],
        _last_interrupted=False,
    )
    agent._inject_pending_steer = lambda: LLMAgent._inject_pending_steer(agent)

    assert LLMAgent.get_response(agent) is None
    assert broker.submission == pending
    assert agent._last_interrupted


def test_call_api_sets_output_limit_and_rejects_truncated_response():
    kwargs_seen = {}
    added = []
    max_output_tokens = 20_000
    usage = ProviderUsage(
        prompt_tokens=1,
        completion_tokens=max_output_tokens,
        total_tokens=max_output_tokens + 1,
    )

    def completion(request):
        kwargs_seen.update({
            "model": request.model,
            "max_tokens": request.max_output_tokens,
        })
        return CompletionResult(
            {"role": "assistant", "content": "partial"}, usage, "length"
        )

    agent = NS(
        context_length=0,
        model="test-model",
        max_output_tokens=max_output_tokens,
        INITIAL_BACKOFF=1.0,
        MAX_BACKOFF=64.0,
        verbose=False,
        round_id=1,
        _session_id=None,
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
    class Stream:
        close_calls = 0

        def consume(self, **kwargs):
            return CompletionResult(
                {"role": "assistant", "content": "partial"}, None, None
            )

        def close(self):
            self.close_calls += 1

    stream = Stream()
    provider = NS(open_stream=lambda request: stream, cancel=lambda: None)

    def interruptible(fn, cancellation, on_cancel=None):
        fn()
        raise RequestInterrupted()

    monkeypatch.setattr("kiui.agent.backend.run_interruptible", interruptible)
    agent = NS(
        console=_Console(),
        cancellation=None,
        show_thinking=False,
        _status_suffix=lambda: "",
        provider=provider,
    )

    request = NS(stream=True)
    with pytest.raises(RequestInterrupted):
        LLMAgent._stream_completion(agent, request)

    assert stream.close_calls == 1


def test_stream_status_remains_active_while_body_is_consumed(monkeypatch):
    state = {"thinking": False}

    class Thinking:
        def __enter__(self):
            state["thinking"] = True

        def __exit__(self, *args):
            state["thinking"] = False

    class Console(_Console):
        def thinking(self, **kwargs):
            return Thinking()

    class Stream:
        def consume(self, **kwargs):
            assert state["thinking"] is True
            return CompletionResult(
                {"role": "assistant", "content": "done"}, None, "stop"
            )

        def close(self):
            pass

    monkeypatch.setattr(
        "kiui.agent.backend.run_interruptible",
        lambda fn, cancellation, on_cancel=None: fn(),
    )
    agent = NS(
        console=Console(),
        cancellation=None,
        show_thinking=False,
        _status_suffix=lambda: "",
        provider=NS(open_stream=lambda request: Stream(), cancel=lambda: None),
    )

    LLMAgent._stream_completion(agent, NS(stream=True))

    assert state["thinking"] is False


def _tool_call_msg(name, args_json):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": name, "arguments": args_json}}
        ],
    }

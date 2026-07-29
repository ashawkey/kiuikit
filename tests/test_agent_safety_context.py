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
    SUMMARY_MARKER,
    ContextManager,
    TokenEstimator,
    ToolResultEnvelope,
    _artifact_path_in,
    compact_context,
    compact_tool_result_envelope,
    estimate_context_chars,
    get_role,
    get_text,
    tool_result_char_budget,
)
from kiui.agent.utils.interrupt import RequestInterrupted
from kiui.agent.utils.io import EventHub, InputBroker
from kiui.agent.permissions import PermissionController, PermissionMode
from kiui.agent.tools import ToolExecutor


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

    result, _ = compact_context(messages, summarize)

    assert prompts and "first" in prompts[0]
    content = result[0]["content"]
    assert content.startswith("[Previous conversation summary]")
    assert "## Original request\nfirst" in content
    assert "condensed history" in content


def test_failed_compaction_is_reported_as_no_compaction():
    """A failed summarization must be indistinguishable from 'nothing to do'.

    ``_run_compaction`` reads identity to decide whether the context shrank, so
    handing back a copy would make it announce a successful 0 % compaction.
    """
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
    ]
    def fail(_prompt):
        raise RuntimeError("offline")

    result, _ = compact_context(messages, fail)

    assert result is messages


def test_cancelling_a_compaction_is_not_reported_as_a_failure():
    """Escape tears down the request; that is the user's doing, not an error."""
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
    ]
    warnings = []
    console = _Console()
    console.warn = lambda msg, **kwargs: warnings.append(msg)

    def cancel(_prompt):
        raise RequestInterrupted()

    with pytest.raises(RequestInterrupted):
        compact_context(messages, cancel, console=console)

    assert warnings == []


@pytest.mark.parametrize(
    "error",
    [RequestInterrupted(), RuntimeError("400 prompt is too long")],
    ids=["cancelled", "rejected"],
)
def test_a_failed_request_keeps_the_context_management_it_paid_for(error):
    """Eviction and compaction are window maintenance, not turn state.

    Rolling them back would discard a summarization round-trip already spent —
    and on the overflow-retry path, the exact remediation the next attempt needs.
    """
    context = ContextManager("system prompt")
    for i in range(4):
        context.add({"role": "user", "content": f"message {i}"})

    console = _Console()
    console.system = lambda *args, **kwargs: None
    console.error = lambda *args, **kwargs: None

    def call_api():
        # Stands in for the eviction + compaction call_api runs before the
        # request; the assistant message is only ever appended on success.
        context.replace_messages([{"role": "user", "content": SUMMARY_MARKER}])
        raise error

    agent = NS(
        console=console,
        context=context,
        verbose=False,
        call_api=call_api,
        _pending_images=[],
        _last_interrupted=False,
    )

    assert LLMAgent.get_response(agent) is None
    assert [get_text(m) for m in context.messages] == [SUMMARY_MARKER]


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


def test_midsize_exec_capture_survives_for_eviction(tmp_path):
    """Output too small to compact but big enough to trim must stay recoverable.

    A result between SOFT_TRIM_THRESHOLD and the ingress budget enters history
    whole, so nothing is persisted for it — and layer 2 later cuts it to ~3k
    chars. A command's output cannot be produced again, so the capture is kept
    and pointed at even though no ingress compaction happens.
    """
    console = _Console()
    console.system = lambda *args, **kwargs: None
    executor = ToolExecutor(console=console, work_dir=str(tmp_path))
    command = "python -c \"print('HEAD'); print('x' * 8000); print('TAIL')\""
    tool_call = {"id": "call-mid", "type": "function", "function": {"name": "exec_command", "arguments": json.dumps({"command": command})}}
    agent = NS(
        verbose=False,
        console=console,
        permissions=NS(check=lambda *args: (True, "")),
        tool_executor=executor,
        context_length=200_000,  # exec budget 12k, so 8k of output is not compacted
        token_estimator=NS(chars_per_token=3.3),
        context=NS(messages=[], add=lambda message: agent.context.messages.append(message)),
        cancellation=None,
        work_dir=str(tmp_path),
        _session_id="midsize",
        round_id=4,
        tool_compaction_totals={"calls": 0, "original_chars": 0, "retained_chars": 0},
    )

    LLMAgent.execute_tool_calls(agent, [tool_call])

    stored = agent.context.messages[-1]["content"]
    assert "Large exec_command result compacted" not in stored  # entered whole
    assert "HEAD" in stored and "TAIL" in stored

    # Layer 2 recovers the capture by parsing the pointer out of the message.
    pointer = _artifact_path_in(stored)
    assert pointer is not None
    captured = (tmp_path / pointer).read_text()
    assert "HEAD" in captured and "TAIL" in captured and len(captured) > 8_000


def test_old_session_captures_are_pruned(tmp_path):
    root = tmp_path / ".kia" / "tool-results"
    for i in range(5):
        session = root / f"s{i}"
        session.mkdir(parents=True)
        (session / "r0-call-exec_command.txt").write_text("captured")
        os.utime(session, (i, i))  # s0 oldest … s4 newest
    live = root / "live"
    live.mkdir()
    os.utime(live, (0, 0))  # oldest on disk, but it is the running session

    removed = tool_results.prune_tool_result_artifacts(str(tmp_path), "live", keep=2)

    assert removed == 3
    assert {path.name for path in root.iterdir()} == {"live", "s3", "s4"}


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
    # No producer capture on this stub, so the formatted result is persisted:
    # the whole untruncated output plus the exec status line it ends with.
    captured = artifact.read_text()
    assert captured.startswith(result_text)
    assert captured.endswith("[exit_code: 0, interrupted: false, timed_out: false]")
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


def test_wait_submission_never_steers_current_round():
    broker = InputBroker(EventHub())
    submission = broker.submit("follow up", delay=0, steer=False)
    context = ContextManager("system")
    agent = NS(
        input_broker=broker,
        context=context,
        console=NS(user_input=lambda *args, **kwargs: None),
    )

    assert not LLMAgent._inject_pending_steer(agent)
    assert broker.submission == submission
    assert context.messages == []


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


def test_call_api_preserves_output_limited_response_for_continuation():
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
        token_estimator=NS(observe=lambda *args: None),
        context=NS(add=added.append),
    )

    message = LLMAgent.call_api(agent)

    assert kwargs_seen["max_tokens"] == max_output_tokens
    assert message["content"] == "partial"
    assert added == [message]
    assert agent._last_finish_reason == "length"


@pytest.mark.parametrize("unfinished_reason", [None, "length"])
def test_get_response_warns_and_automatically_continues(unfinished_reason):
    context = ContextManager("system")
    context.add({"role": "user", "content": "write it"})
    warnings = []
    responses = iter([
        ({"role": "assistant", "content": "partial "}, unfinished_reason),
        ({"role": "assistant", "content": "answer"}, "stop"),
    ])

    agent = NS(
        console=NS(
            warn=lambda text: warnings.append(text),
            response=lambda text: None,
            debug=lambda text: None,
            error=lambda text: None,
        ),
        context=context,
        verbose=False,
        stream=False,
        _pending_images=[],
        _last_interrupted=False,
        MAX_AUTO_CONTINUES=3,
    )

    def call_api():
        message, finish_reason = next(responses)
        agent._last_finish_reason = finish_reason
        context.add(message)
        return message

    agent.call_api = call_api

    assert LLMAgent.get_response(agent) == "answer"
    assert len(warnings) == 1
    assert "automatically continuing (1/3)" in warnings[0]
    assert [get_text(message) for message in context.messages] == [
        "write it", "partial ", "answer"
    ]


def test_get_response_continues_after_empty_stopped_response():
    """A "stop" round with no text and no tool call left the task mid-flight."""
    context = ContextManager("system")
    context.add({"role": "user", "content": "keep going"})
    warnings = []
    responses = iter([
        ({"role": "assistant", "content": None}, "stop"),
        ({"role": "assistant", "content": "answer"}, "stop"),
    ])

    agent = NS(
        console=NS(
            warn=lambda text: warnings.append(text),
            response=lambda text: None,
            debug=lambda text: None,
            error=lambda text: None,
        ),
        context=context,
        verbose=False,
        stream=False,
        _pending_images=[],
        _last_interrupted=False,
        MAX_AUTO_CONTINUES=3,
    )

    def call_api():
        message, finish_reason = next(responses)
        agent._last_finish_reason = finish_reason
        context.add(message)
        return message

    agent.call_api = call_api

    assert LLMAgent.get_response(agent) == "answer"
    assert len(warnings) == 1
    assert "no text or tool call" in warnings[0]
    assert "automatically continuing (1/3)" in warnings[0]


def test_get_response_stops_after_repeated_empty_responses():
    context = ContextManager("system")
    context.add({"role": "user", "content": "keep going"})
    warnings = []

    agent = NS(
        console=NS(
            warn=lambda text: warnings.append(text),
            response=lambda text: None,
            debug=lambda text: None,
            error=lambda text: None,
        ),
        context=context,
        verbose=False,
        stream=False,
        _pending_images=[],
        _last_interrupted=False,
        MAX_AUTO_CONTINUES=2,
    )

    def call_api():
        agent._last_finish_reason = "stop"
        message = {"role": "assistant", "content": ""}
        context.add(message)
        return message

    agent.call_api = call_api

    assert LLMAgent.get_response(agent) is None
    assert "still unfinished after 2 automatic continuations" in warnings[-1]
    # Nothing in a contentless assistant turn reaches the next request, so none
    # of them may accumulate in the history that every later round re-sends.
    assert [get_role(message) for message in context.messages] == ["user"]


def _unfinished_agent(context, warnings):
    agent = NS(
        console=NS(
            warn=lambda text: warnings.append(text),
            response=lambda text: None,
            debug=lambda text: None,
            error=lambda text: None,
        ),
        context=context,
        verbose=False,
        stream=False,
        _pending_images=[],
        _last_interrupted=False,
        MAX_AUTO_CONTINUES=3,
    )
    agent._resolve_unexecuted_tool_calls = (
        lambda message: LLMAgent._resolve_unexecuted_tool_calls(agent, message)
    )
    return agent


def test_truncated_tool_call_is_answered_so_history_stays_valid():
    """A cut-off tool call must not leave the turn unresolved.

    Providers reject an assistant message whose tool calls have no matching
    results, so an unanswered pair would fail every later request in the
    session rather than just this round.
    """
    context = ContextManager("system")
    context.add({"role": "user", "content": "run it"})
    warnings = []
    agent = _unfinished_agent(context, warnings)

    def call_api():
        agent._last_finish_reason = "length"
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call-cut",
                "type": "function",
                "function": {"name": "exec_command", "arguments": '{"command": "ls -'},
            }],
        }
        context.add(message)
        return message

    agent.call_api = call_api

    assert LLMAgent.get_response(agent) is None
    assert "cannot automatically continue safely" in warnings[-1]
    assert [get_role(message) for message in context.messages] == [
        "user", "assistant", "tool"
    ]
    result = context.messages[-1]
    assert result["tool_call_id"] == "call-cut"
    assert "never executed" in result["content"]


def test_truncated_tool_call_without_an_id_withdraws_the_message():
    """A call cut off before its id cannot be answered, so it cannot stay."""
    context = ContextManager("system")
    context.add({"role": "user", "content": "run it"})
    warnings = []
    agent = _unfinished_agent(context, warnings)

    def call_api():
        agent._last_finish_reason = "length"
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "", "type": "function", "function": {"name": "", "arguments": ""}}],
        }
        context.add(message)
        return message

    agent.call_api = call_api

    assert LLMAgent.get_response(agent) is None
    assert [get_role(message) for message in context.messages] == ["user"]


def test_empty_response_with_provider_state_is_kept_for_replay():
    """Provider state is what the continuation replays; dropping it loses the turn."""
    context = ContextManager("system")
    context.add({"role": "user", "content": "keep going"})
    warnings = []
    agent = _unfinished_agent(context, warnings)
    responses = iter([
        ({"role": "assistant", "content": None, "provider_state": {"openai-codex": {}}}, "stop"),
        ({"role": "assistant", "content": "answer"}, "stop"),
    ])

    def call_api():
        message, finish_reason = next(responses)
        agent._last_finish_reason = finish_reason
        context.add(message)
        return message

    agent.call_api = call_api

    assert LLMAgent.get_response(agent) == "answer"
    assert [get_role(message) for message in context.messages] == [
        "user", "assistant", "assistant"
    ]


def _compaction_agent(token_readings):
    """Agent stub for :meth:`LLMAgent._run_compaction`, driven by token readings.

    Messages are bulky on purpose: the history has to reach past the protected
    recent window (15 % of the 100k window) *and* leave a split big enough to
    clear the yield bar, otherwise compaction is a no-op and never reaches the
    floor bookkeeping these tests are about.
    """
    console = _Console()
    console.system = lambda *args, **kwargs: None
    context = ContextManager("system prompt")
    for i in range(10):
        context.add({"role": "user", "content": f"request {i} " + "x" * 6_000})
        context.add({"role": "assistant", "content": f"reply {i} " + "y" * 6_000})
    readings = iter(token_readings)
    return NS(
        console=console,
        context=context,
        context_length=100_000,
        max_output_tokens=0,
        token_estimator=TokenEstimator(),
        cancellation=None,
        provider=NS(cancel=lambda: None),
        compaction_totals={"count": 0, "tokens_before": 0, "tokens_after": 0},
        _compaction_floor_tokens=None,
        _context_tokens=lambda: next(readings),
        _snapshot_before_compaction=lambda: None,
        _summarize=lambda prompt: "## Goal\nsummary",
    )


def test_ineffective_compaction_sets_a_floor_against_repeating():
    agent = _compaction_agent([90_000, 89_000])

    assert LLMAgent._run_compaction(agent, "pressure") is True
    # Freed 1k of a 100k window, under the 5% yield bar: hold off until the
    # context has actually grown again.
    assert agent._compaction_floor_tokens == 94_000
    assert agent.compaction_totals["count"] == 1


def test_effective_compaction_still_holds_off_the_next_one():
    """Even a good pass must not leave the marginal pass behind it unguarded."""
    agent = _compaction_agent([90_000, 40_000])
    agent._compaction_floor_tokens = 94_000

    assert LLMAgent._run_compaction(agent, "pressure") is True
    # Freed 50k, well past the yield bar — but the floor still moves to where
    # this pass landed, so the next one waits for 5% of real growth instead of
    # firing again on the very next tool result.
    assert agent._compaction_floor_tokens == 45_000


def test_compaction_is_skipped_while_the_floor_holds():
    context = ContextManager("system prompt")
    for i in range(6):
        context.add({"role": "user", "content": "x" * 5_000})
        context.add({"role": "assistant", "content": "y" * 5_000})

    compactions = []
    usage = ProviderUsage(prompt_tokens=10, completion_tokens=1, total_tokens=11)
    console = _Console()
    console.system = lambda *args, **kwargs: None

    agent = NS(
        console=console,
        context=context,
        context_length=1_000,  # far below usage: compaction would normally fire
        token_estimator=TokenEstimator(),
        _compaction_floor_tokens=10**9,
        _run_compaction=lambda reason: compactions.append(reason) or True,
        model="test-model",
        max_output_tokens=100,
        INITIAL_BACKOFF=0.01,
        MAX_BACKOFF=0.02,
        verbose=False,
        round_id=1,
        _session_id=None,
        _pending_images=[],
        stream=False,
        tools=[],
        profile=NS(reasoning=None),
        reasoning_effort="high",
        cancellation=None,
        _accumulate_usage=lambda value: None,
        _blocking_completion=lambda request: CompletionResult(
            {"role": "assistant", "content": "ok"}, usage, "stop"
        ),
    )
    agent._context_tokens = lambda: agent.token_estimator.prompt_tokens(
        agent.context.total_chars
    )
    agent._messages_with_pending_images = lambda: agent.context.get()

    LLMAgent.call_api(agent)

    assert compactions == []


class _OverflowError(Exception):
    status_code = 400


def _overflow_agent(completion, compactions):
    console = _Console()
    console.system = lambda *args, **kwargs: None
    return NS(
        console=console,
        _interruptible_sleep=lambda seconds: None,
        context_length=0,
        model="test-model",
        max_output_tokens=1_000,
        INITIAL_BACKOFF=0.01,
        MAX_BACKOFF=0.02,
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
        token_estimator=NS(observe=lambda *args: None),
        context=NS(add=lambda message: None),
        _run_compaction=lambda reason: compactions.append(reason) or True,
    )


def test_context_overflow_compacts_and_retries_instead_of_failing():
    compactions = []
    calls = []
    usage = ProviderUsage(prompt_tokens=10, completion_tokens=1, total_tokens=11)

    def completion(request):
        calls.append(request)
        if len(calls) == 1:
            raise _OverflowError("maximum context length is 200000 tokens")
        return CompletionResult({"role": "assistant", "content": "ok"}, usage, "stop")

    message = LLMAgent.call_api(_overflow_agent(completion, compactions))

    assert compactions == ["Context overflow reported by the API"]
    assert len(calls) == 2
    assert message["content"] == "ok"


def test_context_overflow_recovery_is_attempted_only_once():
    compactions = []
    calls = []

    def completion(request):
        calls.append(request)
        raise _OverflowError("maximum context length is 200000 tokens")

    with pytest.raises(RuntimeError, match="API request rejected"):
        LLMAgent.call_api(_overflow_agent(completion, compactions))

    assert len(compactions) == 1
    assert len(calls) == 2


def test_rate_limit_mentioning_tokens_is_not_treated_as_overflow():
    """A retryable 429 must keep retrying, not trigger compaction."""
    compactions = []
    calls = []
    usage = ProviderUsage(prompt_tokens=10, completion_tokens=1, total_tokens=11)

    class _RateLimit(Exception):
        status_code = 429

    def completion(request):
        calls.append(request)
        if len(calls) == 1:
            raise _RateLimit("rate limit: too many tokens per minute")
        return CompletionResult({"role": "assistant", "content": "ok"}, usage, "stop")

    LLMAgent.call_api(_overflow_agent(completion, compactions))

    assert compactions == []
    assert len(calls) == 2


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

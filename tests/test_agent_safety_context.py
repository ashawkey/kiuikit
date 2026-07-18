"""Regression tests for agent safety routing, compaction, and stream cleanup."""

import json
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest

import kiui.agent.backend as backend
import kiui.agent.tool_results as tool_results
from kiui.agent.backend import LLMAgent
from kiui.agent.context import (
    ToolResultEnvelope,
    compact_context,
    compact_tool_result_envelope,
    tool_result_char_budget,
)
from kiui.agent.interrupt import RequestInterrupted
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


def test_execute_initializes_session_id(monkeypatch):
    console = _Console()
    console.system = lambda *args, **kwargs: None
    console.rule = lambda: None
    messages = []
    agent = NS(
        console=console,
        _session_id=None,
        context=NS(add=messages.append),
        _operation=lambda label: nullcontext(),
        get_response=lambda: "done",
        _print_token_summary=lambda: None,
    )
    monkeypatch.setattr(backend.time, "strftime", lambda pattern: "test-session")

    assert LLMAgent.execute(agent, "review files") == "done"
    assert agent._session_id == "test-session"
    assert messages == [{"role": "user", "content": "review files"}]


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


def test_tool_result_compaction_keeps_edges_diagnostics_and_guidance():
    text = "start\n" + "ordinary log line\n" * 2000 + "ERROR: build failed\n" + "final status\n"

    result, compacted = _compact_text(
        text,
        "exec_command",
        context_length=16_000,
        artifact_path=".kia/tool-results/test.txt",
    )

    assert compacted
    assert len(result) <= tool_result_char_budget(16_000, tool_name="exec_command")
    assert "start" in result and "final status" in result
    assert "ERROR: build failed" in result
    assert ".kia/tool-results/test.txt" in result
    assert "grep_files/read_file" in result


def test_tool_result_compaction_leaves_small_results_unchanged():
    result, compacted = _compact_text("small result", "read_file", 16_000)
    assert result == "small result"
    assert compacted is False


def test_tool_result_compaction_collapses_repeated_logs():
    text = "same noisy line\n" * 2000
    result, compacted = _compact_text(text, "exec_command", 16_000)
    assert compacted
    assert "previous line repeated 1999 more times" in result


def test_tool_result_compaction_keeps_long_middle_diagnostic():
    text = "head\n" + "a" * 5000 + "\nERROR: " + "x" * 800 + "\n" + "b" * 5000 + "\ntail\n"
    result, compacted = _compact_text(text, "exec_command", 16_000)
    assert compacted
    assert "ERROR:" in result
    assert len(result) <= tool_result_char_budget(16_000, tool_name="exec_command")


def test_tool_result_compaction_samples_late_diagnostics():
    diagnostics = "\n".join(f"ERROR early {i} " + "x" * 500 for i in range(8))
    text = "head\n" + diagnostics + "\nFATAL late diagnostic\n" + "tail\n" * 1000
    result, compacted = _compact_text(text, "exec_command", 16_000)
    assert compacted
    assert "ERROR early 0" in result
    assert "FATAL late diagnostic" in result


def test_tool_result_compaction_hard_caps_long_artifact_path():
    text = "x" * 20_000
    result, compacted = _compact_text(
        text, "exec_command", 16_000, artifact_path="/" + "p" * 10_000
    )
    assert compacted
    assert len(result) <= tool_result_char_budget(16_000, tool_name="exec_command")


def test_read_file_compaction_keeps_contiguous_prefix():
    text = "".join(f"line {i}\n" for i in range(3000))
    text += "[output truncated: 1000 of 3000 lines shown. Use offset/limit for more.]"
    result, compacted = _compact_text(text, "read_file", 16_000)
    assert compacted
    assert "line 0\nline 1\n" in result
    assert "middle omitted" not in result
    assert "remainder omitted" in result
    assert "1000 of 3000 lines shown" in result


def test_pytest_semantic_reducer_preserves_failures_and_summary():
    text = (
        "platform linux -- Python 3.11\n" + "collected test noise\n" * 500
        + "________________ test_example ________________\n"
        + "E   AssertionError: expected 1, got 2\n"
        + "FAILED tests/test_example.py::test_example - AssertionError\n"
        + "================ 1 failed, 20 passed in 1.00s ================\n"
    )
    result = compact_tool_result_envelope(
        ToolResultEnvelope(
            "exec_command", {"command": "pytest -q"}, {"success": False}, text
        ),
        16_000,
    )
    assert result.compacted and result.reducer == "pytest" and result.tier == "semantic"
    assert "test_example" in result.text and "1 failed, 20 passed" in result.text
    assert "reducer=pytest" in result.text


def test_pytest_semantic_reducer_keeps_plain_e_diagnostic():
    text = "noise\n" * 1000 + "E   expected 1, got 2\n" + "1 failed in 0.1s\n"
    result = compact_tool_result_envelope(
        ToolResultEnvelope(
            "exec_command", {"command": "pytest -q"}, {"success": False}, text
        ),
        16_000,
    )
    assert "E   expected 1, got 2" in result.text


def test_git_diff_semantic_reducer_keeps_changed_lines():
    text = (
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n"
        "-old\n+new\n" + " unchanged context\n" * 1000
    )
    result = compact_tool_result_envelope(
        ToolResultEnvelope(
            "exec_command", {"command": "git diff"}, {"success": True}, text
        ),
        16_000,
    )
    assert result.reducer == "git-diff" and result.tier == "semantic"
    assert "-old" in result.text and "+new" in result.text


def test_git_diff_semantic_reducer_preserves_duplicate_changed_lines():
    text = (
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n+same\n"
        "diff --git a/b.py b/b.py\n--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n+same\n"
        + " context\n" * 1000
    )
    result = compact_tool_result_envelope(
        ToolResultEnvelope(
            "exec_command", {"command": "git diff"}, {"success": True}, text
        ),
        16_000,
    )
    assert result.text.count("+same") == 2


def test_file_content_is_not_terminal_normalized():
    text = "value  \n-----\n\n\nnext\n" * 1000
    result = compact_tool_result_envelope(
        ToolResultEnvelope("read_file", {}, {"success": True}, text),
        16_000,
    )
    assert "value  \n-----\n\n\nnext" in result.text


def test_semantic_compaction_always_keeps_stderr_excerpt():
    text = (
        "[stderr] plain but important diagnostic\n"
        + "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n"
        + "+changed\n"
        + " context\n" * 1000
    )
    result = compact_tool_result_envelope(
        ToolResultEnvelope(
            "exec_command", {"command": "git diff"}, {"success": False}, text
        ),
        16_000,
    )
    assert result.reducer == "git-diff"
    assert "[stderr excerpt]" in result.text
    assert "plain but important diagnostic" in result.text


def test_generic_cleanup_strips_ansi_and_progress():
    text = "\x1b[31mERROR colored\x1b[0m\n[==========] 50%\n" + "noise\n" * 1000
    result = compact_tool_result_envelope(
        ToolResultEnvelope(
            "exec_command", {"command": "custom-build"}, {"success": False}, text
        ),
        16_000,
    )
    assert "\x1b" not in result.text and "[==========] 50%" not in result.text
    assert "ERROR colored" in result.text


def test_truncated_recovery_artifact_is_disclosed():
    text = "noise\n" * 1000
    result = compact_tool_result_envelope(
        ToolResultEnvelope(
            "exec_command",
            {"command": "custom"},
            {"artifact_truncated": True, "original_output_chars": 200_000_000},
            text,
        ),
        16_000,
        artifact_path=".kia/tool-results/captured.txt",
    )
    assert "100 MiB capture limit" in result.text
    assert result.warnings


def test_unsaved_artifact_guidance_does_not_claim_saved_output():
    text = "ordinary output\n" * 1000
    result, compacted = _compact_text(text, "exec_command", 16_000)
    assert compacted
    assert "could not be saved" in result
    assert "Search the saved output" not in result


def test_small_exec_result_discards_temporary_artifact(tmp_path):
    producer = tmp_path / "producer.txt"
    producer.write_text("small")
    console = _Console()
    console.system = lambda *args, **kwargs: None
    tool_call = NS(function=NS(name="exec_command", arguments='{"command": "echo ok"}'), id="call-small")
    result = {
        "stdout": "ok\n",
        "exit_code": 0,
        "success": True,
        "streamed": True,
        "_artifact_path": str(producer),
        "original_output_chars": 3,
    }
    agent = NS(
        verbose=False,
        console=console,
        permissions=NS(check=lambda *args: (True, "")),
        tool_executor=NS(execute=lambda *args: result),
        context_length=16_000,
        token_estimator=NS(chars_per_token=3.3),
        context=NS(messages=[], add=lambda message: agent.context.messages.append(message)),
        cancellation=None,
    )
    LLMAgent.execute_tool_calls(agent, [tool_call])
    assert not producer.exists()


def test_large_failed_exec_uses_full_capture_for_compaction(tmp_path):
    producer = tmp_path / "producer.txt"
    captured = "HEAD\n" + "noise\n" * 1000 + "ERROR full diagnostic\nTAIL\n"
    producer.write_text(captured)
    console = _Console()
    console.system = lambda *args, **kwargs: None
    result = {
        "stderr": "brief failure",
        "exit_code": 1,
        "success": False,
        "streamed": True,
        "_artifact_path": str(producer),
        "original_output_chars": len(captured),
    }
    tool_call = NS(
        function=NS(name="exec_command", arguments='{"command": "custom-build"}'),
        id="call-failed",
    )
    agent = NS(
        verbose=False,
        console=console,
        permissions=NS(check=lambda *args: (True, "")),
        tool_executor=NS(execute=lambda *args: result),
        context_length=16_000,
        token_estimator=NS(chars_per_token=3.3),
        context=NS(messages=[], add=lambda message: agent.context.messages.append(message)),
        cancellation=None,
        work_dir=str(tmp_path),
        _session_id="test",
        round_id=2,
        tool_compaction_totals={"calls": 0, "original_chars": 0, "retained_chars": 0},
    )

    LLMAgent.execute_tool_calls(agent, [tool_call])

    stored = agent.context.messages[-1]["content"]
    assert "ERROR full diagnostic" in stored
    artifact = tmp_path / ".kia" / "tool-results" / "test" / "r2-call-failed-exec_command.txt"
    assert artifact.read_text() == captured
    assert not producer.exists()


def test_actual_failed_exec_capture_is_compacted_and_persisted(tmp_path):
    console = _Console()
    console.system = lambda *args, **kwargs: None
    executor = ToolExecutor(console=console, work_dir=str(tmp_path))
    command = (
        "python -c \"import sys; print('HEAD'); print('x' * 5000); "
        "print('brief stderr', file=sys.stderr); sys.exit(1)\""
    )
    tool_call = NS(
        function=NS(name="exec_command", arguments=json.dumps({"command": command})),
        id="call-real-failed",
    )
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
    tool_call = NS(function=NS(name="exec_command", arguments='{"command": "noisy"}'), id="call-1")
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
    assert artifact.stat().st_mode & 0o077 == 0


def test_artifact_unlink_failure_does_not_skip_tool_response(tmp_path):
    producer = tmp_path / "producer.txt"
    producer.write_text("small")
    console = _Console()
    console.system = lambda *args, **kwargs: None
    warnings = []
    console.warn = lambda message: warnings.append(message)
    tool_call = NS(
        function=NS(name="exec_command", arguments='{"command": "echo ok"}'),
        id="call-unlink",
    )
    result = {
        "stdout": "ok\n",
        "success": True,
        "streamed": True,
        "exit_code": 0,
        "_artifact_path": str(producer),
        "original_output_chars": 3,
    }
    agent = NS(
        verbose=False,
        console=console,
        permissions=NS(check=lambda *args: (True, "")),
        tool_executor=NS(execute=lambda *args: result),
        context_length=16_000,
        token_estimator=NS(chars_per_token=3.3),
        context=NS(messages=[], add=lambda message: agent.context.messages.append(message)),
        cancellation=None,
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        backend, "discard_tool_result_artifact", lambda result: OSError("unlink denied")
    )

    try:
        LLMAgent.execute_tool_calls(agent, [tool_call])
    finally:
        monkeypatch.undo()

    assert agent.context.messages[-1]["tool_call_id"] == "call-unlink"
    assert "unlink denied" in warnings[-1]


def test_tool_artifact_write_failure_leaves_no_partial_file(tmp_path, monkeypatch):
    original_fdopen = tool_results.os.fdopen

    class FailingFile:
        def __init__(self, fd, *args, **kwargs):
            self.file = original_fdopen(fd, *args, **kwargs)

        def __enter__(self):
            return self

        def write(self, text):
            raise OSError("disk full")

        def __exit__(self, *args):
            self.file.close()

    monkeypatch.setattr(tool_results.os, "fdopen", FailingFile)
    with pytest.raises(OSError, match="disk full"):
        tool_results.persist_tool_result_artifact(
            "exec_command", "captured", {}, "call", str(tmp_path), "test", 1
        )
    artifact_dir = tmp_path / ".kia" / "tool-results" / "test"
    assert list(artifact_dir.iterdir()) == []


def test_tool_artifact_move_failure_cleans_source_and_destination(tmp_path, monkeypatch):
    producer = tmp_path / "producer.txt"
    producer.write_text("captured")
    destination = (
        tmp_path / ".kia" / "tool-results" / "test" / "r1-call-exec_command.txt"
    )
    monkeypatch.setattr(
        tool_results.os,
        "replace",
        lambda *args: (_ for _ in ()).throw(OSError("move failed")),
    )
    with pytest.raises(OSError, match="move failed"):
        tool_results.persist_tool_result_artifact(
            "exec_command",
            "captured",
            {"_artifact_path": str(producer)},
            "call",
            str(tmp_path),
            "test",
            1,
        )
    assert not producer.exists() and not destination.exists()


def test_tool_artifact_names_do_not_overwrite_existing_files(tmp_path):
    first = tool_results.persist_tool_result_artifact(
        "exec_command", "first", {}, "call-1", str(tmp_path), "test", 2
    )
    second = tool_results.persist_tool_result_artifact(
        "exec_command", "second", {}, "call-1", str(tmp_path), "test", 2
    )
    assert first != second
    assert (tmp_path / first).read_text() == "first"
    assert (tmp_path / second).read_text() == "second"


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

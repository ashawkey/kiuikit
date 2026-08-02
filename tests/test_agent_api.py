"""Tests for the public one-shot Python API."""

from contextlib import nullcontext

import pytest

from kiui.config import conf
from kiui.agent import AgentRunResult, TurnOutcome, run_agent
from kiui.agent import api


class _Console:
    def __init__(self):
        self.suppressed_calls = 0

    def suppressed(self):
        self.suppressed_calls += 1
        return nullcontext()


@pytest.fixture
def model_config(monkeypatch):
    monkeypatch.setitem(conf, "openai", {
        "test": {
            "model": "test-model",
            "api_key": "key",
            "base_url": "url",
            "provider": "openai",
            "reasoning_effort": "medium",
            "context_length": 200_000,
            "max_output_tokens": 16_000,
        }
    })


def test_run_agent_constructs_exec_agent_and_closes(monkeypatch, model_config, tmp_path):
    created = []

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._last_turn_outcome = TurnOutcome.COMPLETED
            self._last_error = None
            self.token_totals = {"total": 12, "prompt": 8, "completion": 4}
            self.closed = False
            created.append(self)

        def execute(self, task):
            assert task == "inspect this"
            return "done"

        def close(self):
            self.closed = True

    console = _Console()
    monkeypatch.setattr(api, "LLMAgent", FakeAgent)

    result = run_agent(
        "inspect this",
        model_alias="test",
        persona="coder",
        work_dir=tmp_path,
        console=console,
    )

    assert isinstance(result, AgentRunResult)
    assert result.success
    assert result.response == "done"
    assert result.outcome == TurnOutcome.COMPLETED
    assert result.token_usage == {"total": 12, "prompt": 8, "completion": 4}
    assert result.error is None
    assert console.suppressed_calls == 1
    assert created[0].closed
    assert created[0].kwargs["exec_mode"] is True
    assert created[0].kwargs["model_alias"] == "test"
    assert created[0].kwargs["persona"] == "coder"
    assert created[0].kwargs["work_dir"] == str(tmp_path)
    assert created[0].kwargs["stream"] is False
    assert created[0].kwargs["reasoning_effort"] == "medium"
    assert created[0].kwargs["context_length"] == 200_000
    assert created[0].kwargs["max_output_tokens"] == 16_000


def test_run_agent_returns_failed_outcome(monkeypatch, model_config):
    class FakeAgent:
        def __init__(self, **kwargs):
            self._last_turn_outcome = TurnOutcome.FAILED
            self._last_error = "request failed"
            self.token_totals = {"total": 0}
            self.closed = False

        def execute(self, task):
            return None

        def close(self):
            self.closed = True

    monkeypatch.setattr(api, "LLMAgent", FakeAgent)

    result = run_agent("try once", model_alias="test", quiet=False)

    assert not result.success
    assert result.outcome == TurnOutcome.FAILED
    assert result.error == "request failed"


def test_run_agent_closes_when_execution_raises(monkeypatch, model_config):
    created = []

    class FakeAgent:
        def __init__(self, **kwargs):
            self.closed = False
            created.append(self)

        def execute(self, task):
            raise LookupError("boom")

        def close(self):
            self.closed = True

    monkeypatch.setattr(api, "LLMAgent", FakeAgent)

    with pytest.raises(LookupError, match="boom"):
        run_agent("fail", model_alias="test", quiet=False)

    assert created[0].closed


def test_run_agent_validates_task_and_model(model_config):
    with pytest.raises(ValueError, match="non-empty"):
        run_agent("  ", model_alias="test")
    with pytest.raises(ValueError, match="Model 'missing' not found"):
        run_agent("task", model_alias="missing")

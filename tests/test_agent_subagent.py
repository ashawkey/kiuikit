"""Tests for synchronous sub-agent execution."""

from kiui.config import conf
from kiui.agent.subagent import SubagentManager


class _Console:
    def system(self, *args, **kwargs):
        pass


class _ToolExecutor:
    def __init__(self):
        self.stopped = False

    def shutdown_processes(self):
        self.stopped = True


def test_subagent_returns_full_response(monkeypatch, tmp_path):
    response = "x" * 3000 + "THE END"
    created = []

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._last_interrupted = False
            self.tool_executor = _ToolExecutor()
            created.append(self)

        def execute(self, task, manage_operation=False):
            assert task == "inspect this"
            assert manage_operation is False
            return response

    monkeypatch.setitem(conf, "openai", {
        "test": {"model": "model", "api_key": "key", "base_url": "url"}
    })
    monkeypatch.setattr("kiui.agent.backend.LLMAgent", FakeAgent)

    manager = SubagentManager(model_alias="test", console=_Console())
    result = manager.spawn("inspect this", cwd=str(tmp_path))

    assert result["success"]
    assert result["message"] == f"Sub-agent completed.\n{response}"
    assert result["message"].endswith("THE END")
    assert created[0].kwargs["is_subagent"] is True
    assert created[0].kwargs["work_dir"] == str(tmp_path)
    assert created[0].tool_executor.stopped

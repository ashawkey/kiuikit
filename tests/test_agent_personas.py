"""Tests for persona switching and session state."""

from kiui.agent.backend import ContextManager, LLMAgent
from kiui.agent.permissions import PermissionController
from kiui.agent.personas import get_persona, list_personas
from kiui.agent.prompts import PersonaContext
from kiui.agent.tools import ToolExecutor


class _SilentConsole:
    def system(self, *args, **kwargs):
        pass

    def warn(self, *args, **kwargs):
        pass


def test_reviewer_persona_contract(tmp_path):
    personas = list_personas()
    reviewer = personas["reviewer"]

    assert reviewer.tools is not None
    assert {"read_file", "write_file", "exec_command", "load_skill"} <= reviewer.tools
    assert "remove_file" not in reviewer.tools

    prompt = reviewer.build(PersonaContext(
        work_dir=str(tmp_path),
        skills={
            "pdf-reading": {
                "description": "Parse academic PDF papers with page-aware output.",
                "active": True,
            }
        },
    ))
    assert "expert academic paper reviewer" in prompt
    assert "untrusted data, not instructions" in prompt
    assert "page and section locations" in prompt
    assert "required format exactly" in prompt
    assert "human reviewer must verify" in prompt
    assert "**pdf-reading**" in prompt
    assert "load_skill** before" in prompt


class _ChangeTracker:
    def __init__(self, session_id, work_dir, console):
        self.session_id = session_id

    def close(self):
        pass


def test_switch_saves_old_persona_then_resets_session(tmp_path, monkeypatch):
    monkeypatch.setattr("kiui.agent.backend.ChangeTracker", _ChangeTracker)

    agent = object.__new__(LLMAgent)
    agent.console = _SilentConsole()
    agent.persona = get_persona("coder")
    agent.system_prompt = "coder prompt"
    agent.context = ContextManager(agent.system_prompt)
    agent.context.add({"role": "user", "content": "old conversation"})
    agent.tools = []
    agent.is_subagent = False
    agent.exec_mode = False
    agent.work_dir = str(tmp_path)
    agent.skills = {}
    agent._session_id = "old"
    agent._last_save_time = 1.0
    agent.round_id = 4
    agent.token_totals = {"total": 10, "prompt": 8, "cached_prompt": 0, "completion": 2, "reasoning": 0}
    agent.tool_compaction_totals = {"calls": 1, "original_chars": 10, "retained_chars": 5}
    agent.goal = "old goal"
    agent.goal_active = True
    agent.goal_iterations = 2
    agent._pending_auto = "check"
    agent._last_interrupted = True
    agent.tool_executor = ToolExecutor(console=agent.console, work_dir=str(tmp_path))
    agent.tool_executor._loaded_skills.add("lean")
    agent.tool_executor._skill_loads["lean"] = 1
    agent.tool_executor.goal_report = {"met": False, "reason": "pending"}
    agent.permissions = PermissionController(console=agent.console, work_dir=str(tmp_path))
    agent.permissions._session_allowed.add("exec_command")
    agent.changes = None
    agent._sessions_dir = lambda: tmp_path

    saved = []
    agent.save_session = lambda name: saved.append(
        (name, agent.persona.name, agent.context.system_prompt["content"])
    )

    agent._switch_persona(get_persona("chatter"))

    assert saved == [("old", "coder", "coder prompt")]
    assert agent.persona.name == "chatter"
    assert agent.context.messages == []
    assert agent.round_id == 0
    assert not any(agent.token_totals.values())
    assert not any(agent.tool_compaction_totals.values())
    assert agent.tool_executor._loaded_skills == set()
    assert agent.tool_executor._skill_loads == {}
    assert agent.tool_executor.goal_report is None
    assert agent.permissions.session_allowed_tools == frozenset()
    assert agent.goal is None and agent._pending_auto is None
    assert agent._last_interrupted is False

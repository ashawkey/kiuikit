"""Tests for the /goal auto-iteration feature (kiui.agent)."""

from kiui.agent.backend import LLMAgent
from kiui.agent.personas import get_persona
from kiui.agent.tools import ToolExecutor, get_tool_definitions
from kiui.agent.ui import AgentConsole


def make_agent() -> LLMAgent:
    """Build a bare LLMAgent with just the state the goal helpers touch."""
    agent = object.__new__(LLMAgent)
    agent.console = AgentConsole()
    agent.goal = None
    agent.goal_active = False
    agent.goal_iterations = 0
    agent._pending_auto = None
    agent._last_interrupted = False
    agent.persona = get_persona("coder")
    agent.tool_executor = ToolExecutor(console=agent.console)
    return agent


# ----- slash command -------------------------------------------------------

def test_goal_set_arms_auto_loop():
    agent = make_agent()
    agent._cmd_goal("/goal make the tests pass")

    assert agent.goal == "make the tests pass"
    assert agent.goal_active is True
    assert agent.goal_iterations == 0
    assert agent._pending_auto is not None
    assert "make the tests pass" in agent._pending_auto
    assert "report_goal" in agent._pending_auto


def test_goal_status_without_goal_is_noop():
    agent = make_agent()
    agent._cmd_goal("/goal")  # should not raise
    assert agent.goal is None
    assert agent._pending_auto is None


def test_goal_disabled_without_report_goal_access():
    agent = make_agent()
    agent.persona = get_persona("chatter")

    agent._cmd_goal("/goal keep going")

    assert agent.goal is None
    assert agent.goal_active is False
    assert agent._pending_auto is None


def test_goal_off_clears_everything():
    agent = make_agent()
    agent._cmd_goal("/goal fix the bug")
    agent._cmd_goal("/goal off")

    assert agent.goal is None
    assert agent.goal_active is False
    assert agent.goal_iterations == 0
    assert agent._pending_auto is None


# ----- report_goal tool ----------------------------------------------------

def test_report_goal_tool_is_registered():
    names = {t["function"]["name"] for t in get_tool_definitions()}
    assert "report_goal" in names


def test_report_goal_records_result():
    executor = ToolExecutor(console=AgentConsole())
    result = executor.execute("report_goal", {"met": True, "reason": "all green"})

    assert result["success"] is True
    assert executor.goal_report == {"met": True, "reason": "all green"}


def test_report_goal_defaults_to_not_met():
    executor = ToolExecutor(console=AgentConsole())
    executor.execute("report_goal", {})
    assert executor.goal_report == {"met": False, "reason": ""}


# ----- iteration control ---------------------------------------------------

def test_maybe_continue_met_stops_loop():
    agent = make_agent()
    agent._cmd_goal("/goal x")
    agent._pending_auto = None  # consumed by the round
    agent.tool_executor.goal_report = {"met": True, "reason": "done"}

    agent._maybe_continue_goal()
    assert agent.goal_active is False
    assert agent._pending_auto is None


def test_maybe_continue_unmet_iterates():
    agent = make_agent()
    agent._cmd_goal("/goal x")
    agent._pending_auto = None
    agent.tool_executor.goal_report = {"met": False, "reason": "still failing"}

    agent._maybe_continue_goal()
    assert agent.goal_active is True
    assert agent.goal_iterations == 1
    assert agent._pending_auto is not None


def test_maybe_continue_missing_report_still_iterates():
    agent = make_agent()
    agent._cmd_goal("/goal x")
    agent._pending_auto = None
    agent.tool_executor.goal_report = None  # model forgot to call report_goal

    agent._maybe_continue_goal()
    assert agent.goal_active is True
    assert agent._pending_auto is not None


def test_maybe_continue_interrupt_clears_goal():
    agent = make_agent()
    agent._cmd_goal("/goal x")
    agent._pending_auto = None
    agent._last_interrupted = True
    agent.tool_executor.goal_report = None

    agent._maybe_continue_goal()
    assert agent.goal is None
    assert agent.goal_active is False
    assert agent.goal_iterations == 0
    assert agent._pending_auto is None


def test_maybe_continue_noop_when_no_goal():
    agent = make_agent()
    agent.tool_executor.goal_report = {"met": False, "reason": "x"}
    agent._maybe_continue_goal()
    assert agent._pending_auto is None
    assert agent.goal_iterations == 0

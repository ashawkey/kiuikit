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

# ----- report_goal tool ----------------------------------------------------

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

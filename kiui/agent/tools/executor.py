"""Tool dispatcher assembled from focused tool mixins, backed by the registry.

The :class:`~kiui.agent.tools.registry.ToolRegistry` is the single source of
truth for every tool (built-in and skill-provided): its schema, dispatch
handler, and advertising gates. The executor holds one
registry and dispatches through it, so there is no separate built-in/skill
branch to keep in sync.
"""

from pathlib import Path
from typing import Any

from kiui.agent.utils.io import CancellationToken
from kiui.agent.ui import AgentConsole

from .commands import CommandToolsMixin
from .control import ControlToolsMixin
from .files import FileToolsMixin
from .formatting import log_tool_call
from .process_manager import ProcessManagerMixin
from .registry import ToolRegistry
from .search import SearchToolsMixin
from .session import SessionToolsMixin
from .web import WebToolsMixin


class ToolExecutor(
    FileToolsMixin,
    CommandToolsMixin,
    ControlToolsMixin,
    ProcessManagerMixin,
    SearchToolsMixin,
    WebToolsMixin,
    SessionToolsMixin,
):
    """Execute tool calls (built-in or skill-provided) and return results."""

    def __init__(
        self,
        console: AgentConsole | None = None,
        work_dir: str | None = None,
        change_tracker=None,
        get_round_id=None,
        skills: dict | None = None,
        cancellation: CancellationToken | None = None,
        isolated_turn=None,
    ):
        self.console = console or AgentConsole()
        self.cancellation = cancellation
        # LLMAgent.run_isolated_turn: run one turn and discard its context.
        # Skill tools that drive repetitive work (the `batch` skill) call it;
        # None when no agent owns this executor.
        self.isolated_turn = isolated_turn
        self._work_dir = str(Path(work_dir).absolute()) if work_dir else str(Path.cwd())
        self._change_tracker = change_tracker
        self._get_round_id = get_round_id  # callable → int
        self._skills = skills or {}
        self._loaded_skills: set[str] = set()
        # Per-session usage counter: skill name → number of load_skill invocations,
        # including reloads. Persisted with the session
        # for telemetry and surfaced in /usage and the final summary.
        self._skill_loads: dict[str, int] = {}
        # Single source of truth for all tools (built-ins seeded; skill tools
        # added/removed as skills load and unload).
        self.registry = ToolRegistry()
        self._tool_resource_cleanups: dict[str, Any] = {}
        self._init_process_registry()

    def execute(self, function_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch and execute a tool call. Returns dict with success key."""
        if self.cancellation is not None and self.cancellation.cancelled:
            return {
                "error": "Tool call skipped: the user interrupted the turn.",
                "success": False,
                "interrupted": True,
            }
        spec = self.registry.get(function_name)
        if spec is None:
            return {"error": f"Unknown tool: {function_name}", "success": False}
        log_tool_call(self.console, function_name, arguments, spec.describe)
        try:
            return spec.handler(self, **arguments)
        except Exception as e:
            return {"error": f"Tool execution failed: {e}", "success": False}

    # -- skill-provided tools ----------------------------------------------

    def register_tool_resource(self, name: str, cleanup) -> None:
        """Register cleanup for session-scoped state owned by a native skill."""
        previous = self._tool_resource_cleanups.get(name)
        if previous is not None and previous is not cleanup:
            previous()
        self._tool_resource_cleanups[name] = cleanup

    def close_tool_resource(self, name: str) -> None:
        """Close and forget one native skill's session-scoped resource."""
        cleanup = self._tool_resource_cleanups.pop(name, None)
        if cleanup is not None:
            cleanup()

    def shutdown_tool_resources(self, clear: bool = False) -> None:
        """Close native skill resources, optionally forgetting their callbacks."""
        for cleanup in list(self._tool_resource_cleanups.values()):
            cleanup()
        if clear:
            self._tool_resource_cleanups.clear()

    def register_skill_tools(self, name: str, entries: list[dict[str, Any]]) -> None:
        """Register the tools contributed by a loaded skill.

        Delegates to the registry, which validates atomically and rejects any
        name colliding with a built-in or another loaded skill's tool.
        """
        self.registry.register_skill(name, entries)

    def unregister_skill_tools(self, name: str) -> None:
        """Drop a skill's tools and close its session-scoped resource."""
        self.close_tool_resource(name)
        self.registry.unregister_skill(name)

    def skill_tool_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI schemas for all currently-registered skill tools."""
        return self.registry.skill_tool_schemas()

    def reset_skill_tools(self) -> None:
        """Drop skill tools and close their session-scoped resources."""
        self.shutdown_tool_resources(clear=True)
        self.registry.clear_skill_tools()

    def skill_state(self) -> dict[str, Any]:
        """Snapshot which skills are loaded and what tools they contributed.

        Skill state lives on the executor rather than in the conversation, so a
        caller that discards a turn's context (see
        ``LLMAgent.run_isolated_turn``) must restore it explicitly: otherwise a
        skill loaded inside that turn stays loaded and leaves its contributed
        tools registered in the enclosing conversation.

        Session-scoped tool *resources* are deliberately not part of the
        snapshot: they are external (a browser, a process) and their owning
        skill is responsible for their lifetime.
        """
        return {
            "loaded": set(self._loaded_skills),
            "loads": dict(self._skill_loads),
            "specs": self.registry.skill_specs(),
        }

    def restore_skill_state(self, state: dict[str, Any]) -> None:
        """Restore a snapshot taken by :meth:`skill_state`."""
        self._loaded_skills = set(state["loaded"])
        self._skill_loads = dict(state["loads"])
        self.registry.restore_skill_specs(state["specs"])

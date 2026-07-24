"""Tool dispatcher assembled from focused tool mixins, backed by the registry.

The :class:`~kiui.agent.tools.registry.ToolRegistry` is the single source of
truth for every tool (built-in and skill-provided): its schema, dispatch
handler, permission class, and advertising gates. The executor holds one
registry and dispatches through it, so there is no separate built-in/skill
branch to keep in sync.
"""

from pathlib import Path
from typing import Any

from kiui.agent.utils.io import CancellationToken
from kiui.agent.ui import AgentConsole

from .commands import CommandToolsMixin
from .files import FileToolsMixin
from .process_manager import ProcessManagerMixin
from .registry import ToolRegistry
from .search import SearchToolsMixin
from .session import SessionToolsMixin
from .web import WebToolsMixin


class ToolExecutor(
    FileToolsMixin,
    CommandToolsMixin,
    ProcessManagerMixin,
    SearchToolsMixin,
    WebToolsMixin,
    SessionToolsMixin,
):
    """Execute tool calls (built-in or skill-provided) and return results."""

    def __init__(
        self,
        console: AgentConsole | None = None,
        subagent_manager=None,
        work_dir: str | None = None,
        change_tracker=None,
        get_round_id=None,
        skills: dict | None = None,
        cancellation: CancellationToken | None = None,
    ):
        self.console = console or AgentConsole()
        self.cancellation = cancellation
        self.subagent_manager = subagent_manager
        self._work_dir = str(Path(work_dir).absolute()) if work_dir else str(Path.cwd())
        self._change_tracker = change_tracker
        self._get_round_id = get_round_id  # callable → int
        self._skills = skills or {}
        self._loaded_skills: set[str] = set()
        # Per-session usage counter: skill name → number of load_skill invocations
        # (including redundant "already loaded" calls). Persisted with the session
        # for telemetry and surfaced in /usage and the final summary.
        self._skill_loads: dict[str, int] = {}
        # Last report_goal() call result: None, or {"met": bool, "reason": str}.
        # Consumed by LLMAgent after each goal-check round.
        self.goal_report: dict | None = None
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

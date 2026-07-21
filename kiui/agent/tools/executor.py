"""Built-in tool dispatcher assembled from focused tool mixins."""

import threading
from pathlib import Path
from typing import Any

from kiui.agent.utils.io import CancellationToken
from kiui.agent.ui import AgentConsole

from .commands import CommandToolsMixin
from .files import FileToolsMixin
from .processes import ProcessToolsMixin
from .search import SearchToolsMixin
from .session import SessionToolsMixin
from .web import WebToolsMixin


class ToolExecutor(
    FileToolsMixin,
    CommandToolsMixin,
    ProcessToolsMixin,
    SearchToolsMixin,
    WebToolsMixin,
    SessionToolsMixin,
):
    """Execute built-in tool calls and return structured results."""

    _DISPATCH_MAP: dict[str, str] = {
        "read_file": "_read_file",
        "read_image": "_read_image",
        "write_file": "_write_file",
        "edit_file": "_edit_file",
        "multi_edit": "_multi_edit",
        "ls": "_ls",
        "exec_command": "_exec_command",
        "start_process": "_start_process",
        "inspect_processes": "_inspect_processes",
        "stop_process": "_stop_process",
        "glob_files": "_glob_files",
        "grep_files": "_grep_files",
        "web_search": "_web_search",
        "web_fetch": "_web_fetch",
        "remove_file": "_remove_file",
        "spawn_subagent": "_spawn_subagent",
        "load_skill": "_load_skill",
        "report_goal": "_report_goal",
    }

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
        self._processes: dict[str, dict[str, Any]] = {}
        self._process_lock = threading.Lock()

    def execute(self, function_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch and execute a tool call. Returns dict with success key."""
        method_name = self._DISPATCH_MAP.get(function_name)
        if not method_name:
            return {"error": f"Unknown tool: {function_name}", "success": False}
        if self.cancellation is not None and self.cancellation.cancelled:
            return {
                "error": "Tool call skipped: the user interrupted the turn.",
                "success": False,
                "interrupted": True,
            }
        try:
            return getattr(self, method_name)(**arguments)
        except Exception as e:
            return {"error": f"Tool execution failed: {e}", "success": False}

"""Agent-specific sub-agent, skill, and goal tools."""

from pathlib import Path
from typing import Any


class SessionToolsMixin:
    def _spawn_subagent(self, task: str = "") -> dict[str, Any]:
        """Spawn a sub-agent and wait for it to complete."""
        self.console.tool(f"spawn_subagent: {task[:60]}")

        if self.subagent_manager is None:
            return {"error": "Sub-agent spawning is not available.", "success": False}
        if not task:
            return {"error": "task is required.", "success": False}

        return self.subagent_manager.spawn(task=task, cwd=self._work_dir)

    def _load_skill(self, name: str) -> dict[str, Any]:
        """Load a skill's full prompt instructions into the conversation context."""
        self.console.tool(f"load_skill {name}")

        if not self._skills:
            return {
                "error": "No skills available. Create a folder under .kia/skills/<name>/ with a SKILL.md file.",
                "success": False,
            }

        if name not in self._skills:
            available = ", ".join(sorted(self._skills.keys()))
            return {
                "error": f"Skill '{name}' not found. Available: {available}",
                "success": False,
            }

        if name in self._loaded_skills:
            self._skill_loads[name] = self._skill_loads.get(name, 0) + 1
            return {"message": f"Skill '{name}' is already loaded.", "success": True}

        skill = self._skills[name]
        body = skill["body"]
        skill_dir = skill.get("dir")
        # Register contributed tools before marking the skill loaded so a
        # packaging error (e.g. a tool shadowing a built-in) fails the whole
        # load atomically instead of leaving half-registered state.
        error = self._register_skill_tools(name, skill_dir)
        if error is not None:
            return {"error": error, "success": False}

        self._loaded_skills.add(name)
        self._skill_loads[name] = self._skill_loads.get(name, 0) + 1
        resources = [
            directory
            for directory in ("references", "scripts", "assets")
            if skill_dir and (Path(skill_dir) / directory).is_dir()
        ]
        if resources:
            resource_list = ", ".join(f"{directory}/…" for directory in resources)
            header = (
                f"[Skill '{name}' loaded. Its directory is {skill_dir} — resolve relative "
                f"files in {resource_list} against that path using read_file / exec_command "
                f"as the instructions require.]\n\n"
            )
        else:
            header = f"[Skill '{name}' loaded.]\n\n"
        body = header + body
        return {"content": body, "success": True}

    def _register_skill_tools(self, name: str, skill_dir: str | None) -> str | None:
        """Import a loaded skill's tools.py (if any) and register its tools.

        Returns ``None`` on success (including when the skill ships no tools),
        or an error message string when the tools.py is broken (import error or
        a tool that shadows a built-in). A broken tools.py fails the whole skill
        load rather than silently loading a skill whose advertised tools are
        missing; registration is atomic so no partial state is left behind.
        """
        if not skill_dir:
            return None
        from kiui.agent.skills import load_skill_tools

        try:
            entries = load_skill_tools(skill_dir)
            if entries:
                self.register_skill_tools(name, entries)
        except Exception as e:
            return f"Skill '{name}' tools.py failed to load: {e}"
        return None

    def _report_goal(self, met: bool = False, reason: str = "") -> dict[str, Any]:
        """Record whether the current standing goal is met.

        The result is stashed on ``self.goal_report`` for the agent loop to
        read after the round; it decides whether to keep auto-iterating.
        """
        met = bool(met)
        reason = reason or ""
        self.console.tool(f"report_goal(met={met})")
        self.goal_report = {"met": met, "reason": reason}
        status = "goal met" if met else "goal not yet met"
        return {
            "message": f"Recorded: {status}." + (f" {reason}" if reason else ""),
            "success": True,
        }

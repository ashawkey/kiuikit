"""Agent-specific skill tools."""

from pathlib import Path
from typing import Any

class SessionToolsMixin:
    def _load_skill(self, name: str) -> dict[str, Any]:
        """Load a skill's full prompt instructions into the conversation context."""
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

        skill = self._skills[name]
        body = skill["body"]
        skill_dir = skill.get("dir")
        if name not in self._loaded_skills:
            # Register contributed tools before marking the skill loaded so a
            # packaging error (e.g. a tool shadowing a built-in) fails the whole
            # load atomically instead of leaving half-registered state.
            error = self._register_skill_tools(name, skill_dir)
            if error is not None:
                return {"error": error, "success": False}
            self._loaded_skills.add(name)

        # Always return the instructions. They may have been compacted out of
        # the conversation since the previous load, and callers cannot reliably
        # tell whether the body is still present.
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

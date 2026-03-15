"""Sub-agent spawning for kiui agent — foreground in-process only."""

import os
from typing import Any

from kiui.agent.ui import AgentConsole


class SubagentManager:
    """Spawn in-process sub-agents and wait for completion (synchronous)."""

    def __init__(
        self,
        model_key: str,
        max_depth: int = 3,
        console: AgentConsole | None = None,
    ):
        self.model_key = model_key
        self.max_depth = max_depth
        self.console = console or AgentConsole()
        self._depth = int(os.environ.get("KIA_SPAWN_DEPTH", "0"))

    def spawn(
        self,
        task: str,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Spawn an in-process sub-agent and wait for it to complete.

        Creates a new LLMAgent in the same process and calls execute().
        Returns the result directly from memory.
        """
        if self._depth >= self.max_depth:
            return {"error": f"Max spawn depth reached ({self.max_depth}).", "success": False}

        # avoid circular import at beginning of file
        from kiui.config import conf
        from kiui.agent.backend import LLMAgent
        from kiui.agent.permissions import PermissionMode

        openai_conf = conf.get("openai", {})
        if self.model_key not in openai_conf:
            return {"error": f"Model '{self.model_key}' not found in config.", "success": False}

        model_conf = openai_conf[self.model_key]
        work_dir = cwd or os.getcwd()
        old_cwd = os.getcwd()

        label = task[:60]
        self.console.system(f"── sub-agent '{label}' start ──")

        old_depth = os.environ.get("KIA_SPAWN_DEPTH")
        os.environ["KIA_SPAWN_DEPTH"] = str(self._depth + 1)

        try:
            os.chdir(work_dir)

            agent = LLMAgent(
                model=model_conf.get("model", self.model_key),
                api_key=model_conf.get("api_key", ""),
                base_url=model_conf.get("base_url", ""),
                model_key=self.model_key,
                verbose=False,
                permission_mode=PermissionMode.AUTO,
                exec_mode=True,
            )
            response = agent.execute(task)

            self.console.system(f"── sub-agent '{label}' finished ──")

            summary = response[:2000] if response else ""
            return {
                "message": f"Sub-agent completed.\n{summary}" if summary else "Sub-agent completed with no response.",
                "success": True,
            }

        except Exception as e:
            self.console.system(f"── sub-agent '{label}' error: {e} ──")
            return {"error": f"Sub-agent error: {e}", "success": False}

        finally:
            os.chdir(old_cwd)
            if old_depth is None:
                os.environ.pop("KIA_SPAWN_DEPTH", None)
            else:
                os.environ["KIA_SPAWN_DEPTH"] = old_depth

"""Sub-agent spawning for kiui agent — foreground one-shot only."""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from kiui.agent.ui import AgentConsole
from kiui.agent.utils import get_kia_dir


class SubagentManager:
    """Spawn isolated sub-agent processes and wait for completion (synchronous)."""

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
        timeout_seconds: int = 0,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Spawn a sub-agent and wait for it to complete.

        Runs ``kia exec --model <key> "task"`` as a subprocess.
        Blocks until the subprocess finishes and returns the result directly.
        """
        if self._depth >= self.max_depth:
            return {"error": f"Max spawn depth reached ({self.max_depth}).", "success": False}

        run_id = str(uuid4())[:8]
        work_dir = cwd or os.getcwd()
        result_file = get_kia_dir(work_dir) / "subagent-runs" / f"{run_id}.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)

        label = task[:60]
        self.console.system(f"── sub-agent '{label}' ({run_id}) start ──")

        process = subprocess.Popen(
            [
                sys.executable, "-m", "kiui.agent.cli",
                "exec", "--model", self.model_key,
                "--result-file", str(result_file),
                "--prompt", task,
            ],
            cwd=work_dir,
            env={**os.environ, "KIA_SPAWN_DEPTH": str(self._depth + 1)},
            stderr=subprocess.PIPE,
        )

        try:
            effective_timeout = timeout_seconds if timeout_seconds > 0 else None
            try:
                _, stderr = process.communicate(timeout=effective_timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                self.console.system(f"── sub-agent '{label}' ({run_id}) timed out ──")
                return {"error": f"Timed out after {timeout_seconds}s", "run_id": run_id, "success": False}

            exit_code = process.returncode
            stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""

            result = None
            if result_file.exists():
                try:
                    result = json.loads(result_file.read_text())
                except (json.JSONDecodeError, OSError):
                    pass

            summary = ""
            if result:
                summary = result.get("summary", result.get("response", ""))

            self.console.system(f"── sub-agent '{label}' ({run_id}) finished (exit code {exit_code}) ──")

            if exit_code != 0:
                error_detail = stderr_text[:1000] if stderr_text else "(no stderr)"
                return {
                    "error": f"Sub-agent failed (exit code {exit_code}):\n{error_detail}",
                    "run_id": run_id,
                    "exit_code": exit_code,
                    "success": False,
                }

            return {
                "message": f"Sub-agent completed.\n{summary[:2000]}",
                "run_id": run_id,
                "exit_code": 0,
                "success": True,
            }

        except Exception as e:
            self.console.system(f"── sub-agent '{label}' ({run_id}) error: {e} ──")
            return {"error": f"Sub-agent error: {e}", "run_id": run_id, "success": False}

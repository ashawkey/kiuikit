"""Sub-agent spawning and lifecycle management for kiui agent."""

import json
import os
import subprocess
import sys
import time
import threading
import queue
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

if sys.platform != "win32":
    import select as _select

from kiui.agent.utils import get_kia_dir


@dataclass
class SubagentRun:
    """State for a single sub-agent run."""

    id: str
    task: str
    label: str | None
    process: subprocess.Popen
    result_file: Path
    started_at: datetime
    mode: str = "run"
    timeout_seconds: int = 0
    completed: bool = False
    last_activity: datetime | None = None
    result: dict | None = None
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    _read_buffer: bytes = b""
    # Windows-specific: Queue for stdout reader thread to prevent race conditions
    _stdout_queue: Any = None


class SubagentManager:
    """Spawn and manage isolated sub-agent processes (synchronous)."""

    def __init__(
        self,
        model_key: str,
        max_children: int = 5,
        max_depth: int = 3,
    ):
        self.model_key = model_key
        self.max_children = max_children
        self.max_depth = max_depth
        self.active_runs: dict[str, SubagentRun] = {}
        self._depth = int(os.environ.get("KIA_SPAWN_DEPTH", "0"))
        self._pending_results: list[dict] = []
        self._lock = threading.Lock()

    def _check_limits(self, session_label: str | None = None) -> str | None:
        """Return an error string if spawn limits are exceeded, else None.

        Must be called while holding ``self._lock``.
        """
        if self._depth >= self.max_depth:
            return f"Max spawn depth reached ({self.max_depth})."
        active_count = sum(1 for r in self.active_runs.values() if not r.completed)
        if active_count >= self.max_children:
            return f"Max active children reached ({self.max_children})."
        if session_label is not None:
            for r in self.active_runs.values():
                if not r.completed and r.label == session_label and r.mode == "session":
                    return f"Session with label '{session_label}' already exists."
        return None

    def _spawn_env(self) -> dict[str, str]:
        return {**os.environ, "KIA_SPAWN_DEPTH": str(self._depth + 1)}

    def spawn(
        self,
        task: str,
        label: str | None = None,
        timeout_seconds: int = 0,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Spawn a one-shot sub-agent that runs a task in the background.

        The sub-agent runs `kia exec --model <key> "task"` as a subprocess.
        A background thread monitors completion and stores the result.
        """
        with self._lock:
            if err := self._check_limits():
                return {"error": err, "success": False}

            run_id = str(uuid4())[:8]
            work_dir = cwd or os.getcwd()
            result_file = get_kia_dir(work_dir) / "subagent-runs" / f"{run_id}.json"
            result_file.parent.mkdir(parents=True, exist_ok=True)

            process = subprocess.Popen(
                [
                    sys.executable, "-m", "kiui.agent.cli",
                    "exec", "--model", self.model_key,
                    "--result-file", str(result_file),
                    "--prompt", task,
                ],
                cwd=work_dir, env=self._spawn_env(),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )

            run = SubagentRun(
                id=run_id, task=task, label=label, process=process,
                result_file=result_file, started_at=datetime.now(),
                mode="run", timeout_seconds=timeout_seconds,
            )
            self.active_runs[run_id] = run

        threading.Thread(target=self._wait_for_completion, args=(run,), daemon=True).start()

        return {
            "message": f"Sub-agent {run_id} started: {task[:80]}",
            "run_id": run_id,
            "success": True,
        }

    def spawn_session(
        self,
        label: str,
        cwd: str | None = None,
        idle_timeout: int = 1800,
    ) -> dict[str, Any]:
        """Spawn a persistent sub-agent session communicating via JSON pipes.

        The sub-agent runs `kia pipe --model <key>` and stays alive for
        follow-up messages via send().
        """
        with self._lock:
            if err := self._check_limits(session_label=label):
                return {"error": err, "success": False}

            run_id = str(uuid4())[:8]
            work_dir = cwd or os.getcwd()
            result_file = get_kia_dir(work_dir) / "subagent-runs" / f"{run_id}.json"
            result_file.parent.mkdir(parents=True, exist_ok=True)

            process = subprocess.Popen(
                [
                    sys.executable, "-m", "kiui.agent.cli",
                    "pipe", "--model", self.model_key,
                ],
                cwd=work_dir, env=self._spawn_env(),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )

            run = SubagentRun(
                id=run_id, task=f"[session:{label}]", label=label, process=process,
                result_file=result_file, started_at=datetime.now(),
                mode="session", last_activity=datetime.now(),
            )

            if sys.platform == "win32":
                run._stdout_queue = queue.Queue()
                threading.Thread(target=self._windows_reader_thread, args=(run,), daemon=True).start()

            self.active_runs[run_id] = run

        if idle_timeout > 0:
            threading.Thread(target=self._idle_watcher, args=(run, idle_timeout), daemon=True).start()

        return {
            "message": f"Persistent session '{label}' started (id={run_id}). Use send_to_subagent to communicate.",
            "run_id": run_id,
            "label": label,
            "success": True,
        }

    def send(self, target: str, message: str, timeout: float = 120.0) -> dict[str, Any]:
        """Send a message to a persistent session and return its response."""
        run = self._find_session(target)
        if not run:
            return {"error": f"No active session found for '{target}'.", "success": False}
        if run.completed:
            return {"error": f"Session '{target}' has ended.", "success": False}
        if run.mode != "session":
            return {"error": f"'{target}' is a one-shot run, not a persistent session.", "success": False}

        if run.process.poll() is not None:
            run.completed = True
            run.exit_code = run.process.returncode
            run.error = f"Process exited with code {run.process.returncode}"
            return {"error": f"Session process has died (exit code {run.process.returncode}).", "success": False}

        run.last_activity = datetime.now()

        msg_json = json.dumps({"type": "message", "content": message}) + "\n"
        try:
            run.process.stdin.write(msg_json.encode("utf-8"))
            run.process.stdin.flush()
        except (BrokenPipeError, OSError):
            run.completed = True
            run.error = "Process stdin closed"
            return {"error": "Session process has died.", "success": False}

        deadline = time.time() + timeout
        buf = run._read_buffer
        run._read_buffer = b""

        while b"\n" not in buf:
            remaining = deadline - time.time()
            if remaining <= 0:
                run._read_buffer = buf
                self._kill_run(run, f"Timed out after {timeout}s")
                return {"error": f"Timed out waiting for response after {timeout}s. Session killed.", "success": False}

            chunk = self._read_stdout_chunk(run, min(remaining, 2.0))
            if chunk is None:
                run.completed = True
                run.error = "Process stdout error"
                return {"error": "Session process stdout error.", "success": False}
            if chunk == b"":
                if run.process.poll() is not None:
                    run.completed = True
                    run.error = "Process exited during read"
                    return {"error": "Session process has ended.", "success": False}
                continue
            buf += chunk

        resp_line, _, remainder = buf.partition(b"\n")
        run._read_buffer = remainder

        try:
            resp = json.loads(resp_line.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            return {"error": f"Invalid JSON from session: {resp_line[:200]}", "success": False}

        run.last_activity = datetime.now()
        return {
            "response": resp.get("content", ""),
            "usage": resp.get("usage"),
            "success": True,
        }

    def kill(self, run_id: str) -> dict[str, Any]:
        """Kill a running sub-agent by ID."""
        run = self.active_runs.get(run_id)
        if not run:
            return {"error": f"No run with id {run_id}.", "success": False}
        if not run.completed:
            self._kill_run(run, "Killed by parent")
        return {"message": f"Sub-agent {run_id} killed.", "success": True}

    def list_runs(self) -> list[dict[str, Any]]:
        """Return a summary of all tracked sub-agent runs."""
        runs = []
        for r in self.active_runs.values():
            info = {
                "id": r.id,
                "task": r.task[:100],
                "label": r.label,
                "mode": r.mode,
                "status": "completed" if r.completed else "running",
                "started": r.started_at.isoformat(),
            }
            if r.completed and r.result:
                info["result_summary"] = r.result.get("summary", "")[:200]
            if r.mode == "session" and r.last_activity and not r.completed:
                info["idle_seconds"] = int((datetime.now() - r.last_activity).total_seconds())
            if r.error:
                info["error"] = r.error
            runs.append(info)
        return runs

    def get_pending_results(self) -> list[dict]:
        """Return and clear pending completion messages for one-shot runs."""
        with self._lock:
            results = list(self._pending_results)
            self._pending_results.clear()
        return results

    def kill_all(self):
        """Kill all active sub-agents. Used at shutdown."""
        for run in self.active_runs.values():
            if not run.completed:
                self._kill_run(run, "Killed by parent (shutdown)")

    @staticmethod
    def _read_stdout_chunk(run: SubagentRun, wait: float) -> bytes | None:
        """Read up to 8192 bytes from a sub-agent's stdout with a timeout.

        Returns the bytes read, b"" if nothing was ready within *wait*
        seconds, or None on an unrecoverable I/O error.
        """
        if sys.platform != "win32":
            try:
                ready, _, _ = _select.select([run.process.stdout], [], [], wait)
            except (ValueError, OSError):
                return None
            if not ready:
                return b""
            reader = run.process.stdout
            chunk = reader.read1(8192) if hasattr(reader, "read1") else reader.read(8192)
            return chunk if chunk else b""

        # Windows: read from queue populated by _windows_reader_thread
        if run._stdout_queue is None:
            return None 
            
        try:
            return run._stdout_queue.get(timeout=wait)
        except queue.Empty:
            return b""

    @staticmethod
    def _windows_reader_thread(run: SubagentRun):
        """Background thread to read stdout on Windows."""
        try:
            reader = run.process.stdout
            while not run.completed:
                # read1 is preferred if available (BufferedIO), else read(1)
                data = reader.read1(8192) if hasattr(reader, "read1") else reader.read(1)
                if not data:
                    break
                run._stdout_queue.put(data)
        except (OSError, ValueError):
            pass

    def _find_session(self, target: str) -> SubagentRun | None:
        """Find an active session by label or run_id."""
        if target in self.active_runs:
            return self.active_runs[target]
        for r in self.active_runs.values():
            if r.label == target and not r.completed and r.mode == "session":
                return r
        return None

    def _wait_for_completion(self, run: SubagentRun):
        """Background thread: wait for one-shot process to finish and collect output."""
        try:
            if run.timeout_seconds > 0:
                try:
                    stdout, stderr = run.process.communicate(timeout=run.timeout_seconds)
                except subprocess.TimeoutExpired:
                    run.process.kill()
                    stdout, stderr = run.process.communicate()
                    with self._lock:
                        run.error = "Timed out"
                        run.completed = True
                    self._enqueue_result(run)
                    return
            else:
                stdout, stderr = run.process.communicate()

            run.exit_code = run.process.returncode
            run.stdout = stdout.decode("utf-8", errors="replace") if stdout else ""
            run.stderr = stderr.decode("utf-8", errors="replace") if stderr else ""

            if run.result_file.exists():
                try:
                    run.result = json.loads(run.result_file.read_text())
                except (json.JSONDecodeError, OSError):
                    run.result = None

            with self._lock:
                run.completed = True
            self._enqueue_result(run)

        except Exception as e:
            with self._lock:
                run.error = str(e)
                run.completed = True
            self._enqueue_result(run)

    def _enqueue_result(self, run: SubagentRun):
        """Add a completion message to pending results for the main agent."""
        summary = ""
        if run.result:
            summary = run.result.get("summary", run.result.get("response", ""))
        if run.error:
            summary = f"Error: {run.error}"
        if not summary and run.stdout:
            summary = run.stdout[:500]

        label = run.label or run.task[:60]
        msg = {
            "role": "system",
            "content": (
                f"[Sub-agent '{label}' (id={run.id}) completed]\n"
                f"Exit code: {run.exit_code}\n"
                f"Result: {summary[:1000]}"
            ),
        }
        with self._lock:
            self._pending_results.append(msg)

    def _idle_watcher(self, run: SubagentRun, idle_timeout: int):
        """Background thread: kill session after idle_timeout seconds of inactivity."""
        while not run.completed:
            time.sleep(60)
            if run.completed:
                break
            if run.last_activity and (datetime.now() - run.last_activity).total_seconds() > idle_timeout:
                self._kill_run(run, f"Idle timeout ({idle_timeout}s)")
                break

    def _kill_run(self, run: SubagentRun, reason: str):
        """Kill a run and clean up."""
        if run.completed:
            return

        if run.mode == "session" and run.process.stdin:
            try:
                shutdown = json.dumps({"type": "shutdown"}) + "\n"
                run.process.stdin.write(shutdown.encode("utf-8"))
                run.process.stdin.flush()
                run.process.wait(timeout=2)
            except Exception:
                pass

        if run.process.poll() is None:
            try:
                run.process.kill()
            except ProcessLookupError:
                pass

        run.completed = True
        run.error = reason

"""Managed background-process tools owned by the ``monitor`` skill.

These tools are injected into the agent's tool surface only while the monitor
skill is loaded (via ``load_skill`` or ``/skills monitor``). Each entry exposes
an OpenAI function *schema*, a ``run`` callable invoked as ``run(executor,
**arguments)``, and a ``permission`` class consulted by the permission
controller. The executor owns the process *registry* and cleanup
(``ProcessManagerMixin``); these functions drive it.
"""

import json
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from kiui.agent.tools.constants import (
    MAX_PROCESS_LOG_BYTES,
    MAX_PROCESS_LOG_TAIL_CHARS,
    MAX_TOOL_OUTPUT_CHARS,
)
from kiui.agent.tools.process_util import (
    _close_windows_job,
    _create_windows_job,
    _resume_windows_process,
    _terminate_process,
    _windows_job_active_processes,
)
from kiui.agent.utils.interrupt import CancelWatcher


def _release_completed_windows_job(record: dict[str, Any]) -> bool:
    """Close a completed Windows job exactly once."""
    with record["job_lock"]:
        job_handle = record["job_handle"]
        if job_handle is None or _windows_job_active_processes(job_handle):
            return False
        record["job_handle"] = None
    _close_windows_job(job_handle)
    return True


def _process_info(record: dict[str, Any]) -> dict[str, Any]:
    with record["state_lock"]:
        proc = record["process"]
        exit_code = record["exit_code"]
        if proc is not None:
            polled = proc.poll()
            if polled is not None:
                record["exit_code"] = exit_code = polled
    if record["job_handle"] is not None:
        _release_completed_windows_job(record)
        if record["job_handle"] is not None:
            exit_code = None
    return {
        "process_id": record["process_id"],
        "pid": record["pid"],
        "status": "running" if exit_code is None else "exited",
        "exit_code": exit_code,
        "command": record["command"],
        "cwd": record["cwd"],
        "log_path": record["log_path"],
        "log_truncated": record["log_truncated"],
        "log_error": record.get("log_error"),
        "capture_error": record.get("capture_error"),
    }


def start_process(executor, command: str, cwd: str | None = None) -> dict[str, Any]:
    """Start a session-managed background process with file-backed output."""
    cwd = str(executor._resolve_path(cwd or "."))
    executor.console.tool(f"start_process: {command} (cwd={cwd})")

    process_id = f"p-{uuid.uuid4().hex[:8]}"
    log_dir = executor._resolve_path(".kia/processes")
    log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    log_dir.chmod(0o700)
    log_path = log_dir / f"{process_id}.log"
    log_file = log_path.open("xb", buffering=0)
    try:
        log_path.chmod(0o600)
        if sys.platform == "win32":
            shell_cmd = ["powershell", "-NoLogo", "-Command", command]
            proc = subprocess.Popen(
                shell_cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000004  # CREATE_SUSPENDED
                ),
            )
            job_handle = None
            try:
                job_handle = _create_windows_job(proc)
                _resume_windows_process(proc)
            except Exception:
                if job_handle is not None:
                    _close_windows_job(job_handle, terminate=True)
                else:
                    _terminate_process(proc)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
                raise
            process_backend = "windows_job"
        elif sys.platform.startswith("linux"):
            job_handle = None
            supervisor = Path(__file__).parent / "process_supervisor.py"
            shell_cmd = [sys.executable, str(supervisor), command]
            proc = subprocess.Popen(
                shell_cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                start_new_session=True,
            )
            process_backend = "linux_supervisor"
        else:
            job_handle = None
            shell_cmd = ["/bin/bash", "-lc", command]
            proc = subprocess.Popen(
                shell_cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                start_new_session=True,
            )
            process_backend = "process_group"
    except Exception:
        log_file.close()
        log_path.unlink(missing_ok=True)
        raise

    record = {
        "process_id": process_id,
        "pid": proc.pid,
        "process": proc,
        "exit_code": None,
        "command": command,
        "cwd": cwd,
        "log_path": str(log_path.relative_to(executor._resolve_path("."))),
        "started_at": time.time(),
        "log_truncated": False,
        "job_handle": job_handle,
        "process_backend": process_backend,
        "job_lock": threading.Lock(),
        "state_lock": threading.Lock(),
    }

    def _capture_output() -> None:
        written = 0
        log_enabled = True
        try:
            while chunk := proc.stdout.read1(65536):
                remaining = MAX_PROCESS_LOG_BYTES - written
                if remaining > 0 and log_enabled:
                    data = chunk[:remaining]
                    try:
                        log_file.write(data)
                    except OSError as e:
                        record["log_error"] = str(e)
                        record["log_truncated"] = True
                        log_enabled = False
                    else:
                        written += len(data)
                if len(chunk) > remaining:
                    record["log_truncated"] = True
        except OSError as e:
            record["capture_error"] = str(e)
        finally:
            proc.stdout.close()
            log_file.close()
            proc.wait()
            with record["state_lock"]:
                record["exit_code"] = proc.returncode
            if record["process_backend"] == "windows_job":
                while record["job_handle"] is not None:
                    _release_completed_windows_job(record)
                    if record["job_handle"] is not None:
                        time.sleep(0.1)
            with record["state_lock"]:
                record["process"] = None

    capture_thread = threading.Thread(target=_capture_output, daemon=True)
    record["capture_thread"] = capture_thread
    with executor._process_lock:
        executor._processes[process_id] = record
    capture_thread.start()
    return {**_process_info(record), "success": True}


def inspect_processes(
    executor, process_id: str | None = None, wait: float = 0, log_tail_chars: int = 0
) -> dict[str, Any]:
    """Optionally wait, then return status and a bounded log tail."""
    executor.console.tool(
        f"inspect_processes: {process_id or 'all'} "
        f"(wait={wait}s, log_tail_chars={log_tail_chars})"
    )
    if wait < 0:
        return {"error": "wait must be non-negative", "success": False}
    if log_tail_chars < 0 or log_tail_chars > MAX_PROCESS_LOG_TAIL_CHARS:
        return {
            "error": f"log_tail_chars must be between 0 and {MAX_PROCESS_LOG_TAIL_CHARS}",
            "success": False,
        }
    if log_tail_chars and process_id is None:
        return {
            "error": "process_id is required when log_tail_chars is non-zero",
            "success": False,
        }

    record = None
    if process_id is not None:
        with executor._process_lock:
            record = executor._processes.get(process_id)
        if record is None:
            return {"error": f"Unknown managed process: {process_id}", "success": False}

    if wait:
        deadline = time.monotonic() + wait
        interrupted = False
        try:
            with CancelWatcher(executor.cancellation) as watcher:
                while (remaining := deadline - time.monotonic()) > 0:
                    if watcher.is_cancelled:
                        interrupted = True
                        break
                    time.sleep(min(0.1, remaining))
        except KeyboardInterrupt:
            interrupted = True
        if interrupted:
            return {
                "error": "Process inspection wait was interrupted by user.",
                "success": False,
                "interrupted": True,
            }

    if record is not None:
        records = [record]
    else:
        with executor._process_lock:
            records = list(executor._processes.values())
    processes = [_process_info(record) for record in records]
    if log_tail_chars:
        log_path = executor._resolve_path(processes[0]["log_path"])
        size = log_path.stat().st_size
        # Read extra bytes so a multibyte UTF-8 boundary does not shorten the requested tail.
        byte_limit = log_tail_chars * 4
        start = max(0, size - byte_limit)
        with log_path.open("rb") as f:
            f.seek(start)
            decoded = f.read().decode("utf-8", errors="replace")
        processes[0]["log_tail"] = decoded[-log_tail_chars:]
        processes[0]["log_tail_truncated"] = start > 0 or len(decoded) > log_tail_chars
    truncated = bool(log_tail_chars and processes[0].get("log_tail_truncated"))
    result = {
        "processes": processes,
        "count": len(processes),
        "truncated": truncated,
        "success": True,
    }
    if truncated:
        result["truncation_reason"] = "log tail limit"
        result["guidance"] = "Request a different log tail, or read the log file in focused slices."

    if len(json.dumps(result, indent=2)) > MAX_TOOL_OUTPUT_CHARS:
        result["truncated"] = True
        result["truncation_reason"] = "character cap"
        result["guidance"] = "Inspect one process at a time or read its log file in focused slices."
        if log_tail_chars:
            tail = processes[0]["log_tail"]
            overflow = len(json.dumps(result, indent=2)) - MAX_TOOL_OUTPUT_CHARS
            processes[0]["log_tail"] = tail[overflow:]
            processes[0]["log_tail_truncated"] = True
        while processes and len(json.dumps(result, indent=2)) > MAX_TOOL_OUTPUT_CHARS:
            processes.pop()
        result["count"] = len(processes)
    return result


def stop_process(executor, process_id: str) -> dict[str, Any]:
    """Stop one managed process and its process tree."""
    executor.console.tool(f"stop_process: {process_id}")
    with executor._process_lock:
        record = executor._processes.get(process_id)
    if record is None:
        return {"error": f"Unknown managed process: {process_id}", "success": False}

    executor._stop_process_record(record)
    return {**_process_info(record), "success": True}


TOOLS = [
    {
        "permission": "risky",
        "run": start_process,
        "schema": {
            "type": "function",
            "function": {
                "name": "start_process",
                "description": (
                    "Start a managed background process and return immediately. "
                    "Its combined output is written to a readable log file."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to run in the background"},
                        "cwd": {"type": "string", "description": "Working directory (optional)"},
                    },
                    "required": ["command"],
                },
            },
        },
    },
    {
        "permission": "safe",
        "run": inspect_processes,
        "schema": {
            "type": "function",
            "function": {
                "name": "inspect_processes",
                "description": (
                    "Inspect managed background process status after an optional bounded wait. "
                    "Optionally include a bounded tail from one process's log. "
                    "Omit process_id to list all processes."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "process_id": {"type": "string", "description": "Process ID to inspect (optional)"},
                        "wait": {
                            "type": "number",
                            "minimum": 0,
                            "default": 0,
                            "description": "Seconds to wait before returning a status snapshot (default: 0)",
                        },
                        "log_tail_chars": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": MAX_PROCESS_LOG_TAIL_CHARS,
                            "default": 0,
                            "description": (
                                f"Characters of recent log content to return directly (default: 0, max: {MAX_PROCESS_LOG_TAIL_CHARS}). "
                                "Requires process_id."
                            ),
                        },
                    },
                    "required": [],
                },
            },
        },
    },
    {
        "permission": "risky",
        "run": stop_process,
        "schema": {
            "type": "function",
            "function": {
                "name": "stop_process",
                "description": "Stop a managed background process.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "process_id": {"type": "string", "description": "Managed process ID returned by start_process"},
                    },
                    "required": ["process_id"],
                },
            },
        },
    },
]

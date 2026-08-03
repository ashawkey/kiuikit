"""Core managed-process tools, registry, status, and lifecycle."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from .constants import (
    MAX_PROCESS_LOG_BYTES,
    MAX_PROCESS_LOG_TAIL_CHARS,
    MAX_TOOL_OUTPUT_CHARS,
)
from .process_util import (
    _close_windows_job,
    _create_windows_job,
    _resume_windows_process,
    _terminate_process,
    _windows_job_active_processes,
)


def format_process_status(running: int, finished: int) -> str:
    """Return the compact process summary shared by terminal and web UIs."""
    if running <= 0:
        return ""
    return f"(Proc: {running} running [{finished} finished])"


class ProcessManagerMixin:
    """Manage session-scoped background processes and their UI observers."""

    def _init_process_registry(self) -> None:
        self._processes: dict[str, dict[str, Any]] = {}
        self._process_lock = threading.Lock()
        self._process_listeners: set[Callable[[int, int], None]] = set()
        self._last_process_counts = (0, 0)
        self._process_status_callback = None

    def add_process_listener(
        self, listener: Callable[[int, int], None], *, notify: bool = True
    ) -> None:
        with self._process_lock:
            self._process_listeners.add(listener)
        if notify:
            listener(*self.process_counts())

    def set_process_status_callback(
        self, callback: Callable[[int, int], None] | None
    ) -> None:
        """Install the session-level observer used by terminal and web UIs."""
        previous = getattr(self, "_process_status_callback", None)
        if previous is not None:
            self.remove_process_listener(previous)
        self._process_status_callback = callback
        if callback is not None:
            self.add_process_listener(callback)

    def remove_process_listener(self, listener: Callable[[int, int], None]) -> None:
        with self._process_lock:
            self._process_listeners.discard(listener)

    def process_counts(self) -> tuple[int, int]:
        """Return ``(running, finished)`` from authoritative process records."""
        with self._process_lock:
            records = list(self._processes.values())
        running = sum(self._process_info(record)["status"] == "running" for record in records)
        return running, len(records) - running

    def _notify_process_status(self, *, force: bool = False) -> None:
        counts = self.process_counts()
        with self._process_lock:
            if not force and counts == self._last_process_counts:
                return
            self._last_process_counts = counts
            listeners = list(self._process_listeners)
        for listener in listeners:
            try:
                listener(*counts)
            except Exception:
                # Process lifecycle must not depend on a UI observer.
                pass

    @staticmethod
    def _release_completed_windows_job(record: dict[str, Any]) -> bool:
        """Close a completed Windows job exactly once."""
        with record["job_lock"]:
            job_handle = record["job_handle"]
            if job_handle is None or _windows_job_active_processes(job_handle):
                return False
            record["job_handle"] = None
        _close_windows_job(job_handle)
        return True

    def _process_info(self, record: dict[str, Any]) -> dict[str, Any]:
        with record["state_lock"]:
            proc = record["process"]
            exit_code = record["exit_code"]
            if proc is not None:
                polled = proc.poll()
                if polled is not None:
                    record["exit_code"] = exit_code = polled
        if record["job_handle"] is not None:
            self._release_completed_windows_job(record)
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

    def _start_process(self, command: str, cwd: str | None = None) -> dict[str, Any]:
        """Start a session-managed background process with file-backed output."""
        cwd = str(self._resolve_path(cwd or "."))
        process_id = f"p-{uuid.uuid4().hex[:8]}"
        log_dir = self._resolve_path(".kia/processes")
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
                        subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000004
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
            "log_path": str(log_path.relative_to(self._resolve_path("."))),
            "started_at": time.time(),
            "log_truncated": False,
            "job_handle": job_handle,
            "process_backend": process_backend,
            "job_lock": threading.Lock(),
            "state_lock": threading.Lock(),
        }

        def capture_output() -> None:
            written = 0
            log_enabled = True
            try:
                while chunk := proc.stdout.read1(65536):
                    remaining = MAX_PROCESS_LOG_BYTES - written
                    if remaining > 0 and log_enabled:
                        data = chunk[:remaining]
                        try:
                            log_file.write(data)
                        except OSError as exc:
                            record["log_error"] = str(exc)
                            record["log_truncated"] = True
                            log_enabled = False
                        else:
                            written += len(data)
                    if len(chunk) > remaining:
                        record["log_truncated"] = True
            except OSError as exc:
                record["capture_error"] = str(exc)
            finally:
                proc.stdout.close()
                log_file.close()
                proc.wait()
                with record["state_lock"]:
                    record["exit_code"] = proc.returncode
                if record["process_backend"] == "windows_job":
                    while record["job_handle"] is not None:
                        self._release_completed_windows_job(record)
                        if record["job_handle"] is not None:
                            time.sleep(0.1)
                with record["state_lock"]:
                    record["process"] = None
                self._notify_process_status()

        capture_thread = threading.Thread(target=capture_output, daemon=True)
        record["capture_thread"] = capture_thread
        with self._process_lock:
            self._processes[process_id] = record
        capture_thread.start()
        self._notify_process_status()
        return {**self._process_info(record), "success": True}

    def inspect_processes(
        self, process_id: str | None = None, log_tail_chars: int = 0
    ) -> dict[str, Any]:
        """Return process status and an optional bounded log tail."""
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
            with self._process_lock:
                record = self._processes.get(process_id)
                if record is None and process_id.isdigit():
                    pid = int(process_id)
                    record = next(
                        (item for item in self._processes.values() if item["pid"] == pid),
                        None,
                    )
            if record is None:
                return {"error": f"Unknown managed process: {process_id}", "success": False}

        if record is not None:
            records = [record]
        else:
            with self._process_lock:
                records = list(self._processes.values())
        processes = [self._process_info(item) for item in records]
        if log_tail_chars:
            log_path = self._resolve_path(processes[0]["log_path"])
            size = log_path.stat().st_size
            byte_limit = log_tail_chars * 4
            start = max(0, size - byte_limit)
            with log_path.open("rb") as handle:
                handle.seek(start)
                decoded = handle.read().decode("utf-8", errors="replace")
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

    def _inspect_processes(
        self, process_id: str | None = None, log_tail_chars: int = 0
    ) -> dict[str, Any]:
        return self.inspect_processes(process_id, log_tail_chars)

    @staticmethod
    def _stop_process_record(record: dict[str, Any]) -> None:
        with record["state_lock"]:
            proc = record["process"]
        if proc is None:
            record["capture_thread"].join(timeout=5)
            return
        backend = record["process_backend"]
        if backend == "windows_job":
            with record["job_lock"]:
                job_handle = record["job_handle"]
                if job_handle is not None:
                    _close_windows_job(job_handle, terminate=True)
                    record["job_handle"] = None
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        elif backend == "linux_supervisor":
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=7)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        elif proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
        record["capture_thread"].join(timeout=5)

    def _stop_process(self, process_id: str) -> dict[str, Any]:
        """Stop one managed process and its process tree."""
        with self._process_lock:
            record = self._processes.get(process_id)
        if record is None:
            return {"error": f"Unknown managed process: {process_id}", "success": False}
        self._stop_process_record(record)
        self._notify_process_status()
        return {**self._process_info(record), "success": True}

    def shutdown_processes(self, clear: bool = False) -> None:
        """Stop all running managed processes, optionally forgetting their records."""
        with self._process_lock:
            records = list(self._processes.values())
        for record in records:
            self._stop_process_record(record)
        if clear:
            with self._process_lock:
                self._processes.clear()
        self._notify_process_status()

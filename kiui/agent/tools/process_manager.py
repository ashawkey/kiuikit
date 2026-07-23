"""Core managed-process registry and lifecycle.

The executor always owns the process *registry* and cleanup, so background
processes are reliably terminated on session switch, ``/clear``, and exit even
when the ``monitor`` skill (which provides the LLM-facing start/inspect/stop
tools) is not loaded. The tool surface lives in ``bundled_skills/monitor``.
"""

import os
import signal
import subprocess
import threading
from typing import Any

from .process_util import _close_windows_job


class ProcessManagerMixin:
    """Manage the lifecycle of session-scoped background processes."""

    def _init_process_registry(self) -> None:
        self._processes: dict[str, dict[str, Any]] = {}
        self._process_lock = threading.Lock()

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

    def shutdown_processes(self, clear: bool = False) -> None:
        """Stop all running managed processes, optionally forgetting their records."""
        with self._process_lock:
            records = list(self._processes.values())
        for record in records:
            self._stop_process_record(record)
        if clear:
            with self._process_lock:
                self._processes.clear()

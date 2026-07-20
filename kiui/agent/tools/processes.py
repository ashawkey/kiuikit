"""Session-managed background process tools."""

import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .constants import MAX_PROCESS_LOG_BYTES


def _create_windows_job(proc: subprocess.Popen) -> int:
    """Place a process in a kill-on-close Windows Job Object."""
    import ctypes
    from ctypes import wintypes

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
    ]
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = 0x00002000  # KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
        error = ctypes.WinError(ctypes.get_last_error())
        kernel32.CloseHandle(job)
        raise error
    if not kernel32.AssignProcessToJobObject(job, wintypes.HANDLE(proc._handle)):
        error = ctypes.WinError(ctypes.get_last_error())
        kernel32.CloseHandle(job)
        raise error
    return int(job)


def _windows_job_active_processes(job: int) -> int:
    """Return the number of processes still active in a Windows Job Object."""
    import ctypes
    from ctypes import wintypes

    class JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("TotalUserTime", ctypes.c_longlong),
            ("TotalKernelTime", ctypes.c_longlong),
            ("ThisPeriodTotalUserTime", ctypes.c_longlong),
            ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
            ("TotalPageFaultCount", wintypes.DWORD),
            ("TotalProcesses", wintypes.DWORD),
            ("ActiveProcesses", wintypes.DWORD),
            ("TotalTerminatedProcesses", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryInformationJobObject.restype = wintypes.BOOL
    info = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
    if not kernel32.QueryInformationJobObject(
        wintypes.HANDLE(job), 1, ctypes.byref(info), ctypes.sizeof(info), None
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(info.ActiveProcesses)


def _resume_windows_process(proc: subprocess.Popen) -> None:
    """Resume the primary thread of a process created with CREATE_SUSPENDED."""
    import ctypes
    from ctypes import wintypes

    class THREADENTRY32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ThreadID", wintypes.DWORD),
            ("th32OwnerProcessID", wintypes.DWORD),
            ("tpBasePri", wintypes.LONG),
            ("tpDeltaPri", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Thread32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(THREADENTRY32)]
    kernel32.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(THREADENTRY32)]
    kernel32.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenThread.restype = wintypes.HANDLE
    kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
    kernel32.ResumeThread.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000004, 0)  # TH32CS_SNAPTHREAD
    invalid_handle = ctypes.c_void_p(-1).value
    if snapshot == invalid_handle:
        raise ctypes.WinError(ctypes.get_last_error())

    thread = None
    try:
        entry = THREADENTRY32(dwSize=ctypes.sizeof(THREADENTRY32))
        found = kernel32.Thread32First(snapshot, ctypes.byref(entry))
        while found:
            if entry.th32OwnerProcessID == proc.pid:
                thread = kernel32.OpenThread(0x0002, False, entry.th32ThreadID)
                if not thread:
                    raise ctypes.WinError(ctypes.get_last_error())
                break
            found = kernel32.Thread32Next(snapshot, ctypes.byref(entry))
        if thread is None:
            raise RuntimeError(f"No thread found for suspended process {proc.pid}")
        if kernel32.ResumeThread(wintypes.HANDLE(thread)) == 0xFFFFFFFF:
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        if thread is not None:
            kernel32.CloseHandle(wintypes.HANDLE(thread))
        kernel32.CloseHandle(wintypes.HANDLE(snapshot))


def _close_windows_job(job: int, terminate: bool = False) -> None:
    """Close a Windows Job Object, optionally terminating all contained processes."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    error = None
    if terminate and not kernel32.TerminateJobObject(wintypes.HANDLE(job), 1):
        error = ctypes.WinError(ctypes.get_last_error())
    if not kernel32.CloseHandle(wintypes.HANDLE(job)) and error is None:
        error = ctypes.WinError(ctypes.get_last_error())
    if error is not None:
        raise error


def _terminate_process(proc: subprocess.Popen) -> None:
    """Kill *proc* and its child process tree (best-effort)."""
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        else:
            # start_new_session=True makes the child PID its process-group ID;
            # that group can outlive the leader when a command backgrounds work.
            os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass


class ProcessToolsMixin:
    def _start_process(self, command: str, cwd: str | None = None) -> dict[str, Any]:
        """Start a session-managed background process with file-backed output."""
        cwd = str(self._resolve_path(cwd or "."))
        self.console.tool(f"start_process: {command} (cwd={cwd})")

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
                supervisor = Path(__file__).parent.parent / "process_supervisor.py"
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
                        ProcessToolsMixin._release_completed_windows_job(record)
                        if record["job_handle"] is not None:
                            time.sleep(0.1)
                with record["state_lock"]:
                    record["process"] = None

        capture_thread = threading.Thread(target=_capture_output, daemon=True)
        record["capture_thread"] = capture_thread
        with self._process_lock:
            self._processes[process_id] = record
        capture_thread.start()
        return {**self._process_info(record), "success": True}

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

    @staticmethod
    def _process_info(record: dict[str, Any]) -> dict[str, Any]:
        with record["state_lock"]:
            proc = record["process"]
            exit_code = record["exit_code"]
            if proc is not None:
                polled = proc.poll()
                if polled is not None:
                    record["exit_code"] = exit_code = polled
        if record["job_handle"] is not None:
            ProcessToolsMixin._release_completed_windows_job(record)
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

    def _inspect_processes(self, process_id: str | None = None) -> dict[str, Any]:
        """Return status and metadata for one or all managed processes."""
        self.console.tool(f"inspect_processes: {process_id or 'all'}")
        with self._process_lock:
            if process_id is not None:
                record = self._processes.get(process_id)
                if record is None:
                    return {"error": f"Unknown managed process: {process_id}", "success": False}
                records = [record]
            else:
                records = list(self._processes.values())
        return {
            "processes": [self._process_info(record) for record in records],
            "success": True,
        }

    def _stop_process(self, process_id: str) -> dict[str, Any]:
        """Stop one managed process and its process tree."""
        self.console.tool(f"stop_process: {process_id}")
        with self._process_lock:
            record = self._processes.get(process_id)
        if record is None:
            return {"error": f"Unknown managed process: {process_id}", "success": False}

        self._stop_process_record(record)
        return {**self._process_info(record), "success": True}

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

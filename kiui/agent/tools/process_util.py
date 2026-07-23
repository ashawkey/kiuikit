"""Platform helpers for controlling OS processes.

Shared by the foreground ``exec_command`` tool (which only needs
``_terminate_process``) and the managed-background-process tools bundled with
the ``monitor`` skill. Keeping these here means the process-killing and
Windows Job Object machinery lives in one place regardless of where the
higher-level tools are defined.
"""

import os
import signal
import subprocess
import sys


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

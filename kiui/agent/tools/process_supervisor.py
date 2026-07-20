"""Linux subprocess supervisor used by the managed background-process tool."""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
import time

_STOPPING = False


def _request_stop(signum, frame) -> None:
    global _STOPPING
    _STOPPING = True


def _set_child_subreaper() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))


def _descendants(root_pid: int) -> list[int]:
    children: dict[int, list[int]] = {}
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            stat = open(f"/proc/{entry.name}/stat", encoding="utf-8").read()
            ppid = int(stat[stat.rfind(")") + 2:].split()[1])
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(entry.name))

    result: list[int] = []
    pending = list(children.get(root_pid, ()))
    while pending:
        pid = pending.pop()
        result.append(pid)
        pending.extend(children.get(pid, ()))
    return result


def _signal_descendants(signum: int) -> None:
    for pid in reversed(_descendants(os.getpid())):
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            pass


def _reap_children(shell_pid: int, shell_status: int | None) -> tuple[bool, int | None]:
    have_children = False
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return False, shell_status
        if pid == 0:
            return True, shell_status
        have_children = True
        if pid == shell_pid:
            shell_status = status


def _terminate_descendants(shell_pid: int, shell_status: int | None) -> int | None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        _signal_descendants(signal.SIGTERM)
        have_children, shell_status = _reap_children(shell_pid, shell_status)
        if not have_children:
            return shell_status
        time.sleep(0.05)

    _signal_descendants(signal.SIGKILL)
    while True:
        _signal_descendants(signal.SIGKILL)
        have_children, shell_status = _reap_children(shell_pid, shell_status)
        if not have_children:
            return shell_status
        time.sleep(0.05)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: process_supervisor.py <command>")

    _set_child_subreaper()
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, _request_stop)

    shell = subprocess.Popen(["/bin/bash", "-lc", sys.argv[1]])
    shell_status = None
    while True:
        if _STOPPING:
            _terminate_descendants(shell.pid, shell_status)
            return 128 + signal.SIGTERM

        have_children, shell_status = _reap_children(shell.pid, shell_status)
        if not have_children:
            if shell_status is None:
                return 1
            if os.WIFEXITED(shell_status):
                return os.WEXITSTATUS(shell_status)
            if os.WIFSIGNALED(shell_status):
                return 128 + os.WTERMSIG(shell_status)
            return 1
        time.sleep(0.05)


if __name__ == "__main__":
    raise SystemExit(main())

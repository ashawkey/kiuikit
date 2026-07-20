"""Tests for kiui.agent.permissions safety and permission modes."""

import sys

import pytest

from kiui.agent.permissions import (
    PermissionController,
    PermissionMode,
    SafetyGuard,
)

# The safety guard matches the shell that actually runs exec_command: Unix
# shells on POSIX, PowerShell on Windows. Unix-command patterns are only
# enforced (and only meaningful) on POSIX.
unix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="Unix shell/filesystem semantics"
)
windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="Windows shell semantics"
)


@unix_only
@pytest.mark.parametrize(
    "command",
    [
        ":(){ :|:& };:",
        "mkfs.ext4 /dev/sda1",
        "/sbin/mkfs.xfs /dev/nvme0n1p1",
        "wipefs -a /dev/sda",
        "sudo -n /sbin/reboot",
        "VAR=x env FOO=y /bin/rm -r /etc",
        "echo ok\nrm -rf /etc",
        "rm --recursive /etc/*",
        "chmod -R 000 /",
        "find /usr -delete",
        "dd if=image of=/dev/nvme0n1",
        "cat image > /dev/vda",
        "tee /dev/mapper/vg-root < image",
        "bash -c 'rm -rf /etc'",
        "python3 -c \"import os; os.system('rm -rf /')\"",
        "echo $(rm -rf /etc)",
        "eval 'rm -rf /etc'",
        "busybox rm -rf /etc",
        "if true; then rm -rf /etc; fi",
        "echo ok > /tmp/log; rm -rf /etc",
        "timeout 5 rm -rf /etc",
        "rm -rf /etc/nginx",
        "rm -rf /{etc,usr}",
        "git reset --hard",
        "git clean -fdx",
        "git restore .",
        "systemctl reboot",
        "zpool destroy tank",
    ],
)
def test_safety_blocks_dangerous_commands(command):
    allowed, reason = SafetyGuard().check("exec_command", {"command": command})
    assert not allowed and reason.startswith("Blocked:")


@pytest.mark.parametrize(
    "command",
    [
        "ls -la",
        "echo 'reboot tomorrow'",
        "rm -rf ./build",
        "find ./build -delete",
        "chmod -R u+rw ./build",
        "python script.py",
        "python3 -c 'print(1)'",
        "python -c 'import shutil; shutil.rmtree(\"/etc\")'",
        "node -e 'console.log(1)'",
        "bash -c 'echo ok'",
        "sh script.sh",
        "echo $(date)",
        "eval 'echo ok'",
        "xargs echo < files",
        "xargs rm -rf < files",
        "find . -exec rm -rf {} +",
        "rm -rf $TARGET",
        "$CMD --all",
        "echo ok > /tmp/log",
        "if true; then echo ok; fi",
    ],
)
def test_safety_allows_normal_commands(command):
    allowed, reason = SafetyGuard().check("exec_command", {"command": command})
    assert allowed and reason == ""


@unix_only
def test_start_process_uses_exec_command_safety_checks():
    allowed, reason = SafetyGuard().check(
        "start_process", {"command": "rm -rf /etc"}
    )
    assert not allowed and reason.startswith("Blocked:")


@unix_only
def test_safety_blocks_recursive_delete_of_work_dir(tmp_path):
    allowed, reason = SafetyGuard(work_dir=tmp_path).check(
        "exec_command", {"command": f"rm -rf {tmp_path}"}
    )
    assert not allowed and reason


def test_safety_allows_recursive_delete_in_unprotected_top_level_tree(tmp_path):
    allowed, reason = SafetyGuard().check(
        "exec_command", {"command": f"rm -rf {tmp_path / 'cache'}"}
    )
    assert allowed and reason == ""


@unix_only
def test_safety_blocks_symlink_to_critical_path(tmp_path):
    link = tmp_path / "system"
    link.symlink_to("/etc")
    allowed, reason = SafetyGuard(work_dir=tmp_path).check(
        "exec_command", {"command": "rm -rf system"}
    )
    assert not allowed and reason


@unix_only
def test_safety_blocks_relative_critical_delete_from_cwd(tmp_path):
    allowed, reason = SafetyGuard(work_dir=tmp_path).check(
        "exec_command", {"command": "rm -rf .", "cwd": "/etc"}
    )
    assert not allowed and reason


@unix_only
def test_safety_blocks_file_tool_write_to_block_device():
    allowed, reason = SafetyGuard().check(
        "write_file", {"file": "/dev/sda", "content": "data"}
    )
    assert not allowed and reason


@unix_only
@pytest.mark.parametrize("path", ["/", "/etc", "~"])
def test_safety_blocks_remove_file_on_critical_paths(path):
    allowed, reason = SafetyGuard().check("remove_file", {"file": path})
    assert not allowed and reason


@unix_only
def test_safety_blocks_remove_block_device():
    allowed, reason = SafetyGuard().check("remove_file", {"file": "/dev/sda"})
    assert not allowed and reason


def test_safety_allows_remove_file_elsewhere(tmp_path):
    allowed, reason = SafetyGuard().check(
        "remove_file", {"file": str(tmp_path / "cache")}
    )
    assert allowed and reason == ""


@windows_only
@pytest.mark.parametrize(
    "command",
    [
        "format c: /q /y",
        "echo x; diskpart /s script.txt",
        "Clear-Disk -Number 1",
        "Initialize-Disk 2",
        "Stop-Computer -Force",
        "Restart-Computer",
        "Remove-Item C:\\ -Recurse -Force",
        "rmdir C:\\ /s",
    ],
)
def test_safety_blocks_dangerous_windows_commands(command):
    allowed, reason = SafetyGuard().check("exec_command", {"command": command})
    assert not allowed and reason.startswith("Blocked:")


@windows_only
@pytest.mark.parametrize(
    "command",
    [
        "Get-ChildItem",
        "Remove-Item -Recurse .\\build",
        "echo 'format the report'",
        "git status",
    ],
)
def test_safety_allows_normal_windows_commands(command):
    allowed, reason = SafetyGuard().check("exec_command", {"command": command})
    assert allowed and reason == ""


class _FakeConsole:
    def __init__(self, answer="Yes"):
        self.answer = answer
        self.prompt_broker = None

    def print(self, *a, **k):
        pass

    def local(self, *a, **k):
        pass

    def select(self, *a, **k):
        return self.answer

    def ask_text(self, *a, **k):
        return ""


def test_background_process_permissions():
    ctrl = PermissionController(
        mode=PermissionMode.DEFAULT,
        console=_FakeConsole(answer="No"),
    )
    assert not ctrl.check("start_process", {"command": "python server.py"})[0]
    assert ctrl.check("inspect_processes", {})[0]
    assert not ctrl.check("stop_process", {"process_id": "p-test"})[0]


def test_outside_write_uses_normal_default_permission():
    ctrl = PermissionController(
        mode=PermissionMode.DEFAULT,
        console=_FakeConsole(answer="Yes"),
    )
    allowed, reason = ctrl.check(
        "write_file", {"file": "/etc/example.conf", "content": "x"}
    )
    assert allowed and reason == ""


def test_outside_write_allowed_in_auto_mode():
    ctrl = PermissionController(
        mode=PermissionMode.AUTO,
        console=_FakeConsole(),
    )
    allowed, reason = ctrl.check(
        "write_file", {"file": "/etc/example.conf", "content": "x"}
    )
    assert allowed and reason == ""

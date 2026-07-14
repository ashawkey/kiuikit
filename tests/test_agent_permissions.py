"""Tests for kiui.agent.permissions: SafetyGuard and PermissionController.

Focus on the path-containment override flow, which must actually honor the
user's choice (a prior bug ignored the prompt result because a non-empty
tuple is always truthy).
"""

import pytest

from kiui.agent.permissions import (
    PermissionController,
    PermissionMode,
    SafetyGuard,
)


# ----- SafetyGuard: dangerous commands -------------------------------------

def test_safety_blocks_fork_bomb():
    guard = SafetyGuard()
    allowed, reason = guard.check("exec_command", {"command": ":(){ :|:& };:"})
    assert not allowed and reason


def test_safety_allows_normal_command():
    guard = SafetyGuard()
    allowed, reason = guard.check("exec_command", {"command": "ls -la"})
    assert allowed and reason == ""


# ----- SafetyGuard: path containment ---------------------------------------

def test_check_path_inside_workdir(tmp_path):
    guard = SafetyGuard(work_dir=tmp_path)
    allowed, _ = guard.check_path(str(tmp_path / "sub" / "file.txt"))
    assert allowed


def test_check_path_outside_workdir(tmp_path):
    guard = SafetyGuard(work_dir=tmp_path)
    allowed, reason = guard.check_path("/etc/passwd")
    assert not allowed and reason


# ----- PermissionController: out-of-scope override -------------------------

class _FakeConsole:
    """Minimal console stub that answers select() based on offered choices.

    ``path_answer`` handles the out-of-scope override prompt; ``risky_answer``
    handles the subsequent risky-tool confirmation prompt.
    """

    def __init__(self, path_answer="Deny", risky_answer="Yes"):
        self._path_answer = path_answer
        self._risky_answer = risky_answer
        self.prompt_broker = None

    def print(self, *a, **k):
        pass

    def local(self, *a, **k):
        pass

    def select(self, *a, **k):
        choices = k.get("choices") or (a[1] if len(a) > 1 else [])
        if "Deny" in choices:  # path-override prompt
            return self._path_answer
        return self._risky_answer

    def ask_text(self, *a, **k):
        return ""


def test_out_of_scope_write_denied_when_user_declines(tmp_path):
    ctrl = PermissionController(
        mode=PermissionMode.DEFAULT,
        console=_FakeConsole(path_answer="Deny"),
        work_dir=tmp_path,
    )
    allowed, reason = ctrl.check("write_file", {"file": "/etc/evil.conf", "content": "x"})
    assert not allowed and reason


def test_out_of_scope_write_allowed_when_user_confirms(tmp_path):
    # After overriding the path warning, DEFAULT mode still prompts for the
    # risky write itself; approve that too.
    ctrl = PermissionController(
        mode=PermissionMode.DEFAULT,
        console=_FakeConsole(path_answer="Allow this call", risky_answer="Yes"),
        work_dir=tmp_path,
    )
    allowed, _ = ctrl.check("write_file", {"file": "/etc/evil.conf", "content": "x"})
    assert allowed


def test_out_of_scope_write_blocked_in_auto_mode(tmp_path):
    ctrl = PermissionController(
        mode=PermissionMode.AUTO,
        console=_FakeConsole(path_answer="Allow this call"),
        work_dir=tmp_path,
    )
    allowed, reason = ctrl.check("write_file", {"file": "/etc/evil.conf", "content": "x"})
    assert not allowed and reason

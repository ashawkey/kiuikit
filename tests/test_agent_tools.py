"""Tests for the core file tools: apply_edit, multi_edit, ls, and
gitignore-aware glob/grep (kiui.agent.tools / kiui.agent.tools.gitignore)."""

import base64
import os
import shlex
import sys
import threading
import time
from pathlib import Path

import pytest

import kiui.agent.tools as tools
import kiui.agent.tools.commands as command_tools
from kiui.agent.skills import load_skill_tools
from kiui.agent.tools import (
    ToolExecutor,
    _human_size,
    apply_edit,
    find_match,
    format_tool_result,
)
from kiui.agent.tools.gitignore import build_gitignore_matcher
from kiui.agent.permissions import SafetyGuard
from kiui.agent.utils.io import CancellationToken, EventHub


class _SilentConsole:
    def tool(self, *args, **kwargs):
        pass

    def print(self, *args, **kwargs):
        pass

    def warn(self, *args, **kwargs):
        pass


_MONITOR_SKILL_DIR = (
    Path(tools.__file__).parent.parent / "bundled_skills" / "monitor"
)


def _executor_with_monitor(tmp_path):
    """Build an executor with the bundled monitor skill's process tools loaded."""
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))
    te.register_skill_tools("monitor", load_skill_tools(_MONITOR_SKILL_DIR))
    return te


# ----- apply_edit ----------------------------------------------------------

# ----- gitignore matcher ---------------------------------------------------

@pytest.fixture
def repo(tmp_path):
    (tmp_path / ".gitignore").write_text("*.log\nbuild/\n/secret.txt\n")
    (tmp_path / "a.log").write_text("x")
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "secret.txt").write_text("x")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "out.o").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.log").write_text("x")
    (tmp_path / "sub" / "secret.txt").write_text("x")  # anchored rule → NOT ignored
    return tmp_path


def _assert_gitignore(m, repo):
    assert m.is_ignored((repo / "a.log").resolve(), False)
    assert not m.is_ignored((repo / "a.py").resolve(), False)
    assert m.is_ignored((repo / "secret.txt").resolve(), False)
    assert m.is_ignored((repo / "sub" / "b.log").resolve(), False)
    assert m.is_ignored((repo / "build").resolve(), True)
    assert m.is_ignored((repo / "build" / "out.o").resolve(), False)
    assert not m.is_ignored((repo / "sub" / "secret.txt").resolve(), False)


# ----- file / ls / glob / multi_edit tools --------------------------------


def test_relative_paths_resolve_against_work_dir(tmp_path, monkeypatch):
    process_cwd = tmp_path / "cwd"
    work_dir = tmp_path / "work"
    process_cwd.mkdir()
    work_dir.mkdir()
    monkeypatch.chdir(process_cwd)
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(work_dir))

    assert te._write_file("nested/a.txt", "one")["success"]
    assert (work_dir / "nested/a.txt").read_text() == "one"
    assert not (process_cwd / "nested/a.txt").exists()
    assert te._edit_file("nested/a.txt", "one", "two")["success"]
    assert te._multi_edit(
        "nested/a.txt", [{"old_text": "two", "new_text": "three"}]
    )["success"]
    assert te._read_file("nested/a.txt")["content"] == "three"
    assert "a.txt" in te._ls("nested")["content"]
    assert te._glob_files("*.txt", base_dir="nested")["count"] == 1
    assert te._grep_files("three", path="nested")["count"] == 1
    # exec_command runs via PowerShell on Windows, bash elsewhere.
    pwd = "(pwd).Path" if os.name == "nt" else "pwd"
    assert te._exec_command(pwd, cwd="nested")["stdout"].strip() == str(
        work_dir / "nested"
    )
    if os.name == "posix":  # Unix-shell safety patterns are POSIX-only
        guard = SafetyGuard(work_dir="/tmp/job")
        safe, _ = guard.check(
            "exec_command", {"command": "rm -rf .", "cwd": "../../etc"}
        )
        assert not safe
        safe, _ = guard.check(
            "exec_command", {"command": "chmod -R 000 .", "cwd": "~"}
        )
        assert not safe
    assert te._remove_file("nested/a.txt")["success"]
    assert not (work_dir / "nested/a.txt").exists()


def test_web_fetch_rejects_non_public_destinations():
    te = ToolExecutor(console=_SilentConsole())
    for url in (
        "file:///etc/passwd",
        "http://127.0.0.1/",
        "http://2130706433/",
        "http://[::1]/",
        "http://169.254.169.254/latest/meta-data/",
    ):
        result = te._web_fetch(url)
        assert not result["success"], url


def test_read_image_returns_data_url(tmp_path):
    data = b"\x89PNG\r\n\x1a\n" + b"test"
    image = tmp_path / "image.png"
    image.write_bytes(data)
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))

    result = te.execute("read_image", {"file": "image.png"})

    assert result["success"]
    assert result["mime_type"] == "image/png"
    assert result["image_url"] == (
        "data:image/png;base64," + base64.b64encode(data).decode("ascii")
    )


def test_multi_edit_atomic_failure(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("one\ntwo\n")
    te = ToolExecutor(work_dir=str(tmp_path))
    res = te._multi_edit(str(f), edits=[
        {"old_text": "one", "new_text": "1"},
        {"old_text": "MISSING", "new_text": "x"},
    ])
    assert not res["success"] and f.read_text() == "one\ntwo\n"


def test_failed_exec_format_keeps_stdout_and_stderr():
    text = format_tool_result({
        "success": False,
        "stdout": "large stdout",
        "stderr": "brief stderr",
        "exit_code": 1,
    })
    assert "large stdout" in text and "brief stderr" in text


def test_exec_command_captures_full_output_artifact(tmp_path):
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))
    res = te._exec_command(
        "python -c \"print('HEAD'); print('x' * 30000); print('TAIL')\""
    )
    artifact = Path(res["_artifact_path"])
    try:
        captured = artifact.read_text()
        assert "HEAD" in captured and "TAIL" in captured
        assert len(captured) > len(res["stdout"])
        assert res["original_output_chars"] > 24_000
    finally:
        artifact.unlink(missing_ok=True)


def test_managed_background_process_lifecycle(tmp_path):
    te = _executor_with_monitor(tmp_path)
    started = te.execute(
        "start_process",
        {"command": "python -u -c \"import time; print('ready'); time.sleep(30)\""},
    )
    process_id = started["process_id"]
    log_path = tmp_path / started["log_path"]
    try:
        assert started["status"] == "running"
        assert log_path.is_file()
        for _ in range(100):
            if "ready" in log_path.read_text():
                break
            time.sleep(0.05)
        assert "ready" in log_path.read_text()

        inspected = te.execute("inspect_processes", {"process_id": process_id})
        assert inspected["success"]
        assert inspected["processes"][0]["status"] == "running"

        stopped = te.execute("stop_process", {"process_id": process_id})
        assert stopped["success"]
        assert stopped["status"] == "exited"
        assert stopped["exit_code"] is not None
    finally:
        te.shutdown_processes()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux subreaper semantics")
def test_managed_background_process_stops_detached_descendant(tmp_path):
    te = _executor_with_monitor(tmp_path)
    child_pid_file = tmp_path / "detached.pid"
    child_code = (
        "import os,pathlib,time; os.setsid(); "
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(os.getpid())); "
        "time.sleep(30)"
    )
    launcher = f"import subprocess,sys; subprocess.Popen([sys.executable, '-c', {child_code!r}])"
    started = te.execute("start_process", {"command": f"python -c {shlex.quote(launcher)}"})
    for _ in range(100):
        if child_pid_file.exists():
            break
        time.sleep(0.05)
    assert child_pid_file.exists()

    te.execute("stop_process", {"process_id": started["process_id"]})

    child_pid = int(child_pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)

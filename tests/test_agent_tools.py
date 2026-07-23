"""Tests for core file, glob, grep, and process tools."""

import base64
import os
import shlex
import shutil
import sys
import threading
import time
from pathlib import Path

import pytest

import kiui.agent.tools as tools
import kiui.agent.tools.commands as command_tools
import kiui.agent.tools.search as search_tools
from kiui.agent.skills import load_skill_tools
from kiui.agent.tools import (
    ToolExecutor,
    _human_size,
    apply_edit,
    find_match,
    format_tool_result,
)
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

@pytest.fixture
def glob_tree(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / ".hidden.py").write_text("")
    (tmp_path / "note.txt").write_text("")
    (tmp_path / "!important").write_text("")
    (tmp_path / "ignored.py").write_text("")
    (tmp_path / ".gitignore").write_text("ignored.py\nignored/\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "b.py").write_text("")
    (tmp_path / "src" / "nested-ignored.py").write_text("")
    (tmp_path / "src" / ".gitignore").write_text("nested-ignored.py\n")
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "c.py").write_text("")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "d.py").write_text("")
    return tmp_path


def _assert_glob_semantics(te):
    result = te._glob_files("*.py")
    assert result["success"]
    assert result["matches"] == [".hidden.py", "a.py", "src/b.py"]
    assert te._glob_files("src/**/*.py")["matches"] == ["src/b.py"]
    assert te._glob_files("!important")["matches"] == ["!important"]
    assert te._glob_files("*.py", recursive=False)["matches"] == [".hidden.py", "a.py"]
    assert "ignored.py" in te._glob_files("*.py", include_ignored=True)["matches"]
    assert "ignored/c.py" in te._glob_files("*.py", include_ignored=True)["matches"]
    assert "node_modules/d.py" not in te._glob_files("*.py", include_ignored=True)["matches"]
    assert not te._glob_files("**/*.py", recursive=False)["success"]


# ----- file / ls / glob / multi_edit tools --------------------------------


def test_glob_ripgrep_semantics(glob_tree):
    if not shutil.which("rg"):
        pytest.skip("ripgrep is not installed")
    _assert_glob_semantics(ToolExecutor(console=_SilentConsole(), work_dir=str(glob_tree)))


def test_search_tools_require_ripgrep(glob_tree, monkeypatch):
    monkeypatch.setattr(search_tools.shutil, "which", lambda _: None)
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(glob_tree))
    assert "requires ripgrep" in te._glob_files("*.py")["error"]
    assert "requires ripgrep" in te._grep_files("x")["error"]


def test_ls_filters_locally_without_recursive_scan(glob_tree):
    result = ToolExecutor(console=_SilentConsole(), work_dir=str(glob_tree))._ls()
    assert result["success"]
    assert "src/" in result["content"]
    assert "ignored.py" not in result["content"]
    assert "ignored/" not in result["content"]
    assert "node_modules/" not in result["content"]


def test_glob_truncates_without_returning_extra_match(tmp_path, monkeypatch):
    for index in range(4):
        (tmp_path / f"{index}.txt").write_text("")
    monkeypatch.setattr(search_tools, "MAX_GLOB_RESULTS", 3)
    result = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))._glob_files("*.txt")
    assert result["success"] and result["truncated"]
    assert len(result["matches"]) == 3


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

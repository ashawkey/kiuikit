"""Tests for core file, glob, grep, and process tools."""

import base64
import os
import shlex
import shutil
import sys
import threading
import time
import tracemalloc
from pathlib import Path

import pytest

import kiui.agent.tools as tools
import kiui.agent.tools.commands as command_tools
import kiui.agent.tools.search as search_tools
from kiui.agent.skills import load_skill_tools
from kiui.agent.bundled_skills.browser import tools as browser_tools
from kiui.agent.tools import (
    ToolExecutor,
    _human_size,
    apply_edit,
    find_match,
    format_tool_result,
    result_text_failed,
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


def test_native_tool_resource_cleanup(tmp_path):
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))
    closed = []

    te.register_tool_resource("browser", lambda: closed.append("first"))
    te.register_tool_resource("browser", lambda: closed.append("second"))
    assert closed == ["first"]

    te.shutdown_tool_resources(clear=True)
    assert closed == ["first", "second"]
    assert te._tool_resource_cleanups == {}


def test_browser_connection_reused_for_same_cdp_server(monkeypatch, tmp_path):
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))

    class Connection:
        endpoint = "ws://127.0.0.1:9222/devtools/browser/session-id"

    connection = Connection()
    te._browser_connection = connection
    monkeypatch.setattr(
        browser_tools,
        "_resolve_endpoint",
        lambda _: pytest.fail("same CDP server should reuse the existing connection"),
    )

    assert browser_tools._connection(te, "http://127.0.0.1:9222") is connection


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


def test_exec_format_appends_status_without_labeling_output():
    text = format_tool_result({
        "success": False,
        "stdout": "ordinary output\nerror output\n",
        "exit_code": 1,
    })
    assert text == (
        "ordinary output\nerror output\n"
        "[exit_code: 1, interrupted: false, timed_out: false]"
    )
    assert result_text_failed(text)


def test_exec_command_merges_output_and_is_noninteractive(tmp_path):
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))
    res = te._exec_command(
        "printf 'out\\n'; printf 'err\\n' >&2; "
        "python -c \"import sys; print('stdin=' + sys.stdin.read())\""
    )
    Path(res["_artifact_path"]).unlink(missing_ok=True)
    assert res["stdout"] == "out\nerr\nstdin=\n"
    assert "stderr" not in res


def test_exec_command_timeout_and_null_override(tmp_path):
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))
    timed_out = te._exec_command("sleep 1", timeout=0.05)
    Path(timed_out["_artifact_path"]).unlink(missing_ok=True)
    assert timed_out["timed_out"] and not timed_out["success"]
    assert "timed_out: true" in format_tool_result(timed_out)

    completed = te._exec_command("sleep 0.05; printf done", timeout=None)
    Path(completed["_artifact_path"]).unlink(missing_ok=True)
    assert completed["success"] and completed["stdout"] == "done"


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


def test_exec_command_output_is_not_parsed_as_markup(tmp_path):
    """Command output is data: rich markup in it must not be interpreted."""
    from kiui.agent.ui import AgentConsole

    console = AgentConsole()
    console._console.file = open(os.devnull, "w")
    try:
        te = ToolExecutor(console=console, work_dir=str(tmp_path))
        # "[/bad]" is a closing tag with no opening tag: parsed as markup it
        # raises MarkupError, and "[dim]" would silently vanish from display.
        res = te._exec_command("printf '%s\\n' one '[/bad] markup' '[dim]tag' two")
    finally:
        console._console.file.close()
    assert res["exit_code"] == 0
    assert res["stdout"] == "one\n[/bad] markup\n[dim]tag\ntwo\n"


def test_exec_command_display_is_batched(tmp_path, monkeypatch):
    """Output is rendered in batches, not once per line."""
    calls = []

    class _CountingConsole(_SilentConsole):
        def print(self, *args, **kwargs):
            calls.append(args[0] if args else "")

    te = ToolExecutor(console=_CountingConsole(), work_dir=str(tmp_path))
    res = te._exec_command("seq 1 5000")
    assert res["exit_code"] == 0
    # One print per line would be 5000 calls; batching keeps it far lower while
    # still emitting something for the user to watch.
    assert 0 < len(calls) < 200
    assert res["original_output_chars"] > 20_000


def test_grep_stops_reading_at_match_cap(tmp_path):
    """grep streams and terminates ripgrep at the cap instead of buffering all output."""
    if not shutil.which("rg"):
        pytest.skip("ripgrep is required")
    # Far more matches than the cap, so the search must stop early.
    big = tmp_path / "big.txt"
    big.write_text("needle\n" * 20_000, encoding="utf-8")

    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))
    res = te._grep_files("needle")
    assert res["success"] and res["truncated"]
    assert res["count"] <= search_tools.MAX_GREP_MATCHES
    assert res["truncation_reason"] in ("item cap", "character cap")


def test_read_file_reads_only_the_requested_window(tmp_path):
    """A bounded read must not materialize the whole file.

    Regression: readlines() loaded every line before slicing, so reading 20
    lines out of a large log cost memory proportional to the file.
    """
    path = tmp_path / "big.txt"
    with path.open("w", encoding="utf-8") as f:
        for i in range(1, 200_001):
            f.write(f"line{i}\n")

    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))

    tracemalloc.start()
    try:
        result = te._read_file("big.txt", offset=10, limit=5)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # The file is ~2.3 MB; a bounded read must stay far below holding it all.
    assert peak < 500_000, f"read_file allocated {peak} bytes for 5 lines"

    assert result["content"].startswith("line10\nline11\n")
    assert result["lines_read"] == 5
    assert result["truncated"] and result["truncation_reason"] == "line cap"
    # The notice still reports the true total, which requires counting the rest.
    assert "of 200000 lines shown" in result["content"]

    # Reading past the end yields nothing rather than failing.
    assert te._read_file("big.txt", offset=999_999)["content"] == ""
    # An exact-limit read is not falsely marked truncated.
    assert not te._read_file("big.txt", offset=199_996, limit=5)["truncated"]


def test_read_file_handles_missing_trailing_newline(tmp_path):
    path = tmp_path / "noeol.txt"
    path.write_text("a\nb\nc", encoding="utf-8")
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))
    assert te._read_file("noeol.txt")["content"] == "a\nb\nc"
    limited = te._read_file("noeol.txt", limit=2)
    assert limited["truncated"] and "of 3 lines shown" in limited["content"]


def _slow_silent_rg(tmp_path: Path) -> Path:
    """A stand-in ripgrep that runs long while emitting nothing on stdout."""
    fake = tmp_path / "bin"
    fake.mkdir()
    script = fake / "rg"
    script.write_text("#!/bin/sh\nsleep 30\n", encoding="utf-8")
    script.chmod(0o755)
    return fake


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell stub")
def test_grep_timeout_applies_while_ripgrep_is_silent(tmp_path, monkeypatch):
    """A search that matches nothing must still honour its deadline.

    Regression: the deadline used to be checked only when ripgrep emitted a
    line, so a long scan with no matches blocked until the process finished.
    """
    monkeypatch.setenv("PATH", str(_slow_silent_rg(tmp_path)) + os.pathsep + os.environ["PATH"])
    monkeypatch.setattr(search_tools, "GREP_TIMEOUT_SECONDS", 1)

    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))
    started = time.monotonic()
    res = te._grep_files("anything")
    assert not res["success"] and "timed out" in res["error"]
    assert time.monotonic() - started < 10


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell stub")
def test_grep_cancellation_applies_while_ripgrep_is_silent(tmp_path, monkeypatch):
    """Escape must abort a search that has produced no output yet."""
    monkeypatch.setenv("PATH", str(_slow_silent_rg(tmp_path)) + os.pathsep + os.environ["PATH"])
    monkeypatch.setattr(search_tools, "GREP_TIMEOUT_SECONDS", 30)

    cancellation = CancellationToken(EventHub())
    cancellation.watch_keyboard = False
    operation = cancellation.begin("grep")
    te = ToolExecutor(
        console=_SilentConsole(), work_dir=str(tmp_path), cancellation=cancellation
    )
    timer = threading.Timer(0.3, lambda: cancellation.cancel(operation))
    timer.start()
    try:
        started = time.monotonic()
        res = te._grep_files("anything")
    finally:
        timer.cancel()
    assert res["interrupted"] and not res["success"]
    assert time.monotonic() - started < 10


def test_search_honors_gitignore_outside_a_git_repository(tmp_path):
    """glob/grep must respect .gitignore even when the tree is not a git repo.

    ripgrep applies .gitignore only inside a repository unless told otherwise,
    which silently disagreed with `ls` and with both tools' documented contract.
    """
    if not shutil.which("rg"):
        pytest.skip("ripgrep is required")
    (tmp_path / "a.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "hidden.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("hidden.txt\n", encoding="utf-8")
    assert not (tmp_path / ".git").exists()

    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))
    assert te._glob_files("*.txt")["matches"] == ["a.txt"]
    assert [m["file"] for m in te._grep_files("needle")["matches"]] == ["a.txt"]
    # Opting in still reaches the ignored file.
    assert "hidden.txt" in te._glob_files("*.txt", include_ignored=True)["matches"]


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


class _RecordingConsole(_SilentConsole):
    def __init__(self):
        self.labels = []

    def tool(self, msg, *args, **kwargs):
        self.labels.append(msg)


@pytest.mark.parametrize(
    "name, args",
    [
        ("read_file", {"file": "a.txt"}),
        ("read_file", {"file": "a.txt", "offset": 5, "limit": 20}),
        ("write_file", {"file": "a.txt", "content": "hi\n"}),
        ("edit_file", {"file": "a.txt", "old_text": "hi", "new_text": "yo"}),
        ("multi_edit", {"file": "a.txt", "edits": [{"old_text": "hi", "new_text": "yo"}]}),
        ("remove_file", {"file": "a.txt"}),
        ("ls", {}),
        ("ls", {"path": "sub", "all": True}),
        ("glob_files", {"pattern": "*.txt"}),
        ("glob_files", {"pattern": "*.txt", "recursive": False}),
        ("grep_files", {"pattern": "needle"}),
        ("grep_files", {"pattern": "needle", "path": "sub", "file_glob": "*.py", "case_insensitive": True}),
        ("load_skill", {"name": "monitor"}),
    ],
)
def test_replayed_tool_calls_render_like_live_ones(tmp_path, name, args):
    """A replayed call must print the label the live handler prints."""
    (tmp_path / "a.txt").write_text("hi\n")
    (tmp_path / "sub").mkdir()
    console = _RecordingConsole()
    te = ToolExecutor(console=console, work_dir=str(tmp_path))
    te.execute(name, dict(args))

    assert console.labels == [tools.describe_tool_call(name, args)]


def test_unknown_and_malformed_calls_still_describe_compactly():
    describe = tools.describe_tool_call
    assert describe("browser_click", {"index": 3}) == "browser_click(index=3)"
    assert describe("browser_stop", {}) == "browser_stop"
    assert describe("edit_file", {"file": "a.txt"}) == "edit_file a.txt"
    # Missing required arguments fall back instead of failing the render.
    assert describe("edit_file", {"path": "a.txt"}) == "edit_file(path=a.txt)"
    long_value = describe("web_fetch", {"note": "x " * 60})
    assert long_value.endswith("…)") and len(long_value) < 120


def test_replay_only_calls_a_result_failed_when_it_is_formatted_as_one():
    failed = tools.result_text_failed
    assert failed(format_tool_result({"success": False, "error": "boom"}))
    assert failed(format_tool_result({"success": False, "error": "boom", "stdout": "out"}))
    assert not failed("grep hit: raise ValueError('error')")
    assert not failed("")

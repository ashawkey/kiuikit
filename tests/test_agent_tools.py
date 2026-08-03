"""Tests for core file, glob, grep, and process tools."""

import base64
import os
import shlex
import shutil
import sys
import threading
import time
import tracemalloc
from contextlib import contextmanager
from pathlib import Path

import pytest

import kiui.agent.tools as tools
import kiui.agent.tools.commands as command_tools
import kiui.agent.tools.search as search_tools
from kiui.agent.bundled_skills.browser import tools as browser_tools
from kiui.agent.tools import (
    ToolExecutor,
    _human_size,
    apply_edit,
    find_match,
    describe_tool_output,
    format_tool_result,
    result_text_failed,
)
from kiui.agent.utils.io import CancellationToken, EventHub


class _SilentConsole:
    def tool(self, *args, **kwargs):
        pass

    def print(self, *args, **kwargs):
        pass

    def warn(self, *args, **kwargs):
        pass

    @contextmanager
    def thinking(self, **kwargs):
        yield


def _executor_with_monitor(tmp_path):
    """Backward-compatible helper for an executor with core process tools."""
    return ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))


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
    def matches(pattern, **kwargs):
        # ripgrep returns OS-native separators (backslashes on Windows);
        # compare against canonical forward-slash paths.
        return [m.replace("\\", "/") for m in te._glob_files(pattern, **kwargs)["matches"]]

    assert matches("*.py") == [".hidden.py", "a.py", "src/b.py"]
    assert matches("src/**/*.py") == ["src/b.py"]
    assert matches("!important") == ["!important"]
    assert matches("*.py", recursive=False) == [".hidden.py", "a.py"]
    assert "ignored.py" in matches("*.py", include_ignored=True)
    assert "ignored/c.py" in matches("*.py", include_ignored=True)
    assert "node_modules/d.py" not in matches("*.py", include_ignored=True)
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


def test_read_image_rejects_oversized_file(tmp_path, monkeypatch):
    import kiui.agent.tools.files as files_mod

    monkeypatch.setattr(files_mod, "MAX_IMAGE_BYTES", 16)
    image = tmp_path / "big.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 64)
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))

    result = te.execute("read_image", {"file": "big.png"})

    assert not result["success"]
    assert "too large" in result["error"]


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
        "python -c \"import sys; print('out', flush=True); "
        "print('err', file=sys.stderr, flush=True); "
        "print('stdin=' + sys.stdin.read(), flush=True)\""
    )
    Path(res["_artifact_path"]).unlink(missing_ok=True)
    # Windows shells emit \r\n; normalize so the merged stream is comparable.
    assert res["stdout"].replace("\r\n", "\n") == "out\nerr\nstdin=\n"
    assert "stderr" not in res


def test_exec_command_timeout_and_null_override(tmp_path):
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))
    timed_out = te._exec_command("sleep 1", timeout=0.05)
    Path(timed_out["_artifact_path"]).unlink(missing_ok=True)
    assert timed_out["timed_out"] and not timed_out["success"]
    assert "timed_out: true" in format_tool_result(timed_out)

    completed = te._exec_command(
        "python -c \"import time; time.sleep(0.05); print('done', end='')\"",
        timeout=None,
    )
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
        res = te._exec_command(
            "python -c \"print('one\\n[/bad] markup\\n[dim]tag\\ntwo')\""
        )
    finally:
        console._console.file.close()
    assert res["exit_code"] == 0
    # Windows shells emit \r\n; normalize so the merged stream is comparable.
    assert res["stdout"].replace("\r\n", "\n") == "one\n[/bad] markup\n[dim]tag\ntwo\n"


def test_exec_command_display_is_batched(tmp_path, monkeypatch):
    """Output is rendered in batches, not once per line."""
    calls = []

    class _CountingConsole(_SilentConsole):
        def print(self, *args, **kwargs):
            calls.append(args[0] if args else "")

    te = ToolExecutor(console=_CountingConsole(), work_dir=str(tmp_path))
    res = te._exec_command(
        "python -c \"print('\\n'.join(str(i) for i in range(1, 5001)))\""
    )
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


def test_wait_tool_is_bounded_and_interruptible(tmp_path):
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))

    started = time.monotonic()
    result = te.execute("wait", {"seconds": 0.02})
    assert result == {"waited_seconds": 0.02, "success": True}
    assert time.monotonic() - started >= 0.015

    events = EventHub()
    cancellation = CancellationToken(events)
    cancellation.begin("test wait")
    te.cancellation = cancellation
    timer = threading.Timer(0.02, cancellation.cancel)
    timer.start()
    try:
        started = time.monotonic()
        result = te.execute("wait", {"seconds": 10})
    finally:
        timer.cancel()
    assert result["interrupted"] and not result["success"]
    assert time.monotonic() - started < 1


@pytest.mark.parametrize("seconds", [0, -1, float("inf"), "1", True])
def test_wait_tool_rejects_invalid_duration(tmp_path, seconds):
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))
    result = te.execute("wait", {"seconds": seconds})
    assert not result["success"]


def test_inspect_processes_no_longer_accepts_wait(tmp_path):
    te = _executor_with_monitor(tmp_path)
    schema = te.registry.get("inspect_processes").schema
    assert "wait" not in schema["function"]["parameters"]["properties"]

    result = te.execute("inspect_processes", {"wait": 0.01})
    assert not result["success"]
    assert "unexpected keyword argument 'wait'" in result["error"]


def test_process_status_notifications_are_delivered_in_order(tmp_path):
    te = _executor_with_monitor(tmp_path)
    counts = [(1, 0)]
    updates = []
    entered = threading.Event()
    release = threading.Event()
    te.process_counts = lambda: counts[0]

    def listener(running, finished):
        if (running, finished) == (1, 0):
            entered.set()
            release.wait(timeout=2)
        updates.append((running, finished))

    te.add_process_listener(listener, notify=False)
    first = threading.Thread(target=te._notify_process_status)
    first.start()
    assert entered.wait(timeout=2)

    counts[0] = (0, 1)
    second = threading.Thread(target=te._notify_process_status)
    second.start()
    time.sleep(0.05)
    assert updates == []

    release.set()
    first.join(timeout=2)
    second.join(timeout=2)
    assert updates == [(1, 0), (0, 1)]


def test_process_status_listener_tracks_start_and_finish(tmp_path):
    te = _executor_with_monitor(tmp_path)
    updates = []
    te.add_process_listener(lambda running, finished: updates.append((running, finished)))
    started = te.execute(
        "start_process",
        {"command": "python -c 'print(123)'"},
    )
    try:
        for _ in range(100):
            if updates and updates[-1] == (0, 1):
                break
            time.sleep(0.05)
        assert (1, 0) in updates
        assert updates[-1] == (0, 1)
        assert te.process_counts() == (0, 1)
        assert started["process_id"]
    finally:
        te.shutdown_processes()


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
        by_pid = te.inspect_processes(process_id=str(started["pid"]))
        assert by_pid["processes"][0]["process_id"] == process_id

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
        ("glob_files", {"pattern": "*.txt", "base_dir": "sub", "include_ignored": True}),
        ("grep_files", {"pattern": "needle"}),
        ("grep_files", {"pattern": "needle", "path": "sub", "file_glob": "*.py", "case_insensitive": True}),
        ("load_skill", {"name": "monitor"}),
        ("wait", {"seconds": 0.001}),
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
    assert describe("browser_click", {"index": 3}) == "browser_click · index=3"
    assert describe("browser_stop", {}) == "browser_stop"
    assert describe("edit_file", {"file": "a.txt"}) == "edit_file a.txt"
    # Missing required arguments fall back instead of failing the render.
    assert describe("edit_file", {"path": "a.txt"}) == "edit_file · path=a.txt"
    long_value = describe("web_fetch", {"note": "x " * 60})
    assert long_value.endswith("…") and len(long_value) < 120


def test_tool_call_descriptions_use_one_visual_grammar():
    describe = tools.describe_tool_call
    assert describe("grep_files", {
        "pattern": "needle", "path": "src", "file_glob": "*.py", "case_insensitive": True,
    }) == 'grep_files "needle" · in src · glob *.py · ignore case'
    assert describe("glob_files", {
        "pattern": "**/*.py", "base_dir": "src", "recursive": False, "include_ignored": True,
    }) == 'glob_files "**/*.py" · in src · non-recursive · include ignored'
    assert describe("read_file", {"file": "a.py", "offset": 5, "limit": 20}) == "read_file a.py · lines 5–24"
    assert describe("multi_edit", {"file": "a.py", "edits": [{}, {}]}) == "multi_edit a.py · 2 edits"
    assert describe("write_file", {"file": "a.py", "content": "secret"}) == "write_file a.py"


def test_exec_command_describe_shows_full_command():
    """exec_command must show the complete shell command, never truncate it."""
    import json

    from kiui.agent.tools.formatting import build_tool_call_description

    describe = tools.describe_tool_call
    command = "python -m pytest tests -q --tb=short " + "x" * 200  # > 60-char cap
    desc = build_tool_call_description(
        "exec_command", {"command": command, "cwd": "src"}
    )
    assert json.loads(desc.primary) == command
    assert "…" not in desc.primary
    assert "cwd src" in desc.qualifiers

    # Other tools keep the compact label so the terminal stays readable.
    long_query = "q " * 50
    assert "…" in describe("web_search", {"query": long_query})


def test_exec_command_streams_full_output(tmp_path):
    """Every line of streamed output reaches the console; nothing is dropped."""
    printed = []

    class _CountingConsole(_SilentConsole):
        def print(self, *args, **kwargs):
            printed.append(args[0] if args else "")

    te = ToolExecutor(console=_CountingConsole(), work_dir=str(tmp_path))
    res = te._exec_command(
        "python -c \"print('\\n'.join('L%d' % i for i in range(300)))\""
    )
    assert res["exit_code"] == 0
    shown = "\n".join(printed)
    assert "not shown" not in shown
    assert len(printed) >= 2  # batched in print calls, not dropped
    for i in range(300):
        assert f"L{i}" in shown


def test_executor_logs_skill_tool_once_and_redacts_text_arguments(tmp_path):
    console = _RecordingConsole()
    te = ToolExecutor(console=console, work_dir=str(tmp_path))
    te.register_skill_tools("demo", [{
        "run": lambda executor, text: {"success": True},
        "schema": {
            "type": "function",
            "function": {
                "name": "demo_type",
                "description": "Type text.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        },
    }])

    assert te.execute("demo_type", {"text": "top secret"})["success"]
    assert console.labels == ["demo_type · text=<10 chars>"]


def test_skill_owned_tool_description_is_used(tmp_path):
    console = _RecordingConsole()
    te = ToolExecutor(console=console, work_dir=str(tmp_path))
    te.register_skill_tools("demo", [{
        "run": lambda executor, process_id, wait=0: {"success": True},
        "describe": lambda args: tools.ToolCallDescription(
            "inspect_demo", args["process_id"], (f"wait {args['wait']}s",),
        ),
        "schema": {
            "type": "function",
            "function": {
                "name": "inspect_demo",
                "description": "Inspect a demo process.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "process_id": {"type": "string"},
                        "wait": {"type": "number"},
                    },
                    "required": ["process_id"],
                },
            },
        },
    }])

    assert te.execute("inspect_demo", {"process_id": "p-1", "wait": 5})["success"]
    assert console.labels == ["inspect_demo p-1 · wait 5s"]


def test_skill_owned_tool_output_description_is_used(tmp_path):
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))
    te.register_skill_tools("demo", [{
        "run": lambda executor, value: {"value": value, "success": True},
        "describe_output": lambda result: f"Returned {result['value']}",
        "schema": {
            "type": "function",
            "function": {
                "name": "demo_output",
                "description": "Return a value.",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            },
        },
    }])

    result = te.execute("demo_output", {"value": "useful"})
    spec = te.registry.get("demo_output")
    assert describe_tool_output("demo_output", result, spec.describe_output) == "Returned useful"


def test_builtin_process_output_description_is_informative():
    result = {
        "processes": [{
            "process_id": "p-1234",
            "pid": 42,
            "status": "running",
            "exit_code": None,
            "command": "python worker.py",
            "log_path": ".kia/processes/p-1234.log",
            "log_tail": "hello\n",
            "log_tail_truncated": False,
        }],
        "count": 1,
        "success": True,
    }

    assert describe_tool_output("inspect_processes", result) == (
        "p-1234 · running · pid 42 · python worker.py\n"
        "log tail: 6 chars · .kia/processes/p-1234.log\n"
        "hello"
    )


def test_failed_tool_output_uses_error_instead_of_success_describer():
    result = {"error": "boom", "success": False}
    assert describe_tool_output("inspect_processes", result) == "Error: boom"


def test_monitor_skill_owns_its_tool_descriptions(tmp_path):
    console = _RecordingConsole()
    te = _executor_with_monitor(tmp_path)
    te.console = console

    te.execute("inspect_processes", {"process_id": "p-1", "log_tail_chars": 1000})
    te.execute("stop_process", {"process_id": "p-1"})

    assert console.labels == [
        "inspect_processes p-1 · tail 1,000 chars",
        "stop_process p-1",
    ]


def test_browser_skill_owns_sensitive_tool_descriptions(tmp_path):
    console = _RecordingConsole()
    te = ToolExecutor(console=console, work_dir=str(tmp_path))
    te.register_skill_tools("browser", browser_tools.TOOLS)

    # Execution may fail without an attached browser, but logging happens first.
    te.execute("browser_type", {"index": 3, "text": "top secret", "clear": False})
    te.execute("browser_scroll", {"direction": "down", "pages": 2})

    assert console.labels == [
        "browser_type #3 · 10 chars · keep existing text",
        "browser_scroll page · down 2 pages",
    ]
    assert "top secret" not in " ".join(console.labels)


def test_replay_only_calls_a_result_failed_when_it_is_formatted_as_one():
    failed = tools.result_text_failed
    assert failed(format_tool_result({"success": False, "error": "boom"}))
    assert failed(format_tool_result({"success": False, "error": "boom", "stdout": "out"}))
    assert not failed("grep hit: raise ValueError('error')")
    assert not failed("")

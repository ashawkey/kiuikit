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
import kiui.agent.tools.processes as process_tools
from kiui.agent.tools import (
    ToolExecutor,
    _human_size,
    apply_edit,
    find_match,
    format_tool_result,
    get_tool_definitions,
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


# ----- apply_edit ----------------------------------------------------------

def test_apply_edit_exact_single():
    new, count, ln, err = apply_edit("a\nb\nc\n", "b", "B", False)
    assert err is None and new == "a\nB\nc\n" and count == 1 and ln == 2


def test_apply_edit_not_found():
    new, count, ln, err = apply_edit("a\nb\n", "zzz", "x", False)
    assert err is not None and count == 0


def test_apply_edit_ambiguous_refused():
    new, count, ln, err = apply_edit("x\nx\nx\n", "x", "y", False)
    assert err is not None and count == 3


def test_apply_edit_replace_all():
    new, count, ln, err = apply_edit("x\nx\nx\n", "x", "y", True)
    assert err is None and new == "y\ny\ny\n" and count == 3


def test_apply_edit_tolerant_trailing_ws():
    src = "def foo():   \n    return 1\n"
    new, count, ln, err = apply_edit(src, "def foo():\n    return 1", "def foo():\n    return 2", False)
    assert err is None and "return 2" in new and count == 1


def test_apply_edit_crlf_normalized():
    src = "line1\r\nline2\r\nline3\r\n"
    new, count, ln, err = apply_edit(src, "line2", "LINE2", False)
    assert err is None and new == "line1\nLINE2\nline3\n"


def test_find_match_spans():
    spans = find_match("a b a", "a")
    assert len(spans) == 2
    assert find_match("no match here", "zzz") == []


def test_human_size():
    assert _human_size(0) == "0B"
    assert _human_size(2048) == "2.0K"


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


def test_gitignore_matcher(repo):
    m = build_gitignore_matcher(repo)
    _assert_gitignore(m, repo)


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


def test_read_image_rejects_unsupported_file(tmp_path):
    (tmp_path / "not-image.txt").write_text("hello")
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))

    result = te.execute("read_image", {"file": "not-image.txt"})

    assert not result["success"]
    assert "Unsupported image format" in result["error"]


def test_read_image_tool_requires_multimodal_capability():
    default_names = {t["function"]["name"] for t in get_tool_definitions()}
    multimodal_names = {
        t["function"]["name"]
        for t in get_tool_definitions(supports_image_input=True)
    }

    assert "read_image" not in default_names
    assert "read_image" in multimodal_names
    assert "read_image" not in {
        t["function"]["name"]
        for t in get_tool_definitions(
            supports_image_input=True, allowed={"read_file"}
        )
    }


def test_read_file_logs_offset_and_limit(tmp_path):
    class Console:
        def __init__(self):
            self.messages = []

        def tool(self, message):
            self.messages.append(message)

    f = tmp_path / "m.py"
    f.write_text("one\ntwo\nthree\n")
    console = Console()
    te = ToolExecutor(console=console, work_dir=str(tmp_path))

    res = te._read_file(str(f), offset=2, limit=1)

    assert res["success"]
    assert console.messages == [f"read_file {f} (offset=2, limit=1)"]


def test_ls_respects_gitignore(repo):
    te = ToolExecutor(work_dir=str(repo))
    res = te._ls(str(repo))
    assert res["success"]
    assert "a.py" in res["content"]
    assert "a.log" not in res["content"] and "build/" not in res["content"]

    res_all = te._ls(str(repo), all=True)
    assert "a.log" in res_all["content"] and "build/" in res_all["content"]


def test_glob_gitignore_aware(repo):
    te = ToolExecutor(work_dir=str(repo))
    assert te._glob_files("**/*.log", base_dir=str(repo))["count"] == 0
    assert te._glob_files("**/*.log", base_dir=str(repo), include_ignored=True)["count"] == 2
    assert te._glob_files("**/*.py", base_dir=str(repo))["count"] == 1


def test_grep_python_gitignore_aware(repo, monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: None)  # force python fallback
    te = ToolExecutor(work_dir=str(repo))
    res = te._grep_files("x", path=str(repo))
    files = {m["file"] for m in res["matches"]}
    assert not any(f.endswith(".log") for f in files)
    assert not any("build" in f for f in files)


def test_glob_truncated_flag_at_cap_boundary(tmp_path, monkeypatch):
    import kiui.agent.tools.search as search
    monkeypatch.setattr(search, "MAX_GLOB_RESULTS", 3)
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))

    for i in range(3):
        (tmp_path / f"f{i}.txt").write_text("x")
    # Exactly at the cap with no further matches: must not be flagged truncated.
    res = te._glob_files("*.txt")
    assert res["count"] == 3 and res["truncated"] is False

    # One more match than the cap: truncated and capped at MAX_GLOB_RESULTS.
    (tmp_path / "f3.txt").write_text("x")
    res = te._glob_files("*.txt")
    assert res["count"] == 3 and res["truncated"] is True


@pytest.mark.parametrize("use_python", [False, True])
def test_grep_truncated_flag_at_cap_boundary(tmp_path, monkeypatch, use_python):
    import kiui.agent.tools.search as search
    if use_python:
        import shutil
        monkeypatch.setattr(shutil, "which", lambda name: None)  # force python fallback
    monkeypatch.setattr(search, "MAX_GREP_MATCHES", 3)
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))

    (tmp_path / "a.txt").write_text("hit\nhit\nhit\n")
    # Exactly at the cap with no further matches: must not be flagged truncated.
    res = te._grep_files("hit")
    assert res["count"] == 3 and res["truncated"] is False

    # One more match than the cap: truncated and capped at MAX_GREP_MATCHES.
    (tmp_path / "b.txt").write_text("hit\n")
    res = te._grep_files("hit")
    assert res["count"] == 3 and res["truncated"] is True


def test_multi_edit_success(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("alpha\nbeta\ngamma\n")
    te = ToolExecutor(work_dir=str(tmp_path))
    res = te._multi_edit(str(f), edits=[
        {"old_text": "alpha", "new_text": "A"},
        {"old_text": "gamma", "new_text": "G"},
    ])
    assert res["success"] and f.read_text() == "A\nbeta\nG\n"


def test_multi_edit_atomic_failure(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("one\ntwo\n")
    te = ToolExecutor(work_dir=str(tmp_path))
    res = te._multi_edit(str(f), edits=[
        {"old_text": "one", "new_text": "1"},
        {"old_text": "MISSING", "new_text": "x"},
    ])
    assert not res["success"] and f.read_text() == "one\ntwo\n"


def test_multi_edit_sequential(tmp_path):
    """Second edit sees the result of the first."""
    f = tmp_path / "m.py"
    f.write_text("foo\n")
    te = ToolExecutor(work_dir=str(tmp_path))
    res = te._multi_edit(str(f), edits=[
        {"old_text": "foo", "new_text": "bar"},
        {"old_text": "bar", "new_text": "baz"},
    ])
    assert res["success"] and f.read_text() == "baz\n"


def test_multi_edit_rejects_non_dict_edit(tmp_path):
    """A malformed edit (not an object) fails loudly with a clear message
    instead of raising an opaque AttributeError, and leaves the file untouched."""
    f = tmp_path / "m.py"
    f.write_text("keep me\n")
    te = ToolExecutor(work_dir=str(tmp_path))
    res = te._multi_edit(str(f), edits=["not-a-dict"])
    assert not res["success"]
    assert "must be an object" in res["error"]
    assert f.read_text() == "keep me\n"


def test_failed_exec_format_keeps_stdout_and_stderr():
    text = format_tool_result({
        "success": False,
        "stdout": "large stdout",
        "stderr": "brief stderr",
        "exit_code": 1,
    })
    assert "large stdout" in text and "brief stderr" in text


def test_exec_command_streams_carriage_return_updates(tmp_path):
    class RecordingConsole(_SilentConsole):
        def __init__(self):
            self.printed = threading.Event()
            self.output = []

        def print(self, text, **kwargs):
            self.output.append(text)
            self.printed.set()

    console = RecordingConsole()
    te = ToolExecutor(console=console, work_dir=str(tmp_path))
    result = {}
    worker = threading.Thread(
        target=lambda: result.update(te._exec_command(
            "python -c \"import sys,time; sys.stderr.write('progress\\r'); "
            "sys.stderr.flush(); time.sleep(1.5); sys.stderr.write('done\\n')\""
        ))
    )
    worker.start()
    try:
        # Generous timeout: PowerShell + Python cold start can exceed 1s.
        assert console.printed.wait(timeout=10)
        assert worker.is_alive()
        assert any("[stderr] progress" in line for line in console.output)
        worker.join(timeout=3)
        assert not worker.is_alive()
    finally:
        if artifact := result.get("_artifact_path"):
            Path(artifact).unlink(missing_ok=True)


def test_exec_command_truncation_reserves_space_for_stderr(tmp_path):
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))
    res = te._exec_command(
        "python -c \"import sys; print('o' * 30000); print('s' * 1000, file=sys.stderr)\""
    )
    artifact = Path(res["_artifact_path"])
    try:
        assert len(res["stdout"]) + len(res["stderr"]) <= tools.MAX_EXEC_OUTPUT_CHARS
        assert "s" * 1000 in res["stderr"]
        assert res["stdout"]
    finally:
        artifact.unlink(missing_ok=True)


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


def test_exec_artifact_limit_uses_utf8_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(command_tools, "MAX_EXEC_ARTIFACT_BYTES", 11)
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))
    res = te._exec_command("python -c \"print('é' * 20)\"")
    artifact = Path(res["_artifact_path"])
    try:
        assert artifact.stat().st_size <= 11
        assert res["artifact_truncated"] is True
        # The child emits \r\n on Windows (text-mode pipe), \n elsewhere.
        assert res["original_output_chars"] == 20 + (2 if os.name == "nt" else 1)
    finally:
        artifact.unlink(missing_ok=True)


def test_exec_artifact_creation_failure_does_not_start_process(tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(
        command_tools.tempfile,
        "NamedTemporaryFile",
        lambda **kwargs: (_ for _ in ()).throw(OSError("no temp space")),
    )
    monkeypatch.setattr(command_tools.subprocess, "Popen", lambda *a, **k: started.append(True))
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))
    with pytest.raises(OSError, match="no temp space"):
        te._exec_command("echo unreachable")
    assert started == []


def test_exec_artifact_write_failure_keeps_draining(tmp_path, monkeypatch):
    original = command_tools.tempfile.NamedTemporaryFile

    class FailingWriter:
        def __init__(self, wrapped):
            self._wrapped = wrapped
            self.name = wrapped.name

        def write(self, text):
            raise OSError("disk full")

        def flush(self):
            return self._wrapped.flush()

        def close(self):
            return self._wrapped.close()

    monkeypatch.setattr(
        command_tools.tempfile,
        "NamedTemporaryFile",
        lambda **kwargs: FailingWriter(original(**kwargs)),
    )
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))
    started = time.monotonic()
    res = te._exec_command("python -c \"print('x' * 200000)\"")
    artifact = Path(res["_artifact_path"])
    try:
        assert time.monotonic() - started < 10
        assert res["artifact_truncated"] is True
        assert "disk full" in res["artifact_capture_error"]
        assert res["exit_code"] == 0
    finally:
        artifact.unlink(missing_ok=True)


def test_managed_background_process_lifecycle(tmp_path):
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))
    started = te._start_process(
        "python -u -c \"import time; print('ready'); time.sleep(30)\""
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

        inspected = te._inspect_processes(process_id)
        assert inspected["success"]
        assert inspected["processes"][0]["status"] == "running"

        stopped = te._stop_process(process_id)
        assert stopped["success"]
        assert stopped["status"] == "exited"
        assert stopped["exit_code"] is not None
    finally:
        te.shutdown_processes()


def test_managed_background_process_caps_log(tmp_path, monkeypatch):
    monkeypatch.setattr(process_tools, "MAX_PROCESS_LOG_BYTES", 10)
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))
    started = te._start_process("python -c \"print('x' * 1000)\"")
    record = te._processes[started["process_id"]]
    record["process"].wait(timeout=5)
    record["capture_thread"].join(timeout=5)
    inspected = te._inspect_processes(started["process_id"])["processes"][0]
    assert inspected["log_truncated"] is True
    assert (tmp_path / inspected["log_path"]).stat().st_size == 10


def test_managed_background_process_reports_normal_exit(tmp_path):
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))
    started = te._start_process("python -c \"print('done')\"")
    for _ in range(100):
        process = te._inspect_processes(started["process_id"])["processes"][0]
        if process["status"] == "exited":
            break
        time.sleep(0.05)
    assert process["status"] == "exited"
    assert process["exit_code"] == 0
    assert (tmp_path / process["log_path"]).read_text().strip() == "done"


def test_inspect_process_waits_before_reporting(tmp_path):
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))
    started = te._start_process("python -c \"import time; time.sleep(0.1)\"")
    before = time.monotonic()
    inspected = te._inspect_processes(started["process_id"], wait=0.3)

    assert time.monotonic() - before >= 0.25
    assert inspected["processes"][0]["status"] == "exited"


def test_inspect_process_rejects_negative_wait(tmp_path):
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))
    result = te._inspect_processes(wait=-1)

    assert not result["success"]
    assert "non-negative" in result["error"]


def test_inspect_process_wait_schema():
    inspect = next(
        tool for tool in get_tool_definitions()
        if tool["function"]["name"] == "inspect_processes"
    )
    wait = inspect["function"]["parameters"]["properties"]["wait"]

    assert wait["type"] == "number"
    assert wait["minimum"] == 0
    assert wait["default"] == 0
    log_tail = inspect["function"]["parameters"]["properties"]["log_tail_chars"]
    assert log_tail["type"] == "integer"
    assert log_tail["minimum"] == 0
    assert log_tail["maximum"] == process_tools.MAX_PROCESS_LOG_TAIL_CHARS
    assert log_tail["default"] == 0


def test_inspect_process_returns_bounded_log_tail(tmp_path):
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))
    started = te._start_process("python -c \"print('0123456789')\"")
    record = te._processes[started["process_id"]]
    record["process"].wait(timeout=5)
    record["capture_thread"].join(timeout=5)

    process = te._inspect_processes(
        started["process_id"], log_tail_chars=5
    )["processes"][0]

    assert process["log_tail"] == "6789\n"
    assert process["log_tail_truncated"] is True

    process = te._inspect_processes(
        started["process_id"], log_tail_chars=20
    )["processes"][0]
    assert process["log_tail"] == "0123456789\n"
    assert process["log_tail_truncated"] is False


def test_inspect_process_log_tail_handles_utf8_and_empty_logs(tmp_path):
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))
    printed = te._start_process("python -c \"print('αβγ')\"")
    printed_record = te._processes[printed["process_id"]]
    printed_record["process"].wait(timeout=5)
    printed_record["capture_thread"].join(timeout=5)

    process = te._inspect_processes(
        printed["process_id"], log_tail_chars=3
    )["processes"][0]
    assert process["log_tail"] == "βγ\n"
    assert process["log_tail_truncated"] is True

    quiet = te._start_process("python -c \"import time; time.sleep(30)\"")
    try:
        process = te._inspect_processes(
            quiet["process_id"], log_tail_chars=10
        )["processes"][0]
        assert process["log_tail"] == ""
        assert process["log_tail_truncated"] is False
    finally:
        te.shutdown_processes()


def test_inspect_process_log_tail_validation(tmp_path):
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))

    assert not te._inspect_processes(log_tail_chars=1)["success"]
    assert not te._inspect_processes(log_tail_chars=-1)["success"]
    assert not te._inspect_processes(
        log_tail_chars=process_tools.MAX_PROCESS_LOG_TAIL_CHARS + 1
    )["success"]


def test_inspect_process_wait_can_be_interrupted_without_stopping_process(tmp_path):
    cancellation = CancellationToken(EventHub())
    operation_id = cancellation.begin("test")
    te = ToolExecutor(
        console=_SilentConsole(), work_dir=str(tmp_path), cancellation=cancellation
    )
    started = te._start_process("python -c \"import time; time.sleep(30)\"")
    timer = threading.Timer(0.1, cancellation.cancel)
    timer.start()
    try:
        before = time.monotonic()
        inspected = te._inspect_processes(started["process_id"], wait=30)
        assert time.monotonic() - before < 2
        assert inspected["interrupted"] is True
        assert te._process_info(te._processes[started["process_id"]])["status"] == "running"
    finally:
        timer.cancel()
        cancellation.finish(operation_id)
        te.shutdown_processes()


@pytest.mark.skipif(os.name == "nt", reason="POSIX process semantics")
def test_managed_background_process_stops_child_after_shell_exits(tmp_path):
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))
    child_pid_file = tmp_path / "child.pid"
    child_code = (
        "import os,pathlib,time; "
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(os.getpid())); "
        "time.sleep(30)"
    )
    launcher = f"import subprocess,sys; subprocess.Popen([sys.executable, '-c', {child_code!r}])"
    command = f"python -c {shlex.quote(launcher)}"
    started = te._start_process(command)
    for _ in range(100):
        if child_pid_file.exists() and te._processes[started["process_id"]]["process"].poll() is not None:
            break
        time.sleep(0.05)
    assert child_pid_file.exists()
    te._stop_process(started["process_id"])
    child_pid = int(child_pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux subreaper semantics")
def test_managed_background_process_stops_detached_descendant(tmp_path):
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))
    child_pid_file = tmp_path / "detached.pid"
    child_code = (
        "import os,pathlib,time; os.setsid(); "
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(os.getpid())); "
        "time.sleep(30)"
    )
    launcher = f"import subprocess,sys; subprocess.Popen([sys.executable, '-c', {child_code!r}])"
    started = te._start_process(f"python -c {shlex.quote(launcher)}")
    for _ in range(100):
        if child_pid_file.exists():
            break
        time.sleep(0.05)
    assert child_pid_file.exists()

    te._stop_process(started["process_id"])

    child_pid = int(child_pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_managed_background_process_rejects_unknown_id(tmp_path):
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))
    assert not te._inspect_processes("missing")["success"]
    assert not te._stop_process("missing")["success"]


def test_windows_process_is_suspended_until_assigned_to_job(tmp_path, monkeypatch):
    events = []

    class Stdout:
        def read1(self, size):
            return b""

        def close(self):
            pass

    class Process:
        pid = 123
        stdout = Stdout()
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

    def popen(*args, **kwargs):
        events.append(("create", kwargs["creationflags"]))
        return Process()

    monkeypatch.setattr(process_tools.sys, "platform", "win32")
    monkeypatch.setattr(process_tools.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False)
    monkeypatch.setattr(process_tools.subprocess, "Popen", popen)
    job_active = [True]
    monkeypatch.setattr(process_tools, "_create_windows_job", lambda proc: events.append(("assign", proc.pid)) or 77)
    monkeypatch.setattr(process_tools, "_resume_windows_process", lambda proc: events.append(("resume", proc.pid)))
    monkeypatch.setattr(process_tools, "_windows_job_active_processes", lambda job: int(job_active[0]))
    monkeypatch.setattr(process_tools, "_close_windows_job", lambda job, terminate=False: None)

    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))
    started = te._start_process("server")

    assert started["status"] == "running"
    assert events == [("create", 0x204), ("assign", 123), ("resume", 123)]
    job_active[0] = False
    te._processes[started["process_id"]]["capture_thread"].join(timeout=1)


def test_windows_process_info_keeps_job_open_while_descendants_run(monkeypatch):
    closed = []

    class Process:
        pid = 123
        returncode = 0

        def poll(self):
            return 0

        def wait(self):
            return 0

    record = {
        "process_id": "p-test",
        "pid": 123,
        "process": Process(),
        "exit_code": None,
        "command": "server",
        "cwd": "C:\\work",
        "log_path": ".kia/processes/p-test.log",
        "log_truncated": False,
        "job_handle": 77,
        "job_lock": threading.Lock(),
        "state_lock": threading.Lock(),
    }
    active = iter((1, 0))
    monkeypatch.setattr(process_tools, "_windows_job_active_processes", lambda job: next(active))
    monkeypatch.setattr(process_tools, "_close_windows_job", lambda job: closed.append(job))

    assert ToolExecutor._process_info(record)["status"] == "running"
    assert record["job_handle"] == 77
    assert closed == []
    assert ToolExecutor._process_info(record)["status"] == "exited"
    assert record["job_handle"] is None
    assert closed == [77]


def test_exec_lingering_reader_marks_capture_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(command_tools, "EXEC_READER_JOIN_TIMEOUT", 0.1)
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))
    started = time.monotonic()
    # Spawn a lingering grandchild holding the pipes (portable `sleep`).
    res = te._exec_command(
        "python -c \"import subprocess,sys; subprocess.Popen([sys.executable,"
        " '-c','import time; time.sleep(30)']); print('done')\""
    )
    artifact = Path(res["_artifact_path"])
    try:
        assert time.monotonic() - started < 5
        assert res["artifact_capture_incomplete"] is True
        assert res["artifact_truncated"] is True
    finally:
        artifact.unlink(missing_ok=True)


def test_edit_file_tolerant_on_disk(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("hello   \nworld\n")
    te = ToolExecutor(work_dir=str(tmp_path))
    res = te._edit_file(str(f), "hello\nworld", "hi\nearth")
    assert res["success"] and f.read_text() == "hi\nearth\n"

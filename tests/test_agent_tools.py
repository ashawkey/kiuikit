"""Tests for the core file tools: apply_edit, multi_edit, ls, and
gitignore-aware glob/grep (kiui.agent.tools / kiui.agent.gitignore)."""

import os
import threading
import time
from pathlib import Path

import pytest

import kiui.agent.tools as tools
from kiui.agent.tools import (
    ToolExecutor,
    _human_size,
    apply_edit,
    find_match,
    format_tool_result,
)
from kiui.agent.gitignore import build_gitignore_matcher
from kiui.agent.permissions import SafetyGuard


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
    monkeypatch.setattr(tools, "MAX_EXEC_ARTIFACT_BYTES", 11)
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
        tools.tempfile,
        "NamedTemporaryFile",
        lambda **kwargs: (_ for _ in ()).throw(OSError("no temp space")),
    )
    monkeypatch.setattr(tools.subprocess, "Popen", lambda *a, **k: started.append(True))
    te = ToolExecutor(console=_SilentConsole(), work_dir=str(tmp_path))
    with pytest.raises(OSError, match="no temp space"):
        te._exec_command("echo unreachable")
    assert started == []


def test_exec_artifact_write_failure_keeps_draining(tmp_path, monkeypatch):
    original = tools.tempfile.NamedTemporaryFile

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
        tools.tempfile,
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


def test_exec_lingering_reader_marks_capture_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "EXEC_READER_JOIN_TIMEOUT", 0.1)
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

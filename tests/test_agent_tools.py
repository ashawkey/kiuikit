"""Tests for the core file tools: apply_edit, multi_edit, ls, and
gitignore-aware glob/grep (kiui.agent.tools / kiui.agent.gitignore)."""

import pytest

from kiui.agent.tools import apply_edit, find_match, ToolExecutor, _human_size
from kiui.agent.gitignore import build_gitignore_matcher


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


# ----- ls / glob / multi_edit tools ---------------------------------------

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


def test_edit_file_tolerant_on_disk(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("hello   \nworld\n")
    te = ToolExecutor(work_dir=str(tmp_path))
    res = te._edit_file(str(f), "hello\nworld", "hi\nearth")
    assert res["success"] and f.read_text() == "hi\nearth\n"

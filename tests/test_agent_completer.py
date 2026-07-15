"""Tests for the terminal @-file fuzzy completer (AtFileCompleter).

Focus: correctness of the cached candidate index + fzf-style fuzzy match,
and that bulky / hidden directories stay out of results (git-sourced when
available, pruned filesystem walk otherwise).
"""

import os
import shutil

import pytest

from kiui.agent.terminal import AtFileCompleter


class _Doc:
    def __init__(self, text: str):
        self.text_before_cursor = text


def _complete(comp: AtFileCompleter, text: str) -> list[str]:
    return [c.text for c in comp.get_completions(_Doc(text), None)]


def _touch(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()


def test_fuzzy_finds_nested_file(tmp_path):
    _touch(str(tmp_path / "src" / "a" / "b" / "widget.txt"))
    comp = AtFileCompleter(tmp_path)
    res = _complete(comp, "@widget")
    assert "@src/a/b/widget.txt" in res


def test_fuzzy_char_sequence_match(tmp_path):
    _touch(str(tmp_path / "pkg" / "dottedthing.txt"))
    comp = AtFileCompleter(tmp_path)
    # "dott" matches as a char-sequence / substring
    res = _complete(comp, "@dott")
    assert "@pkg/dottedthing.txt" in res


def test_skip_dirs_are_pruned(tmp_path):
    _touch(str(tmp_path / "node_modules" / "pkg" / "index.js"))
    _touch(str(tmp_path / "src" / "index.js"))
    comp = AtFileCompleter(tmp_path)
    res = _complete(comp, "@index")
    assert "@src/index.js" in res
    assert not any("node_modules" in r for r in res)


def test_hidden_dirs_and_files_skipped(tmp_path):
    _touch(str(tmp_path / ".git" / "config"))
    _touch(str(tmp_path / ".secret_config"))
    comp = AtFileCompleter(tmp_path)
    res = _complete(comp, "@config")
    assert res == []


def test_results_are_capped(tmp_path):
    for i in range(100):
        _touch(str(tmp_path / "src" / f"match_{i}.py"))
    comp = AtFileCompleter(tmp_path)
    res = _complete(comp, "@match")
    assert len(res) == AtFileCompleter.MAX_COMPLETIONS


def test_index_is_cached_across_keystrokes(tmp_path):
    # After the first build, subsequent keystrokes reuse the cached index
    # rather than re-walking the filesystem.
    _touch(str(tmp_path / "src" / "widget.txt"))
    comp = AtFileCompleter(tmp_path)
    _complete(comp, "@w")
    built_at = comp._index_built_at
    assert built_at > 0.0
    _complete(comp, "@wi")
    assert comp._index_built_at == built_at  # not rebuilt


def test_candidate_cap_bounds_work(tmp_path):
    # Force a tiny candidate cap and a larger tree; index build terminates.
    comp = AtFileCompleter(tmp_path)
    comp._MAX_CANDIDATES = 50
    for i in range(500):
        _touch(str(tmp_path / "src" / f"m{i % 10}" / f"file{i}.py"))
    res = _complete(comp, "@file")
    assert isinstance(res, list)


def test_bare_at_lists_top_level(tmp_path):
    _touch(str(tmp_path / "readme.md"))
    os.makedirs(tmp_path / "src", exist_ok=True)
    comp = AtFileCompleter(tmp_path)
    res = _complete(comp, "@")
    assert "@readme.md" in res
    assert "@src/" in res


def test_no_at_yields_nothing(tmp_path):
    comp = AtFileCompleter(tmp_path)
    assert _complete(comp, "hello world") == []


def test_git_index_honors_gitignore(tmp_path):
    import subprocess

    if not shutil.which("git"):
        pytest.skip("git not available")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    _touch(str(tmp_path / "src" / "keep.py"))
    _touch(str(tmp_path / "build" / "ignored.py"))
    (tmp_path / ".gitignore").write_text("build/\n")

    comp = AtFileCompleter(tmp_path)
    # Untracked-but-not-ignored files are included via --others.
    res = _complete(comp, "@keep")
    assert "@src/keep.py" in res
    # Ignored files must not appear even though they exist on disk.
    res = _complete(comp, "@ignored")
    assert not any("ignored" in r for r in res)

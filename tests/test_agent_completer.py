"""Tests for the terminal @-file fuzzy completer (AtFileCompleter).

Focus: correctness of the cached candidate index + fzf-style fuzzy match,
and that bulky / hidden directories stay out of results (git-sourced when
available, pruned filesystem walk otherwise).
"""

import os
import shutil
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from prompt_toolkit.buffer import Buffer, CompletionState
from prompt_toolkit.completion import Completion
from prompt_toolkit.document import Document
from prompt_toolkit.keys import Keys

from kiui.agent.terminal import AtFileCompleter, TerminalInput


class _Doc:
    def __init__(self, text: str):
        self.text_before_cursor = text


def _complete(comp: AtFileCompleter, text: str) -> list[str]:
    return [c.text for c in comp.get_completions(_Doc(text), None)]


def _touch(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()


def _press_enter(buffer: Buffer) -> None:
    terminal = object.__new__(TerminalInput)
    bindings = terminal._create_keybindings()
    handler = next(b.handler for b in bindings.bindings if b.keys == (Keys.ControlM,))
    handler(SimpleNamespace(current_buffer=buffer))


def test_markdown_lexer_does_not_reparse_from_start():
    terminal = TerminalInput()
    lexer = terminal._session.app.layout.current_control.lexer.get_lexer()

    assert not lexer.sync_from_start()


def _message_terminal(*, pending=None, status=None):
    terminal = object.__new__(TerminalInput)
    terminal._busy = True
    terminal._status = list(status or [])
    terminal._status_lock = threading.Lock()
    terminal._pending_text = lambda: pending
    terminal._prompt_label = "> "
    terminal._session = SimpleNamespace(app=SimpleNamespace(
        output=SimpleNamespace(
            get_size=lambda: SimpleNamespace(columns=80)
        )
    ))
    return terminal


def test_busy_prompt_always_explains_that_new_messages_are_queued():
    terminal = _message_terminal()

    message = terminal._message()
    text = "".join(fragment for _, fragment in message)

    assert "Working..." in text
    assert "messages sent now will be queued" in text
    assert "queue> " not in text
    assert ("class:separator.busy", "─" * 79) in message
    assert ("class:prompt.busy", "> ") in message


def test_detailed_status_overrides_busy_fallback_and_shows_pending_message():
    terminal = _message_terminal(
        pending="follow up",
        status=[("class:status.text", "Authenticating...")],
    )

    text = "".join(fragment for _, fragment in terminal._message())

    assert "Authenticating..." in text
    assert "Working..." not in text
    assert "pending: follow up · runs next" in text


def test_enter_applies_selected_completion():
    document = Document("@term")
    completions = [
        Completion("@other.py", start_position=-5),
        Completion("@kiui/agent/terminal.py", start_position=-5),
    ]
    buffer = Buffer(document=document)
    buffer.complete_state = CompletionState(document, completions, complete_index=1)
    buffer.go_to_completion(1)

    _press_enter(buffer)

    assert buffer.text == "@kiui/agent/terminal.py"
    assert buffer.complete_state is None


def test_enter_applies_first_completion_when_none_selected():
    document = Document("@term")
    completion = Completion("@kiui/agent/terminal.py", start_position=-5)
    buffer = Buffer(document=document)
    buffer.complete_state = CompletionState(document, [completion])

    _press_enter(buffer)

    assert buffer.text == "@kiui/agent/terminal.py"
    assert buffer.complete_state is None


def test_up_preserves_normal_multiline_navigation():
    terminal = object.__new__(TerminalInput)
    terminal._pending_text = lambda: "pending"
    terminal._edit_pending = Mock()
    bindings = terminal._create_keybindings()
    handler = next(b.handler for b in bindings.bindings if b.keys == (Keys.Up,))
    buffer = Buffer(document=Document("first\nsecond", cursor_position=12))

    handler(SimpleNamespace(current_buffer=buffer, arg=1))

    assert buffer.document.cursor_position_row == 0
    terminal._edit_pending.assert_not_called()


def test_up_with_empty_buffer_withdraws_pending_message():
    terminal = object.__new__(TerminalInput)
    terminal._pending_text = lambda: "pending"
    terminal._edit_pending = Mock(return_value="pending")
    bindings = terminal._create_keybindings()
    handler = next(b.handler for b in bindings.bindings if b.keys == (Keys.Up,))
    buffer = Buffer()

    handler(SimpleNamespace(current_buffer=buffer, arg=1))

    assert buffer.text == "pending"
    terminal._edit_pending.assert_called_once_with()


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


def test_fuzzy_path_match_continues_after_separator(tmp_path):
    _touch(str(tmp_path / "src" / "components" / "widget.txt"))
    comp = AtFileCompleter(tmp_path)

    assert "@src/components/" in _complete(comp, "@components/")
    assert "@src/components/widget.txt" in _complete(comp, "@components/widget")


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

"""Tests for what /rewind shows before it changes anything."""

import io
import types
from pathlib import Path

from rich.console import Console

from kiui.agent.backend.sessions import SessionMixin
from kiui.agent.session_store import SessionStore
from kiui.agent.utils.rewind import ChangeTracker


class _Console:
    """Capture rendered output and answer prompts from a script."""

    def __init__(self, answers: list[str] | None = None):
        self.buffer = io.StringIO()
        self.rich = Console(file=self.buffer, width=200, no_color=True, legacy_windows=False)
        self.answers = list(answers or [])
        self.prompts: list[list[str]] = []

    def print(self, *args, **kwargs):
        self.rich.print(*args, **kwargs)

    table = print

    def _log(self, message, **kwargs):
        self.rich.print(message, markup=False)

    system = warn = error = user_input = response = tool = _log

    def tool_result(self, message, success=True):
        self._log(f"{'ok' if success else 'fail'}: {message}")

    def diff_edit(self, path, old_text, new_text, line_num=None, **kwargs):
        self._log(f"diff {path}@{line_num} -{old_text!r} +{new_text!r}")

    def reset_timeline(self):
        self._log("[timeline reset]")

    def select(self, message, choices, **kwargs):
        self.prompts.append(choices)
        wanted = self.answers.pop(0)
        return next(choice for choice in choices if choice.startswith(wanted))

    @property
    def text(self) -> str:
        return self.buffer.getvalue()


class _Agent(SessionMixin):
    """SessionMixin with just the collaborators /rewind touches."""

    def __init__(self, store, tracker, console):
        self.console = console
        self._session_id = "session"
        self._session_store = store
        self._session_revision_id = store.head_id
        self.changes = tracker
        self.context = types.SimpleNamespace(messages=[])
        self.round_id = 0
        self.replayed = 0

    def save_session(self, name=None, *, reason="autosave"):
        revision_id, code_revision_id, _ = self._session_store.commit(
            {"round_id": self.round_id, "messages": list(self.context.messages)},
            parent_id=self._session_revision_id,
            code_parent_id=self.changes.code_revision_id,
            changes=self.changes.pending_changes,
            reason=reason,
        )
        self._session_revision_id = revision_id
        self.changes.mark_committed(code_revision_id)

    def _restore_session_data(self, data):
        self.context.messages = list(data["messages"])
        self.round_id = data.get("round_id", 0)

    def _replay_context(self):
        self.replayed += 1


def _build(tmp_path: Path, answers: list[str]) -> tuple[_Agent, Path, _Console]:
    """Two rounds: one adds files, the next edits one and deletes another."""
    work = tmp_path / "work"
    work.mkdir()
    store = SessionStore(tmp_path / "sessions", "session")
    console = _Console(answers)
    tracker = ChangeTracker("session", work, console, store)
    agent = _Agent(store, tracker, console)

    agent.round_id = 1
    agent.context.messages = [{"role": "user", "content": "set up the parser module"}]
    for name, body in (("parser.py", "def parse():\n    return 1\n"), ("util.py", "X = 1\n")):
        tracker.track_write(1, str(work / name), body)
        (work / name).write_text(body)
    agent.save_session(reason="round")

    agent.round_id = 2
    agent.context.messages += [
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "make the parser handle floats and drop util"},
    ]
    new = "def parse(text):\n    return float(text)\n"
    tracker.track_edit_result(2, str(work / "parser.py"), (work / "parser.py").read_text(), new)
    (work / "parser.py").write_text(new)
    tracker.track_remove(2, str(work / "util.py"))
    (work / "util.py").unlink()
    agent.save_session(reason="round")
    return agent, work, console


def test_revision_list_shows_the_prompt_and_file_count_of_each_round(tmp_path: Path):
    agent, _, console = _build(tmp_path, [])
    agent._cmd_rewind("/rewind list")

    assert "set up the parser module" in console.text
    assert "make the parser handle floats and drop util" in console.text
    # Listing must not checkpoint or move anything.
    assert len(agent._session_store.candidates()) == 2


def test_picker_and_preview_describe_the_conversation_and_the_files(tmp_path: Path):
    agent, work, console = _build(tmp_path, [" 2.", "1."])
    agent._cmd_rewind("/rewind")

    picker, modes = console.prompts
    assert "set up the parser module" in picker[1]
    assert "2 files" in picker[1]
    # The preview names every file the checkout touches, and how it touches it.
    assert "modify  parser.py" in console.text
    assert "create  util.py" in console.text
    assert "1 round(s) will be dropped" in console.text
    assert "round 2  make the parser handle floats and drop util" in console.text
    # The mode labels answer "will this revert my files?" before anything happens.
    assert "3 → 1 messages" in modes[0]
    assert "2 file(s) reverted" in modes[0]
    assert "files untouched" in modes[1]

    assert (work / "parser.py").read_text() == "def parse():\n    return 1\n"
    assert (work / "util.py").read_text() == "X = 1\n"
    assert agent.round_id == 1 and agent.replayed == 1


def test_picker_options_are_the_listing_and_name_the_revision_type(tmp_path: Path):
    """No table precedes the picker, so every option must stand on its own."""
    agent, _, console = _build(tmp_path, [" 3.", "5."])
    agent.context.messages.append({"role": "user", "content": "one more thought"})
    agent.save_session(reason="pre-compaction")
    agent._cmd_rewind("/rewind")

    picker = console.prompts[0]
    assert "[current]" in picker[0] and "pre-compaction" in picker[0]
    assert "round 2" in picker[1] and "2 files" in picker[1]
    assert "make the parser handle floats and drop util" in picker[1]
    # The options replace the table rather than repeating it.
    assert "Session revisions" not in console.text


def test_conversation_only_rewind_leaves_the_files_alone(tmp_path: Path):
    agent, work, console = _build(tmp_path, [" 2.", "2."])
    agent._cmd_rewind("/rewind")

    assert (work / "parser.py").read_text() == "def parse(text):\n    return float(text)\n"
    assert not (work / "util.py").exists()
    assert len(agent.context.messages) == 1


def test_diffs_can_be_reviewed_before_choosing_a_mode(tmp_path: Path):
    agent, work, console = _build(tmp_path, [" 2.", "4.", "5."])
    agent._cmd_rewind("/rewind")

    assert "diff parser.py@1" in console.text
    assert "diff util.py@1" in console.text
    assert "Rewind cancelled." in console.text
    # Cancelling changes nothing.
    assert (work / "parser.py").read_text() == "def parse(text):\n    return float(text)\n"
    assert agent.replayed == 0


def test_preview_warns_about_files_edited_outside_the_agent(tmp_path: Path):
    agent, work, console = _build(tmp_path, [" 2.", "5."])
    (work / "parser.py").write_text("hand edited\n")
    agent._cmd_rewind("/rewind")

    assert "changed on disk" in console.text
    assert "parser.py" in console.text


def test_replay_shows_tool_calls_the_way_the_live_view_does(tmp_path: Path):
    """Replay runs after a rewind or a resume; it must not fall back to raw JSON."""
    agent, _, console = _build(tmp_path, [])
    agent.context.messages = [
        {"role": "user", "content": "check the file"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "1", "function": {"name": "read_file", "arguments": '{"file": "a.txt"}'}},
        ]},
        {"role": "tool", "tool_call_id": "1", "content": "1  an error message"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "2", "function": {"name": "edit_file", "arguments": '{"file": "a.txt", "old_text": "x", "new_text": "y"}'}},
        ]},
        {"role": "tool", "tool_call_id": "2", "content": "Error: File not found: a.txt"},
    ]
    SessionMixin._replay_context(agent)

    assert "read_file a.txt:1-1000" in console.text
    assert "edit_file a.txt" in console.text
    assert '{"file"' not in console.text
    # Only the result actually formatted as a failure is marked as one.
    assert "ok: 1  an error message" in console.text
    assert "fail: Error: File not found: a.txt" in console.text


def test_preview_states_when_a_revision_touches_no_files(tmp_path: Path):
    # With nothing to revert the menu offers no diff view, so Cancel is item 4.
    agent, _, console = _build(tmp_path, ["4."])
    agent.round_id = 3
    agent.context.messages.append({"role": "user", "content": "just talking"})
    agent.save_session(reason="round")
    target = agent._session_store.candidates()[1]["id"]
    agent._cmd_rewind(f"/rewind {target[:8]}")

    assert "no files will change" in console.text
    assert "no file changes" in console.prompts[0][0]

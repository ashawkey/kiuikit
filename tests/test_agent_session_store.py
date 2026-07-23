import shutil
from pathlib import Path

import pytest

from kiui.agent.session_store import SessionStore
from kiui.agent.utils.rewind import ChangeTracker


class _Console:
    pass


def _state(round_id: int, text: str) -> dict:
    return {
        "round_id": round_id,
        "messages": [{"role": "user", "content": text}],
    }


def test_session_revisions_branch_without_losing_descendants(tmp_path: Path):
    store = SessionStore(tmp_path, "session")
    root, _, _ = store.commit(
        _state(1, "root"), parent_id=None, code_parent_id=None, changes=[], reason="round"
    )
    old_tip, _, _ = store.commit(
        _state(2, "old tip"), parent_id=root, code_parent_id=None, changes=[], reason="round"
    )

    store.checkout(root)
    new_tip, _, _ = store.commit(
        _state(2, "new tip"), parent_id=root, code_parent_id=None, changes=[], reason="round"
    )

    assert store.revisions[old_tip]["parentId"] == root
    assert store.revisions[new_tip]["parentId"] == root
    assert store.materialize(old_tip)["messages"][0]["content"] == "old tip"
    assert store.materialize(new_tip)["messages"][0]["content"] == "new tip"
    assert not (store.path / "snapshot.json").exists()


def test_session_history_ignores_only_a_torn_final_record(tmp_path: Path):
    store = SessionStore(tmp_path, "session")
    revision, _, _ = store.commit(
        _state(1, "saved"), parent_id=None, code_parent_id=None, changes=[], reason="round"
    )
    with store.history_path.open("ab") as f:
        f.write(b'{"type":"revision"')

    recovered = SessionStore(tmp_path, "session")
    assert recovered.head_id == revision

    with store.history_path.open("ab") as f:
        f.write(b"\n")
    with pytest.raises(ValueError, match="Corrupted JSONL"):
        SessionStore(tmp_path, "session")


def test_code_revision_can_move_between_branches(tmp_path: Path):
    sessions = tmp_path / "sessions"
    work = tmp_path / "work"
    work.mkdir()
    target = work / "value.txt"
    target.write_text("zero")
    store = SessionStore(sessions, "session")
    tracker = ChangeTracker("session", work, _Console(), store)

    tracker.track_edit_result(1, str(target), "zero", "one")
    target.write_text("one")
    root, code_one, _ = store.commit(
        _state(1, "one"),
        parent_id=None,
        code_parent_id=None,
        changes=tracker.pending_changes,
        reason="round",
    )
    tracker.mark_committed(code_one)

    tracker.track_edit_result(2, str(target), "one", "two")
    target.write_text("two")
    old_tip, code_two, _ = store.commit(
        _state(2, "two"),
        parent_id=root,
        code_parent_id=code_one,
        changes=tracker.pending_changes,
        reason="round",
    )
    tracker.mark_committed(code_two)

    assert tracker.checkout_code(code_one) == 1
    assert target.read_text() == "one"
    store.checkout(root)

    tracker.track_edit_result(2, str(target), "one", "three")
    target.write_text("three")
    new_tip, code_three, _ = store.commit(
        _state(2, "three"),
        parent_id=root,
        code_parent_id=code_one,
        changes=tracker.pending_changes,
        reason="round",
    )
    tracker.mark_committed(code_three)

    assert tracker.checkout_code(code_two) == 2
    assert target.read_text() == "two"
    assert store.revisions[old_tip]["parentId"] == store.revisions[new_tip]["parentId"] == root


def test_removed_tree_uses_deduplicated_content_addressed_objects(tmp_path: Path):
    sessions = tmp_path / "sessions"
    work = tmp_path / "work"
    removed = work / "removed"
    removed.mkdir(parents=True)
    (removed / "a.bin").write_bytes(b"same bytes")
    (removed / "b.bin").write_bytes(b"same bytes")

    store = SessionStore(sessions, "session")
    tracker = ChangeTracker("session", work, _Console(), store)
    tracker.track_remove(1, str(removed))
    descriptor = tracker.pending_changes[0]["before"]
    shutil.rmtree(removed)

    root, code_revision, _ = store.commit(
        _state(1, "removed"),
        parent_id=None,
        code_parent_id=None,
        changes=tracker.pending_changes,
        reason="round",
    )
    tracker.mark_committed(code_revision)
    assert descriptor["kind"] == "tree"
    assert len([path for path in store.objects_path.rglob("*") if path.is_file()]) == 2

    tracker.checkout_code(None)
    assert (removed / "a.bin").read_bytes() == b"same bytes"
    assert (removed / "b.bin").read_bytes() == b"same bytes"
    assert store.revisions[root]["codeRevisionId"] == code_revision

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


def test_revisions_are_delta_encoded_but_materialize_in_full(tmp_path: Path):
    """Appending a round must not rewrite the whole message list and state."""
    store = SessionStore(tmp_path, "session")
    system = {"role": "system", "content": "UNIQUE_PROMPT_MARKER " + "x" * 2000}
    messages = []
    parent = None
    for round_id in range(1, 21):
        messages = messages + [{"role": "user", "content": f"m{round_id}"}]
        parent, _, _ = store.commit(
            {"messages": list(messages), "round_id": round_id, "system_prompt": system},
            parent_id=parent, code_parent_id=None, changes=[], reason="round",
        )

    history = (store.path / "history.jsonl").read_text(encoding="utf-8")
    # The unchanging system prompt is written once (with the root revision),
    # not once per round.
    assert history.count("UNIQUE_PROMPT_MARKER") == 1
    # Likewise each message ID is named once by the revision that added it,
    # instead of being repeated by every later revision.
    first_id = store.revisions[parent]["messageIds"][0]
    assert history.count(first_id) == 2  # the message record, and the root revision

    fresh = SessionStore(tmp_path, "session")
    data = fresh.materialize()
    assert data["messages"] == messages
    assert data["system_prompt"] == system and data["round_id"] == 20
    assert fresh.revisions == store.revisions


def test_revision_state_is_snapshotted_not_aliased(tmp_path: Path):
    """A revision records the state it was written with, not a live reference.

    Callers pass their own mutable session state (token totals, the system
    prompt dict), which keeps changing after the commit returns.
    """
    store = SessionStore(tmp_path, "session")
    totals = {"total": 0}
    system = {"role": "system", "content": "persona A"}
    messages = [{"role": "user", "content": "m1"}]

    first, _, _ = store.commit(
        {"messages": list(messages), "round_id": 1, "token_totals": totals,
         "system_prompt": system},
        parent_id=None, code_parent_id=None, changes=[], reason="initial",
    )
    totals["total"] = 12345
    system["content"] = "persona B"
    messages.append({"role": "assistant", "content": "m2"})
    second, _, _ = store.commit(
        {"messages": list(messages), "round_id": 2, "token_totals": totals,
         "system_prompt": system},
        parent_id=first, code_parent_id=None, changes=[], reason="round",
    )

    reloaded = SessionStore(tmp_path, "session")
    assert reloaded.materialize(first)["token_totals"] == {"total": 0}
    assert reloaded.materialize(first)["system_prompt"]["content"] == "persona A"
    assert reloaded.materialize(second)["token_totals"] == {"total": 12345}
    assert reloaded.materialize(second)["system_prompt"]["content"] == "persona B"


def test_commit_picks_up_history_written_by_another_process(tmp_path: Path):
    """Saving reloads only when the file moved, but must never miss a change."""
    first_agent = SessionStore(tmp_path, "session")
    root, _, _ = first_agent.commit(
        _state(1, "root"), parent_id=None, code_parent_id=None, changes=[], reason="initial"
    )

    other_agent = SessionStore(tmp_path, "session")
    theirs, _, _ = other_agent.commit(
        _state(2, "theirs"), parent_id=root, code_parent_id=None, changes=[], reason="round"
    )

    mine, _, _ = first_agent.commit(
        _state(2, "mine"), parent_id=root, code_parent_id=None, changes=[], reason="round"
    )
    assert theirs in first_agent.revisions

    reloaded = SessionStore(tmp_path, "session")
    assert reloaded.materialize(theirs)["messages"][0]["content"] == "theirs"
    assert reloaded.materialize(mine)["messages"][0]["content"] == "mine"
    assert reloaded.head_id == mine


def test_delta_revisions_survive_branching_and_replacement(tmp_path: Path):
    """Rewind branches and compaction replace messages rather than appending."""
    store = SessionStore(tmp_path, "session")
    base = [{"role": "user", "content": "a"}]
    root, _, _ = store.commit(
        {"messages": list(base), "round_id": 1, "goal": "g"},
        parent_id=None, code_parent_id=None, changes=[], reason="round",
    )
    appended = base + [{"role": "assistant", "content": "b"}]
    tip, _, _ = store.commit(
        {"messages": list(appended), "round_id": 2, "goal": "g"},
        parent_id=root, code_parent_id=None, changes=[], reason="round",
    )
    # Divergent branch from the root.
    branched = base + [{"role": "assistant", "content": "other"}]
    branch, _, _ = store.commit(
        {"messages": list(branched), "round_id": 2, "goal": "g2"},
        parent_id=root, code_parent_id=None, changes=[], reason="round",
    )
    # Compaction: messages replaced wholesale and a state key disappears.
    compacted = [{"role": "user", "content": "[summary]"}]
    summary, _, _ = store.commit(
        {"messages": list(compacted), "round_id": 3},
        parent_id=tip, code_parent_id=None, changes=[], reason="round",
    )

    fresh = SessionStore(tmp_path, "session")
    assert fresh.materialize(root)["messages"] == base
    assert fresh.materialize(tip)["messages"] == appended
    assert fresh.materialize(branch)["messages"] == branched
    assert fresh.materialize(branch)["goal"] == "g2"
    assert fresh.materialize(summary)["messages"] == compacted
    assert "goal" not in fresh.materialize(summary)


def test_legacy_full_form_revisions_still_load(tmp_path: Path):
    """Histories written before delta encoding must remain readable."""
    store = SessionStore(tmp_path, "session")
    root, _, _ = store.commit(
        _state(1, "root"), parent_id=None, code_parent_id=None, changes=[], reason="round"
    )
    tip, _, _ = store.commit(
        _state(2, "tip"), parent_id=root, code_parent_id=None, changes=[], reason="round"
    )
    # Rewrite every revision record in the old full form (no delta keys).
    import json

    lines = []
    for line in (store.path / "history.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("type") == "revision":
            revision = store.revisions[record["id"]]
            record = {
                key: value for key, value in record.items()
                if key not in ("baseCount", "addedIds", "stateChanged", "stateRemoved")
            }
            record["messageIds"] = revision["messageIds"]
            record["state"] = revision["state"]
        lines.append(json.dumps(record))
    (store.path / "history.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fresh = SessionStore(tmp_path, "session")
    assert fresh.materialize(tip)["messages"][0]["content"] == "tip"
    assert fresh.revisions == store.revisions
    # A new delta-encoded revision can still extend a legacy history.
    extended, _, _ = fresh.commit(
        _state(3, "next"), parent_id=tip, code_parent_id=None, changes=[], reason="round"
    )
    assert SessionStore(tmp_path, "session").materialize(extended)["round_id"] == 3


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


def _session(tmp_path: Path) -> tuple[SessionStore, ChangeTracker, Path]:
    sessions = tmp_path / "sessions"
    work = tmp_path / "work"
    work.mkdir()
    store = SessionStore(sessions, "session")
    return store, ChangeTracker("session", work, _Console(), store), work


def _round(store, tracker, parent, round_id, text) -> str:
    revision_id, code_revision_id, _ = store.commit(
        _state(round_id, text),
        parent_id=parent,
        code_parent_id=tracker.code_revision_id,
        changes=tracker.pending_changes,
        reason="round",
    )
    tracker.mark_committed(code_revision_id)
    return revision_id


def _write(tracker, work, round_id, name, content) -> None:
    path = work / name
    original = path.read_text() if path.exists() else None
    if original is None:
        tracker.track_write(round_id, str(path), content)
    else:
        tracker.track_edit_result(round_id, str(path), original, content)
    path.write_text(content)


def test_candidates_carry_the_prompt_and_file_counts(tmp_path: Path):
    store, tracker, work = _session(tmp_path)
    root = _round(store, tracker, None, 1, "start the parser")
    _write(tracker, work, 2, "parser.py", "a\nb\n")
    _write(tracker, work, 2, "util.py", "x\n")
    _round(store, tracker, root, 2, "add the parser and a helper")

    newest, oldest = store.candidates()
    assert newest["round_id"] == 2 and newest["current"]
    assert newest["prompt"] == "add the parser and a helper"
    assert newest["files"] == 2
    assert newest["new_messages"] == 0
    assert oldest["prompt"] == "start the parser"
    assert oldest["files"] == 0


def test_plan_reports_no_file_changes_when_code_already_matches(tmp_path: Path):
    store, tracker, work = _session(tmp_path)
    _write(tracker, work, 1, "parser.py", "a\n")
    root = _round(store, tracker, None, 1, "one")
    # A revision that changes no files shares its parent's code revision.
    _round(store, tracker, root, 2, "two")

    plan = tracker.plan_checkout(store.materialize(root)["code_revision_id"])
    assert not plan
    assert plan.files == 0 and plan.steps == []
    assert tracker.apply_plan(plan) == 0
    assert (work / "parser.py").read_text() == "a\n"


def test_plan_classifies_and_counts_what_a_checkout_would_do(tmp_path: Path):
    store, tracker, work = _session(tmp_path)
    _write(tracker, work, 1, "keep.py", "one\ntwo\nthree\n")
    _write(tracker, work, 1, "gone.py", "old\n")
    root = _round(store, tracker, None, 1, "one")

    _write(tracker, work, 2, "keep.py", "one\nTWO\nthree\nfour\n")
    _write(tracker, work, 2, "added.py", "new\n")
    tracker.track_remove(2, str(work / "gone.py"))
    (work / "gone.py").unlink()
    tip = _round(store, tracker, root, 2, "two")

    plan = tracker.plan_checkout(store.materialize(root)["code_revision_id"])
    assert [(delta.op, delta.path) for delta in plan.deltas] == [
        ("delete", "added.py"),
        ("create", "gone.py"),
        ("modify", "keep.py"),
    ]
    assert (plan.added, plan.removed) == (2, 3)
    assert plan.dirty == ()

    tracker.apply_plan(plan)
    assert (work / "keep.py").read_text() == "one\ntwo\nthree\n"
    assert (work / "gone.py").read_text() == "old\n"
    assert not (work / "added.py").exists()
    assert tracker.code_revision_id == store.materialize(root)["code_revision_id"]

    # The plan is symmetric: replaying it forward restores the tip state.
    forward = tracker.plan_checkout(store.materialize(tip)["code_revision_id"])
    tracker.apply_plan(forward)
    assert (work / "keep.py").read_text() == "one\nTWO\nthree\nfour\n"
    assert (work / "added.py").read_text() == "new\n"
    assert not (work / "gone.py").exists()


def test_plan_flags_files_edited_outside_the_agent(tmp_path: Path):
    store, tracker, work = _session(tmp_path)
    _write(tracker, work, 1, "a.py", "one\n")
    _write(tracker, work, 1, "b.py", "one\n")
    root = _round(store, tracker, None, 1, "one")
    _write(tracker, work, 2, "a.py", "two\n")
    _write(tracker, work, 2, "b.py", "two\n")
    _round(store, tracker, root, 2, "two")

    assert tracker.plan_checkout(store.materialize(root)["code_revision_id"]).dirty == ()

    (work / "a.py").write_text("edited by hand\n")
    plan = tracker.plan_checkout(store.materialize(root)["code_revision_id"])
    assert plan.dirty == ("a.py",)


def test_line_endings_do_not_make_a_clean_tree_look_edited(tmp_path: Path):
    """Tools write text mode, so recorded bytes and disk bytes differ on Windows."""
    store, tracker, work = _session(tmp_path)
    path = work / "crlf.py"
    tracker.track_write(1, str(path), "one\ntwo\n")
    path.write_bytes(b"one\r\ntwo\r\n")
    root = _round(store, tracker, None, 1, "one")
    tracker.track_edit_result(2, str(path), "one\ntwo\n", "one\ntwo\nthree\n")
    path.write_bytes(b"one\r\ntwo\r\nthree\r\n")
    _round(store, tracker, root, 2, "two")

    plan = tracker.plan_checkout(store.materialize(root)["code_revision_id"])
    assert plan.dirty == ()
    assert (plan.added, plan.removed) == (0, 1)


def test_text_hunks_cover_only_changed_regions(tmp_path: Path):
    store, tracker, work = _session(tmp_path)
    _write(tracker, work, 1, "a.py", "one\ntwo\nthree\nfour\n")
    root = _round(store, tracker, None, 1, "one")
    _write(tracker, work, 2, "a.py", "one\nTWO\nthree\nfour\nfive\n")
    _round(store, tracker, root, 2, "two")

    delta = tracker.plan_checkout(store.materialize(root)["code_revision_id"]).deltas[0]
    assert store.text_hunks(delta.before, delta.after) == [(2, "TWO", "two"), (5, "five", "")]


def test_directory_trees_are_compared_against_disk_without_being_stored(tmp_path: Path):
    store, tracker, work = _session(tmp_path)
    package = work / "package"
    package.mkdir()
    (package / "a.py").write_text("one\n")
    (package / "b.py").write_text("two\n")
    root = _round(store, tracker, None, 1, "one")

    tracker.track_remove(2, str(package))
    shutil.rmtree(package)
    tip = _round(store, tracker, root, 2, "two")
    tracker.checkout_code(store.materialize(root)["code_revision_id"])

    forward = store.materialize(tip)["code_revision_id"]
    assert tracker.plan_checkout(forward).dirty == ()
    (package / "a.py").write_text("edited by hand\n")
    assert tracker.plan_checkout(forward).dirty == ("package",)


def test_checkout_requires_a_saved_session(tmp_path: Path):
    store, tracker, work = _session(tmp_path)
    _round(store, tracker, None, 1, "one")
    _write(tracker, work, 2, "a.py", "pending\n")

    with pytest.raises(RuntimeError, match="Save the session"):
        tracker.plan_checkout(None)

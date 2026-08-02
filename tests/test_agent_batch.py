"""Tests for context-isolated turns and the bundled batch skill."""

import json
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from kiui.agent.backend import LLMAgent
from kiui.agent.context import CompactionState, ContextManager
from kiui.agent.skills import BUNDLED_SKILLS_DIR, load_skill_tools
from kiui.agent.utils.interrupt import TurnOutcome
from kiui.agent.tools import ToolExecutor
from kiui.agent.tools.registry import ToolRegistry
from kiui.agent.ui import AgentConsole
from kiui.agent.utils.io import EventHub
from kiui.agent.utils.storage import clean_storage

BATCH_SKILL_DIR = BUNDLED_SKILLS_DIR / "batch"


def _load_batch_tools():
    entries = load_skill_tools(BATCH_SKILL_DIR)
    assert len(entries) == 1
    return entries[0]


class _Indicator:
    """Records the suffixes a run-level indicator is updated with."""

    def __init__(self, status_suffix=""):
        self.suffixes = [status_suffix]

    def set_status_suffix(self, status_suffix):
        self.suffixes.append(status_suffix)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Console:
    def __init__(self):
        self.thinking_calls = []
        self.indicators = []
        self.quiet_depth = 0

    def thinking(self, **kwargs):
        self.thinking_calls.append(kwargs)
        indicator = _Indicator(kwargs.get("status_suffix", ""))
        self.indicators.append(indicator)
        return indicator

    @contextmanager
    def suppressed(self):
        self.quiet_depth += 1
        try:
            yield
        finally:
            self.quiet_depth -= 1

    def system(self, *args, **kwargs):
        pass

    def warn(self, *args, **kwargs):
        pass

    def tool(self, *args, **kwargs):
        pass


def _agent(responses=None, system="system prompt", console=None, executor=None):
    """An agent double exposing exactly what run_isolated_turn touches."""
    context = ContextManager(system)
    agent = NS(
        context=context,
        console=console or _Console(),
        # Skill state lives on the executor and is part of the rollback.
        tool_executor=executor if executor is not None else ToolExecutor(),
        cancellation=None,
        _pending_images=[],
        _isolated_turn_active=False,
        _last_interrupted=False,
        _last_turn_outcome=TurnOutcome.COMPLETED,
        _interrupt_reverts_prompt=False,
        _last_finish_reason="stop",
        _compaction_floor_tokens=None,
        seen=[],
    )

    queue = list(responses or [])

    def get_response():
        # Record what this turn actually sees, then behave like a real turn:
        # append an assistant message to the live context.
        agent.seen.append([dict(m) for m in context.messages])
        outcome = queue.pop(0) if queue else "ok"
        if isinstance(outcome, Exception):
            raise outcome
        agent._last_interrupted = bool(getattr(outcome, "interrupted", False))
        agent._last_turn_outcome = (
            TurnOutcome.USER_INTERRUPTED
            if agent._last_interrupted else TurnOutcome.COMPLETED
        )
        text = outcome.text if hasattr(outcome, "text") else outcome
        context.add({"role": "assistant", "content": text or ""})
        return text

    agent.get_response = get_response
    agent.run_isolated_turn = lambda prompt: LLMAgent.run_isolated_turn(agent, prompt)
    return agent


def _interrupted(text=None):
    return NS(text=text, interrupted=True)


# ----- core primitive: run_isolated_turn ------------------------------------

def test_isolated_turn_restores_context_exactly():
    agent = _agent(["first", "second", "third"])
    agent.context.add({"role": "user", "content": "prior conversation"})
    agent.context.add({"role": "assistant", "content": "prior reply"})
    before = [dict(m) for m in agent.context.messages]
    before_chars = agent.context.total_chars

    for i in range(3):
        response, outcome = agent.run_isolated_turn(f"item {i}")
        assert outcome == TurnOutcome.COMPLETED
        assert response == ["first", "second", "third"][i]

    assert agent.context.messages == before
    # The cached char total must be restored too, not just the message list.
    assert agent.context.total_chars == before_chars


def test_isolated_turns_do_not_see_enclosing_context_or_each_other():
    agent = _agent(["a", "b", "c"])
    agent.context.add({"role": "user", "content": "prior"})
    agent.context.add({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "batch-call",
            "type": "function",
            "function": {"name": "run_batch", "arguments": "{}"},
        }],
    })

    for i in range(3):
        agent.run_isolated_turn(f"item {i}")

    # Every turn saw only its own prompt. The enclosing context may end in an
    # unresolved run_batch tool call, so it cannot be sent as a provider prefix.
    for i, seen in enumerate(agent.seen):
        assert seen == [{"role": "user", "content": f"item {i}"}]


def test_isolated_turn_restores_compaction_state_and_images():
    agent = _agent(["only"])
    state = CompactionState(original_request="the original ask")
    agent.context.compaction_state = state
    agent._compaction_floor_tokens = 4321
    queued = {"file": "outer.png", "url": "data:outer"}
    agent._pending_images.append(queued)

    def get_response():
        # Stand in for a turn that reads an image and triggers compaction.
        agent._pending_images.append({"file": "x.png", "url": "data:..."})
        agent.context.compaction_state = CompactionState(original_request="leaked")
        agent._compaction_floor_tokens = 99
        return "done"

    agent.get_response = get_response

    agent.run_isolated_turn("item")

    assert agent.context.compaction_state is state
    assert agent._compaction_floor_tokens == 4321
    assert agent._pending_images == [queued]


def test_isolated_turn_restores_context_after_failure():
    agent = _agent([RuntimeError("provider exploded")])
    agent.context.add({"role": "user", "content": "prior"})
    before = [dict(m) for m in agent.context.messages]

    with pytest.raises(RuntimeError):
        agent.run_isolated_turn("item")

    assert agent.context.messages == before
    assert agent._isolated_turn_active is False


def test_isolated_turn_reports_interruption_without_overwriting_outer_state():
    agent = _agent([_interrupted()])
    agent._last_interrupted = False
    agent._interrupt_reverts_prompt = False
    agent._last_finish_reason = "tool_calls"

    def get_response():
        agent._last_interrupted = True
        agent._last_turn_outcome = TurnOutcome.USER_INTERRUPTED
        agent._interrupt_reverts_prompt = True
        agent._last_finish_reason = "cancelled"
        return None

    agent.get_response = get_response
    response, outcome = agent.run_isolated_turn("item")

    assert response is None
    assert outcome == TurnOutcome.USER_INTERRUPTED
    assert agent._last_interrupted is False
    assert agent._interrupt_reverts_prompt is False
    assert agent._last_finish_reason == "tool_calls"


def test_isolated_turn_skipped_when_already_cancelled():
    agent = _agent(["unused"])
    agent.cancellation = NS(cancelled=True)

    response, outcome = agent.run_isolated_turn("item")

    assert (response, outcome) == (None, TurnOutcome.USER_INTERRUPTED)
    assert agent.seen == []  # the turn never ran


def test_isolated_turns_cannot_nest():
    agent = _agent()

    def get_response():
        return agent.run_isolated_turn("inner")

    agent.get_response = get_response

    with pytest.raises(RuntimeError, match="cannot nest"):
        agent.run_isolated_turn("outer")
    assert agent._isolated_turn_active is False


def test_isolated_turn_never_commits_a_session_revision():
    """A compaction inside an item must not move the durable head onto it."""
    saved = []
    agent = _agent()
    agent.context.add({"role": "user", "content": "the real conversation"})
    agent._session_id = "s1"
    agent._session_store = object()
    agent._session_revision_id = "outer-revision"

    def save_session(name, *, reason="autosave"):
        saved.append((reason, [dict(m) for m in agent.context.messages]))
        agent._session_revision_id = "revision-from-item"

    agent.save_session = save_session

    def get_response():
        agent.context.add({"role": "assistant", "content": "item work"})
        LLMAgent._snapshot_before_compaction(agent)
        return "done"

    agent.get_response = get_response

    agent.run_isolated_turn("caption a.png")

    assert saved == []
    assert agent._session_revision_id == "outer-revision"

    # The same agent still snapshots normally once the turn is over.
    LLMAgent._snapshot_before_compaction(agent)
    assert [reason for reason, _ in saved] == ["pre-compaction"]


def test_isolated_turn_output_is_neither_rendered_nor_published(capsys):
    events = EventHub()
    console = AgentConsole(events)
    agent = _agent(console=console)
    agent.get_response = lambda: (
        console.response("per-item result nobody asked to see"),
        console.tool_result("12 lines read"),
        "done",
    )[-1]

    console.system("before the batch")
    baseline = events.latest_seq
    capsys.readouterr()

    response, _ = agent.run_isolated_turn("item")

    assert response == "done"
    assert events.after(baseline) == []
    assert capsys.readouterr().out == ""
    # Suppression is scoped to the turn, not sticky.
    console.system("after the batch")
    assert [event.type for event in events.after(baseline)] == ["system"]
    assert "after the batch" in capsys.readouterr().out


def _skill_dir(tmp_path, name="demo", body="DEMO INSTRUCTIONS", tools=None):
    directory = tmp_path / "skills" / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: d\n---\n\n{body}\n", encoding="utf-8"
    )
    if tools is not None:
        (directory / "tools.py").write_text(tools, encoding="utf-8")
    return {name: {"body": body, "dir": str(directory), "description": "d"}}


def test_isolated_turn_restores_loaded_skills(tmp_path):
    """A skill loaded by one item must not silently un-instruct the next one."""
    skills = _skill_dir(tmp_path)
    agent = _agent()
    executor = _executor(agent, tmp_path, skills=skills)

    agent.get_response = lambda: executor.execute("load_skill", {"name": "demo"})

    first = agent.run_isolated_turn("item 1")[0]
    second = agent.run_isolated_turn("item 2")[0]

    # Every item gets the instructions even though the executor tracks loads.
    assert "DEMO INSTRUCTIONS" in first["content"]
    assert "DEMO INSTRUCTIONS" in second["content"]
    # And the enclosing conversation can still obtain them itself.
    assert executor._loaded_skills == set()
    assert "DEMO INSTRUCTIONS" in executor.execute("load_skill", {"name": "demo"})["content"]


def test_isolated_turn_restores_skill_tools_and_load_counts(tmp_path):
    """Rollback covers the registry too, not just the loaded-skill names."""
    skills = _skill_dir(
        tmp_path,
        tools=(
            "TOOLS = [{\n"
            "    'run': lambda executor: {'success': True},\n"
            "    'schema': {'type': 'function', 'function': {\n"
            "        'name': 'demo_tool', 'description': 'd',\n"
            "        'parameters': {'type': 'object', 'properties': {}},\n"
            "    }},\n"
            "}]\n"
        ),
    )
    agent = _agent()
    executor = _executor(agent, tmp_path, skills=skills)
    agent.get_response = lambda: executor.execute("load_skill", {"name": "demo"})

    agent.run_isolated_turn("item")

    # An item's skill must not keep advertising its tools to the conversation.
    assert executor.registry.get("demo_tool") is None
    assert executor.skill_tool_schemas() == []
    # Nor inflate the session's usage telemetry with per-item loads.
    assert executor._skill_loads == {}


def test_isolated_turn_keeps_skills_the_conversation_loaded(tmp_path):
    """Rollback restores the enclosing state — it does not reset to empty."""
    skills = _skill_dir(tmp_path)
    agent = _agent()
    executor = _executor(agent, tmp_path, skills=skills)
    executor.execute("load_skill", {"name": "demo"})

    agent.get_response = lambda: "done"
    agent.run_isolated_turn("item")

    assert executor._loaded_skills == {"demo"}
    assert executor._skill_loads == {"demo": 1}


def test_isolated_turn_can_load_a_skill_the_conversation_already_loaded(tmp_path):
    """An item starts from an empty history, so it must get real instructions."""
    skills = _skill_dir(tmp_path)
    agent = _agent()
    executor = _executor(agent, tmp_path, skills=skills)
    executor.execute("load_skill", {"name": "demo"})

    agent.get_response = lambda: executor.execute("load_skill", {"name": "demo"})

    result = agent.run_isolated_turn("item")[0]

    # Reloading returns the body because this isolated item has no prior history.
    assert "DEMO INSTRUCTIONS" in result["content"]


def test_isolated_turn_restores_skills_after_a_failing_item(tmp_path):
    skills = _skill_dir(tmp_path)
    agent = _agent()
    executor = _executor(agent, tmp_path, skills=skills)

    def get_response():
        executor.execute("load_skill", {"name": "demo"})
        raise RuntimeError("item blew up")

    agent.get_response = get_response

    with pytest.raises(RuntimeError):
        agent.run_isolated_turn("item")

    assert executor._loaded_skills == set()


def test_isolated_turn_still_reports_errors(capsys):
    """An item reports only "no response"; the error is the sole diagnosis."""
    events = EventHub()
    console = AgentConsole(events)
    agent = _agent(console=console)
    agent.get_response = lambda: console.error("API call failed: quota exhausted")

    baseline = events.latest_seq
    capsys.readouterr()

    response, _ = agent.run_isolated_turn("item")

    assert response is None
    assert "quota exhausted" in capsys.readouterr().out
    assert [event.type for event in events.after(baseline)] == ["error"]


def test_isolated_turns_do_not_share_a_prompt_cache_key_with_the_conversation():
    requests = []
    agent = NS(
        context_length=0,
        model="test-model",
        max_output_tokens=100,
        INITIAL_BACKOFF=0.01,
        MAX_BACKOFF=0.02,
        verbose=False,
        round_id=1,
        _session_id="20240101_000000",
        _isolated_turn_active=False,
        _pending_images=[],
        _messages_with_pending_images=lambda: [],
        stream=False,
        tools=[],
        profile=NS(reasoning=None),
        reasoning_effort="high",
        cancellation=None,
        _accumulate_usage=lambda usage: None,
        token_estimator=NS(observe=lambda *args: None),
        context=NS(add=lambda message: None),
        _blocking_completion=lambda request: (
            requests.append(request),
            NS(message={"role": "assistant", "content": "ok"}, usage=None, finish_reason="stop"),
        )[-1],
        _estimate_usage=lambda message: NS(
            prompt_tokens=1, completion_tokens=1, total_tokens=2,
            reasoning_tokens=None, cached_prompt_tokens=None,
        ),
    )

    LLMAgent.call_api(agent)
    agent._isolated_turn_active = True
    LLMAgent.call_api(agent)

    outer, isolated = (request.session_id for request in requests)
    assert outer == "20240101_000000"
    assert isolated != outer


def test_steering_is_suppressed_during_an_isolated_turn():
    submission = NS(text="a steering message", steer=True, id="s1", source="web")
    consumed = []
    agent = NS(
        _isolated_turn_active=True,
        input_broker=NS(
            submission=submission,
            get_nowait=lambda sid=None: consumed.append(sid),
        ),
    )

    assert LLMAgent._inject_pending_steer(agent) is False
    assert consumed == []  # the message stays pending for the enclosing round


# ----- batch skill ----------------------------------------------------------

def _executor(agent, tmp_path, console=None, skills=None):
    executor = ToolExecutor(
        console=console or agent.console,
        work_dir=str(tmp_path),
        skills=skills,
        isolated_turn=agent.run_isolated_turn,
    )
    # One agent owns one executor, as LLMAgent does: the isolated turn must roll
    # back skill state on the very executor its items call tools through.
    agent.tool_executor = executor
    return executor


def _output(tmp_path, name="run"):
    """Where the skill writes results for *name* (protected .kia/batch entry)."""
    return tmp_path / ".kia" / "batch" / f"{name}.jsonl"


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_batch_returns_only_a_summary(tmp_path):
    entry = _load_batch_tools()
    agent = _agent([f"caption {i}" for i in range(3)])
    executor = _executor(agent, tmp_path)

    result = entry["run"](
        executor,
        task="Describe {item}",
        items=["a.png", "b.png", "c.png"],
        name="captions",
    )

    assert result["success"]
    assert (result["succeeded"], result["failed"], result["total"]) == (3, 0, 3)
    # The whole point: no per-item result reaches the conversation.
    blob = json.dumps(result)
    assert "caption 0" not in blob and "caption 2" not in blob
    assert "captions.jsonl" in result["message"]

    records = _records(_output(tmp_path, "captions"))
    assert [r["item"] for r in records] == ["a.png", "b.png", "c.png"]
    assert [r["result"] for r in records] == ["caption 0", "caption 1", "caption 2"]
    assert all(r["ok"] for r in records)


def test_batch_leaves_the_conversation_untouched(tmp_path):
    entry = _load_batch_tools()
    agent = _agent(["x", "y"])
    agent.context.add({"role": "user", "content": "prior"})
    before = [dict(m) for m in agent.context.messages]
    executor = _executor(agent, tmp_path)

    entry["run"](executor, task="Do {item}", items=["1", "2"], name="run")

    assert agent.context.messages == before
    for seen in agent.seen:
        assert len(seen) == 1


def test_batch_substitutes_item_and_appends_when_no_placeholder(tmp_path):
    entry = _load_batch_tools()
    agent = _agent(["ok", "ok"])
    executor = _executor(agent, tmp_path)

    entry["run"](
        executor, task="Caption {item} as JSON {\"a\": 1}", items=["p.png"], name="a"
    )
    entry["run"](executor, task="Summarize", items=["doc.txt"], name="b")

    assert agent.seen[0][-1]["content"] == 'Caption p.png as JSON {"a": 1}'
    assert agent.seen[1][-1]["content"] == "Summarize\n\nItem: doc.txt"


def test_batch_reads_items_file_skipping_blanks_and_comments(tmp_path):
    entry = _load_batch_tools()
    (tmp_path / "items.txt").write_text("a\n\n# skip me\nb\n", encoding="utf-8")
    agent = _agent(["1", "2"])
    executor = _executor(agent, tmp_path)

    result = entry["run"](
        executor, task="Do {item}", items_file="items.txt", name="run"
    )

    assert result["total"] == 2
    assert [r["item"] for r in _records(_output(tmp_path))] == ["a", "b"]


def test_batch_resume_skips_successes_and_retries_failures(tmp_path):
    entry = _load_batch_tools()
    agent = _agent([None, "recovered", "fresh"])
    executor = _executor(agent, tmp_path)
    output = _output(tmp_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"item": "a", "index": 1, "ok": True, "result": "kept"}) + "\n"
        + json.dumps({"item": "b", "index": 2, "ok": False, "error": "boom"}) + "\n"
        + '{"item": "c", "ok": tru',  # torn final line from a crash
        encoding="utf-8",
    )

    result = entry["run"](
        executor, task="Do {item}", items=["a", "b", "c"], name="run"
    )

    assert result["skipped"] == 1
    # 'b' is retried (it failed) and 'c' is attempted (its record was torn).
    assert [seen[-1]["content"] for seen in agent.seen] == ["Do b", "Do c"]
    assert result["succeeded"] == 1 and result["failed"] == 1


def test_batch_resume_disabled_reprocesses_everything(tmp_path):
    entry = _load_batch_tools()
    agent = _agent(["again"])
    executor = _executor(agent, tmp_path)
    output = _output(tmp_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"item": "a", "ok": True, "result": "kept"}) + "\n", encoding="utf-8"
    )

    result = entry["run"](
        executor, task="Do {item}", items=["a"], name="run", resume=False
    )

    assert result["skipped"] == 0 and result["succeeded"] == 1
    records = _records(output)
    assert len(records) == 1
    assert records[0]["result"] == "again"


def test_batch_restart_keeps_the_previous_results(tmp_path):
    """resume=False must not be able to destroy a finished run's deliverable."""
    entry = _load_batch_tools()
    agent = _agent([RuntimeError("bad task")])
    executor = _executor(agent, tmp_path)
    output = _output(tmp_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = json.dumps({"item": "a", "ok": True, "result": "kept"}) + "\n"
    output.write_text(previous, encoding="utf-8")

    result = entry["run"](
        executor, task="Do {item}", items=["a"], name="run", resume=False
    )

    backup = output.with_name(output.name + ".bak")
    assert backup.read_text(encoding="utf-8") == previous
    assert result["output"] in result["message"] and str(backup.name) in result["message"]
    assert [record["ok"] for record in _records(output)] == [False]


def test_batch_record_index_is_the_item_position_across_resumes(tmp_path):
    entry = _load_batch_tools()
    agent = _agent(["one", None, "three", "two"])
    executor = _executor(agent, tmp_path)

    entry["run"](executor, task="Do {item}", items=["a", "b", "c"], name="run")
    entry["run"](executor, task="Do {item}", items=["a", "b", "c"], name="run")

    records = _records(_output(tmp_path))
    # 'b' failed first time and is retried; its index still names its position.
    assert [(r["item"], r["index"]) for r in records] == [
        ("a", 1), ("b", 2), ("c", 3), ("b", 2)
    ]


def test_batch_fresh_failure_is_not_masked_by_an_old_success(tmp_path):
    entry = _load_batch_tools()
    agent = _agent([None, "recovered"])
    executor = _executor(agent, tmp_path)
    output = _output(tmp_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"item": "a", "ok": True, "result": "old"}) + "\n",
        encoding="utf-8",
    )

    fresh = entry["run"](
        executor, task="Do {item}", items=["a"], name="run", resume=False
    )
    resumed = entry["run"](
        executor, task="Do {item}", items=["a"], name="run"
    )

    assert fresh["failed"] == 1
    assert resumed["skipped"] == 0 and resumed["succeeded"] == 1
    assert [record["result"] for record in _records(output)] == [None, "recovered"]


def test_batch_all_items_already_done_is_a_no_op(tmp_path):
    entry = _load_batch_tools()
    agent = _agent(["unused"])
    executor = _executor(agent, tmp_path)
    output = _output(tmp_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"item": "a", "ok": True, "result": "kept"}) + "\n", encoding="utf-8"
    )

    result = entry["run"](executor, task="Do {item}", items=["a"], name="run")

    assert result["success"] and result["skipped"] == 1
    assert agent.seen == []


def test_batch_survives_a_failing_item(tmp_path):
    entry = _load_batch_tools()
    agent = _agent(["fine", RuntimeError("bad item"), "fine again"])
    executor = _executor(agent, tmp_path)

    result = entry["run"](
        executor, task="Do {item}", items=["a", "b", "c"], name="run"
    )

    assert (result["succeeded"], result["failed"]) == (2, 1)
    assert result["failures"] == [{"item": "b", "error": "RuntimeError: bad item"}]
    records = _records(_output(tmp_path))
    assert [r["ok"] for r in records] == [True, False, True]


def test_batch_aborts_early_when_nothing_succeeds(tmp_path):
    entry = _load_batch_tools()
    agent = _agent([RuntimeError("broken template")] * 10)
    executor = _executor(agent, tmp_path)

    result = entry["run"](
        executor,
        task="Do {item}",
        items=[str(i) for i in range(10)],
        name="run",
    )

    assert result["failed"] == 3
    assert len(agent.seen) == 3
    assert "aborted" in result["message"]


def test_batch_interruption_stops_and_reports_partial_progress(tmp_path):
    entry = _load_batch_tools()
    agent = _agent(["one", _interrupted(), "never runs"])
    executor = _executor(agent, tmp_path)

    result = entry["run"](
        executor, task="Do {item}", items=["a", "b", "c"], name="run"
    )

    assert result["success"] is True and result["interrupted"] is True
    assert result["succeeded"] == 1
    assert "interrupted" in result["message"]
    # The interrupted item is left unrecorded so a resumed run retries it.
    assert [r["item"] for r in _records(_output(tmp_path))] == ["a"]


def test_batch_shows_progress_per_item(tmp_path):
    entry = _load_batch_tools()
    console = _Console()
    agent = _agent(["a", "b"])
    executor = _executor(agent, tmp_path, console=console)

    entry["run"](
        executor, task="Do {item}", items=["1", "2"], name="run", label="Captioning"
    )

    # One indicator for the whole run, updated per item: an indicator per item
    # would publish a start/stop event pair each time and evict the real
    # timeline from the bounded reconnect history.
    assert len(console.indicators) == 1
    assert console.indicators[0].suffixes == ["0/2", "1/2", "2/2"]
    assert [c["label"] for c in console.thinking_calls] == ["Captioning"]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"task": "", "items": ["a"], "name": "run"}, "task is required"),
        ({"task": "t", "items": ["a"]}, "name is required"),
        # A name is a run identifier, never a path: traversal and separators
        # must not be able to steer results outside the managed directory.
        ({"task": "t", "items": ["a"], "name": "../escape"}, "name is required"),
        ({"task": "t", "items": ["a"], "name": "sub/dir"}, "name is required"),
        ({"task": "t", "items": ["a"], "name": ".hidden"}, "name is required"),
        ({"task": "t", "items": ["a"], "name": "x" * 65}, "name is required"),
        ({"task": "t", "name": "run"}, "exactly one of items or items_file"),
        (
            {"task": "t", "items": ["a"], "items_file": "f.txt", "name": "run"},
            "exactly one of items or items_file",
        ),
        (
            {"task": "t", "items_file": "missing.txt", "name": "run"},
            "items_file not found",
        ),
        (
            {"task": "t", "items": [str(i) for i in range(101)], "name": "run"},
            "limited to 100 entries",
        ),
    ],
)
def test_batch_rejects_bad_arguments_before_running(tmp_path, kwargs, message):
    entry = _load_batch_tools()
    agent = _agent()
    executor = _executor(agent, tmp_path)

    result = entry["run"](executor, **kwargs)

    assert result["success"] is False
    assert message in result["error"]
    assert agent.seen == []


def test_batch_requires_an_agent_backed_executor(tmp_path):
    entry = _load_batch_tools()
    executor = ToolExecutor(console=_Console(), work_dir=str(tmp_path))

    result = entry["run"](executor, task="Do {item}", items=["a"], name="run")

    assert result["success"] is False
    assert "not available" in result["error"]


def test_batch_results_land_in_the_managed_directory(tmp_path):
    entry = _load_batch_tools()
    agent = _agent(["done"])
    executor = _executor(agent, tmp_path)

    result = entry["run"](executor, task="Do {item}", items=["a"], name="my.run-1")

    expected = tmp_path / ".kia" / "batch" / "my.run-1.jsonl"
    assert expected.is_file()
    # The reported path is relative to the working directory, so the model can
    # hand it straight to read_file / grep_files.
    assert result["output"] == str(Path(".kia") / "batch" / "my.run-1.jsonl")
    assert executor._resolve_path(result["output"]) == expected


def test_batch_results_survive_a_default_storage_clean(tmp_path):
    """`kia --clean` must not destroy results a run is still working from."""
    entry = _load_batch_tools()
    agent = _agent(["done"])
    executor = _executor(agent, tmp_path)
    entry["run"](executor, task="Do {item}", items=["a"], name="keep")
    # Something a clean is genuinely meant to reclaim.
    (tmp_path / ".kia" / "tool-results").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".kia" / "tool-results" / "old.txt").write_text("junk")

    clean_storage(tmp_path)

    assert _output(tmp_path, "keep").is_file()
    assert not (tmp_path / ".kia" / "tool-results").exists()


def test_batch_tool_registers_without_colliding(tmp_path):
    entry = _load_batch_tools()
    registry = ToolRegistry()

    registry.register_skill("batch", [entry])

    spec = registry.get("run_batch")
    assert spec is not None
    description = spec.describe({
        "task": "Describe the image",
        "items_file": "items.txt",
        "name": "captions",
    })
    assert "items.txt" in description.text and "captions" in description.text

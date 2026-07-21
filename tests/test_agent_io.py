import asyncio
import json
import threading
import time
from pathlib import Path

import pytest

from kiui.agent.utils.io import (
    EVENT_TEXT_LIMIT,
    CancellationToken,
    EventHub,
    InputBroker,
    PromptBroker,
    sanitize_unicode,
)
from kiui.agent.backend import LLMAgent
from kiui.agent.utils.interrupt import RequestInterrupted, run_interruptible
from kiui.agent.ui import AgentConsole


def test_event_hub_orders_and_bounds_replay():
    hub = EventHub(max_events=2)
    hub.publish("system", text="one")
    second = hub.publish("system", text="two")
    third = hub.publish("system", text="three")

    assert [event.seq for event in hub.after(0)] == [second.seq, third.seq]
    assert hub.latest_seq == third.seq
    assert hub.oldest_seq == second.seq
    assert hub.has_replay_gap(0)
    assert not hub.has_replay_gap(second.seq - 1)


def test_sanitize_unicode_combines_pairs_and_replaces_lone_surrogates():
    value = {"text": "a\ud83d\ude00b\ud800c", "nested": ["\udc00"]}

    assert sanitize_unicode(value) == {"text": "a😀b�c", "nested": ["�"]}


def test_event_hub_sanitizes_invalid_surrogates():
    hub = EventHub()

    event = hub.publish("output", text="a\ud83d\ude00b\ud800c")

    assert event.data["text"] == "a😀b�c"
    assert event.data["text"].encode("utf-8") == b"a\xf0\x9f\x98\x80b\xef\xbf\xbdc"


def test_event_hub_clips_oversized_text_fields():
    from kiui.agent.utils.io import EVENT_TEXT_LIMIT

    hub = EventHub()
    event = hub.publish("output", text="x" * (EVENT_TEXT_LIMIT + 100), count=3)

    assert len(event.data["text"]) < EVENT_TEXT_LIMIT + 100
    assert "truncated, 100 more characters" in event.data["text"]
    assert event.data["count"] == 3  # non-str fields untouched


def test_input_broker_sanitizes_invalid_surrogates():
    hub = EventHub()
    broker = InputBroker(hub)

    item = broker.submit("a\ud83d\ude00b\ud800c")

    assert item.text == "a😀b�c"


def test_input_broker_accepts_only_one_pending_submission():
    hub = EventHub()
    broker = InputBroker(hub)
    first = broker.submit("first")

    with pytest.raises(ValueError, match="already pending"):
        broker.submit("second")

    assert broker.get_nowait() == first
    assert [(e.type, e.data.get("reason")) for e in hub.after(0)] == [
        ("pending_set", None),
        ("pending_cleared", "consumed"),
    ]


def test_input_broker_serializes_state_and_events():
    class BlockingEventHub(EventHub):
        def __init__(self):
            super().__init__()
            self.publishing = threading.Event()
            self.release = threading.Event()

        def publish(self, event_type, **data):
            if event_type == "pending_set":
                self.publishing.set()
                assert self.release.wait(timeout=2)
            return super().publish(event_type, **data)

    hub = BlockingEventHub()
    broker = InputBroker(hub)
    submitted = threading.Thread(target=lambda: broker.submit("first"))
    submitted.start()
    assert hub.publishing.wait(timeout=1)

    withdrawn = threading.Thread(target=broker.withdraw)
    withdrawn.start()
    time.sleep(0.05)
    assert withdrawn.is_alive()

    hub.release.set()
    submitted.join(timeout=1)
    withdrawn.join(timeout=1)
    assert [event.type for event in hub.after(0)] == [
        "pending_set", "pending_cleared"
    ]
    assert broker.pending == 0


def test_input_broker_rejects_messages_that_events_would_truncate():
    hub = EventHub()
    broker = InputBroker(hub)

    with pytest.raises(ValueError, match="too large"):
        broker.submit("é" * (EVENT_TEXT_LIMIT // 2 + 1))


def test_input_broker_notifies_listeners():
    hub = EventHub()
    broker = InputBroker(hub)
    notified = threading.Event()
    broker.add_listener(notified.set)

    broker.submit("first")
    assert notified.wait(timeout=1)

    notified.clear()
    broker.get_nowait()
    assert notified.wait(timeout=1)

    broker.remove_listener(notified.set)


def test_input_broker_withdraws_pending_submission_for_editing():
    hub = EventHub()
    broker = InputBroker(hub)
    first = broker.submit("first")

    assert broker.withdraw("wrong") is None
    assert broker.withdraw(first.id) == first
    assert broker.pending == 0
    assert hub.after(0)[-1].data["reason"] == "withdrawn"


def test_prompt_broker_first_valid_answer_wins():
    hub = EventHub()
    broker = PromptBroker(hub)
    result = []

    thread = threading.Thread(
        target=lambda: result.append(
            broker.ask("select", "Continue?", choices=["Yes", "No"])
        )
    )
    thread.start()
    deadline = time.time() + 2
    while broker.active is None and time.time() < deadline:
        time.sleep(0.01)

    prompt = broker.active
    assert prompt is not None
    assert not broker.resolve(prompt.id, "invalid")
    assert broker.resolve(prompt.id, "Yes")
    assert not broker.resolve(prompt.id, "No")
    thread.join(timeout=2)
    assert result == ["Yes"]


def test_cancellation_is_operation_scoped():
    hub = EventHub()
    token = CancellationToken(hub)
    operation_id = token.begin("test")

    assert not token.cancel("stale")
    assert token.cancel(operation_id)
    assert token.cancelled
    token.finish(operation_id)
    assert not token.cancelled
    assert token.operation_id is None


def test_web_input_can_win_over_pending_terminal_prompt():
    hub = EventHub()
    broker = InputBroker(hub)
    agent = object.__new__(LLMAgent)
    agent.input_broker = broker

    class FakeTerminal:
        def __init__(self):
            self.cancelled = False

        def prompt(self):
            # Block long enough for the web message to arrive first.
            time.sleep(2)
            self.cancelled = True
            return ""

        async def prompt_async(self):
            # Yield to the loop so the web poller can win the race.
            await asyncio.sleep(2)
            self.cancelled = True
            return ""

    terminal = FakeTerminal()
    threading.Timer(0.05, lambda: broker.submit("from phone")).start()
    submission = agent._next_submission(terminal)
    assert submission.text == "from phone"


def test_run_interruptible_observes_web_cancellation():
    hub = EventHub()
    cancellation = CancellationToken(hub)
    operation_id = cancellation.begin("slow")
    threading.Timer(0.05, lambda: cancellation.cancel(operation_id)).start()

    with pytest.raises(RequestInterrupted):
        run_interruptible(lambda: time.sleep(1), cancellation)


def test_console_web_output_strips_rich_markup():
    hub = EventHub()
    console = AgentConsole(events=hub)
    console.print("[bold blue]Available commands:[/bold blue]")
    console.user_input(
        "hello", source="web", submission_id="abc123", with_rule=False
    )

    events = hub.after(0)
    assert [event.type for event in events] == ["output", "user_message"]
    assert events[0].data["text"] == "Available commands:"
    assert events[1].data == {
        "text": "hello", "source": "web", "submission_id": "abc123"
    }


def test_console_table_emits_rendered_text_not_repr():
    from rich.table import Table

    hub = EventHub()
    console = AgentConsole(events=hub)
    table = Table(title="models")
    table.add_column("name")
    table.add_row("gpt-x")
    console.table(table)

    text = hub.after(0)[-1].data["text"]
    assert "gpt-x" in text
    assert "rich.table.Table object" not in text


def test_event_hub_wait_after_wakes_on_publish():
    hub = EventHub()
    threading.Timer(0.05, lambda: hub.publish("system", text="ping")).start()

    start = time.time()
    events = hub.wait_after(0, timeout=2.0)
    assert [event.data["text"] for event in events] == ["ping"]
    assert time.time() - start < 1.0  # woke on publish, not on timeout


def test_prompt_broker_queues_concurrent_asks():
    hub = EventHub()
    broker = PromptBroker(hub)
    answers = []

    def ask(message):
        answers.append((message, broker.ask("text", message)))

    first = threading.Thread(target=ask, args=("first",))
    second = threading.Thread(target=ask, args=("second",))
    first.start()
    deadline = time.time() + 2
    while broker.active is None and time.time() < deadline:
        time.sleep(0.01)
    second.start()
    time.sleep(0.1)  # give the second ask time to block on the first

    assert broker.active is not None and broker.active.message == "first"
    assert broker.resolve(broker.active.id, "one")
    deadline = time.time() + 2
    while (
        broker.active is None or broker.active.message != "second"
    ) and time.time() < deadline:
        time.sleep(0.01)
    assert broker.resolve(broker.active.id, "two")
    first.join(timeout=2)
    second.join(timeout=2)
    assert sorted(answers) == [("first", "one"), ("second", "two")]


def test_prompt_broker_accepts_terminal_async_adapter():
    hub = EventHub()
    broker = PromptBroker(hub)

    async def choose_yes(_prompt):
        await asyncio.sleep(0)
        return "Yes"

    broker.set_terminal_adapter(choose_yes)
    assert broker.ask("select", "Continue?", choices=["Yes", "No"]) == "Yes"
    resolved = [event for event in hub.after(0) if event.type == "prompt_resolved"]
    assert resolved[-1].data["source"] == "terminal"


def test_prompt_broker_ask_inside_running_event_loop():
    hub = EventHub()
    broker = PromptBroker(hub)

    async def choose_yes(_prompt):
        await asyncio.sleep(0)
        return "Yes"

    async def invoke_sync_api():
        # Regression: this used to call asyncio.run() on this active loop.
        return broker.ask("select", "Continue?", choices=["Yes", "No"])

    broker.set_terminal_adapter(choose_yes)
    assert asyncio.run(invoke_sync_api()) == "Yes"


def test_web_answer_cancels_pending_terminal_adapter():
    hub = EventHub()
    broker = PromptBroker(hub)
    terminal_started = threading.Event()
    terminal_cancelled = threading.Event()
    result = []

    async def wait_on_terminal(_prompt):
        terminal_started.set()
        try:
            await asyncio.sleep(5)
        finally:
            terminal_cancelled.set()

    broker.set_terminal_adapter(wait_on_terminal)
    thread = threading.Thread(
        target=lambda: result.append(
            broker.ask("select", "Continue?", choices=["Yes", "No"])
        )
    )
    thread.start()
    assert terminal_started.wait(timeout=2)

    prompt = broker.active
    assert prompt is not None
    assert broker.resolve(prompt.id, "No", source="web")
    thread.join(timeout=2)
    assert result == ["No"]
    assert terminal_cancelled.is_set()


def test_operation_context_always_finishes():
    hub = EventHub()
    cancellation = CancellationToken(hub)

    with pytest.raises(RuntimeError):
        with cancellation.operation("failing"):
            raise RuntimeError("boom")

    assert cancellation.operation_id is None
    assert [event.type for event in hub.after(0)] == [
        "operation_start",
        "operation_end",
    ]


def test_cancellation_releases_active_prompt():
    events = EventHub()
    prompts = PromptBroker(events)
    cancellation = CancellationToken(events, prompts)
    operation_id = cancellation.begin("tool")
    result = []
    thread = threading.Thread(
        target=lambda: result.append(prompts.ask("confirm", "Run tool?"))
    )
    thread.start()
    for _ in range(100):
        if prompts.active is not None:
            break
        time.sleep(0.01)

    assert cancellation.cancel(operation_id)
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert result == [None]


def test_change_tracker_close_persists_resumable_log(tmp_path, monkeypatch):
    from kiui.agent.utils.rewind import ChangeTracker

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    tracker = ChangeTracker("session", work_dir, AgentConsole())
    tracker.track_write(1, "created.txt", "new content")

    tracker.close()

    log_path = tmp_path / ".kia" / "rewind" / "session" / "change_log.json"
    records = json.loads(log_path.read_text(encoding="utf-8"))
    assert records[0]["path"] == "created.txt"

import asyncio
import json
import queue
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
    parse_wait_command,
    sanitize_unicode,
)
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


@pytest.mark.parametrize(
    "text, seconds, prompt",
    [
        ("/wait 30s check status", 30, "check status"),
        ("/WAIT 2min do it", 120, "do it"),
        ("/wait 1.5hours finish task", 5400, "finish task"),
    ],
)
def test_parse_wait_command(text, seconds, prompt):
    assert parse_wait_command(text) == (seconds, prompt)


@pytest.mark.parametrize("text", ["/wait", "/wait 1d prompt", "/wait 0s prompt"])
def test_parse_wait_command_rejects_invalid_input(text):
    with pytest.raises(ValueError):
        parse_wait_command(text)


def test_wait_submission_is_hidden_until_ready_and_is_non_steering():
    hub = EventHub()
    broker = InputBroker(hub)
    ready = threading.Event()
    broker.add_listener(ready.set)

    submission = broker.submit("/wait 0.05s check status", source="web")
    assert submission.text == "check status"
    assert not submission.steer
    assert not submission.ready
    assert not broker.ready
    with pytest.raises(queue.Empty):
        broker.get_nowait()

    ready.clear()  # discard the immediate pending-set notification
    assert ready.wait(1)
    assert broker.ready
    assert broker.get_nowait() == submission


def test_withdrawing_wait_submission_cancels_ready_notification():
    broker = InputBroker(EventHub())
    notified = threading.Event()
    broker.add_listener(notified.set)
    submission = broker.submit("/wait 0.05s later", source="terminal")
    notified.clear()

    assert broker.withdraw() == submission
    notified.clear()  # discard the withdrawal notification
    assert not notified.wait(0.1)


def test_input_broker_consumes_only_the_expected_submission():
    hub = EventHub()
    broker = InputBroker(hub)
    first = broker.submit("first")

    with pytest.raises(queue.Empty):
        broker.get_nowait("different-id")
    assert broker.submission == first
    assert broker.get_nowait(first.id) == first
    assert broker.submission is None
    assert [(e.type, e.data.get("reason")) for e in hub.after(0)] == [
        ("pending_set", None),
        ("pending_cleared", "consumed"),
    ]


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

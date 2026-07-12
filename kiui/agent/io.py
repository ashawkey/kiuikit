"""Thread-safe shared I/O primitives for terminal and web clients."""

from __future__ import annotations

import asyncio
import queue
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable


# Cap per-field payload of published events so the EventHub's replay
# history stays bounded even for large file writes or tool outputs.
EVENT_TEXT_LIMIT = 64 * 1024


def _clip_text(text: str, limit: int = EVENT_TEXT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... (truncated, {len(text) - limit} more characters)"


@dataclass(frozen=True)
class AgentEvent:
    seq: int
    type: str
    data: dict[str, Any]
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventHub:
    """Publish ordered events and retain a bounded reconnect history."""

    def __init__(self, max_events: int = 2000):
        self.stream_id = uuid.uuid4().hex
        self._events: deque[AgentEvent] = deque(maxlen=max_events)
        self._cond = threading.Condition()
        self._seq = 0

    def publish(self, event_type: str, **data: Any) -> AgentEvent:
        data = {
            key: _clip_text(value) if isinstance(value, str) else value
            for key, value in data.items()
        }
        with self._cond:
            self._seq += 1
            event = AgentEvent(self._seq, event_type, data, time.time())
            self._events.append(event)
            self._cond.notify_all()
            return event

    def after(self, seq: int) -> list[AgentEvent]:
        with self._cond:
            return [event for event in self._events if event.seq > seq]

    def wait_after(self, seq: int, timeout: float = 1.0) -> list[AgentEvent]:
        """Block until events newer than *seq* exist (or *timeout*), then return them."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq > seq, timeout=timeout)
            return [event for event in self._events if event.seq > seq]

    @property
    def latest_seq(self) -> int:
        with self._cond:
            return self._seq


@dataclass(frozen=True)
class UserSubmission:
    text: str
    source: str
    id: str


class InputBroker:
    """FIFO queue shared by terminal and authenticated web clients."""

    def __init__(self, events: EventHub, max_pending: int = 20):
        self.events = events
        self.max_pending = max_pending
        self._queue: queue.Queue[UserSubmission] = queue.Queue(max_pending)

    def submit(self, text: str, source: str = "web") -> UserSubmission:
        text = text.strip()
        if not text:
            raise ValueError("Message cannot be empty.")
        if len(text.encode("utf-8")) > 128 * 1024:
            raise ValueError("Message is too large.")
        item = UserSubmission(text=text, source=source, id=uuid.uuid4().hex)
        try:
            self._queue.put_nowait(item)
        except queue.Full as exc:
            raise ValueError("The message queue is full.") from exc
        self.events.publish("queue_changed", pending=self._queue.qsize())
        return item

    def get_nowait(self) -> UserSubmission:
        item = self._queue.get(block=False)
        self.events.publish("queue_changed", pending=self._queue.qsize())
        return item

    @property
    def pending(self) -> int:
        return self._queue.qsize()


@dataclass
class ActivePrompt:
    id: str
    kind: str
    message: str
    choices: list[str]
    default: str
    done: threading.Event
    answer: str | None = None


# How often ``ask`` polls the terminal task for completion while a web
# client may answer concurrently.
PROMPT_POLL_INTERVAL = 0.05


class PromptBroker:
    """Synchronize interactive prompts; the first valid answer wins."""

    def __init__(self, events: EventHub):
        self.events = events
        self._cond = threading.Condition()
        self._active: ActivePrompt | None = None
        self._terminal_asker: (
            Callable[[ActivePrompt], Awaitable[str | None]] | None
        ) = None

    def set_terminal_adapter(
        self,
        asker: Callable[[ActivePrompt], Awaitable[str | None]],
    ) -> None:
        self._terminal_asker = asker

    @property
    def active(self) -> ActivePrompt | None:
        with self._cond:
            return self._active

    def ask(
        self,
        kind: str,
        message: str,
        choices: list[str] | None = None,
        default: str = "",
    ) -> str | None:
        prompt = ActivePrompt(
            id=uuid.uuid4().hex,
            kind=kind,
            message=message,
            choices=list(choices or []),
            default=default,
            done=threading.Event(),
        )
        with self._cond:
            # Queue behind any active prompt (e.g. a concurrent subagent ask)
            # instead of failing the caller.
            while self._active is not None:
                self._cond.wait()
            self._active = prompt
        self.events.publish(
            "prompt_open",
            id=prompt.id,
            kind=kind,
            message=message,
            choices=prompt.choices,
            default=default,
        )

        if self._terminal_asker is not None:
            async def ask_both() -> None:
                if prompt.done.is_set():
                    return
                terminal_task = asyncio.create_task(
                    self._terminal_asker(prompt)
                )
                while not prompt.done.is_set() and not terminal_task.done():
                    await asyncio.sleep(PROMPT_POLL_INTERVAL)
                if prompt.done.is_set():
                    terminal_task.cancel()
                    try:
                        await terminal_task
                    except asyncio.CancelledError:
                        pass
                    return
                try:
                    answer = terminal_task.result()
                except (EOFError, KeyboardInterrupt):
                    answer = None
                if answer is None:
                    self.cancel(prompt.id, source="terminal")
                else:
                    self.resolve(prompt.id, answer, source="terminal")

            try:
                asyncio.run(ask_both())
            except BaseException:
                self.cancel(prompt.id, source="terminal")
                raise

        if not prompt.done.is_set():
            prompt.done.wait()
        return prompt.answer

    def resolve(self, prompt_id: str, answer: str, source: str = "web") -> bool:
        with self._cond:
            prompt = self._active
            if prompt is None or prompt.id != prompt_id or prompt.done.is_set():
                return False
            if prompt.kind == "select" and answer not in prompt.choices:
                return False
            prompt.answer = answer
            prompt.done.set()
            self._active = None
            self._cond.notify_all()
        self.events.publish(
            "prompt_resolved", id=prompt_id, answer=answer, source=source
        )
        return True

    def cancel(self, prompt_id: str, source: str = "terminal") -> bool:
        with self._cond:
            prompt = self._active
            if prompt is None or prompt.id != prompt_id or prompt.done.is_set():
                return False
            prompt.answer = None
            prompt.done.set()
            self._active = None
            self._cond.notify_all()
        self.events.publish(
            "prompt_resolved", id=prompt_id, answer=None, source=source
        )
        return True


class CancellationToken:
    """Operation-scoped cancellation shared by terminal and web controls."""

    def __init__(self, events: EventHub):
        self.events = events
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._operation_id: str | None = None

    def begin(self, label: str) -> str:
        with self._lock:
            self._event.clear()
            self._operation_id = uuid.uuid4().hex
            operation_id = self._operation_id
        self.events.publish("operation_start", id=operation_id, label=label)
        return operation_id

    @contextmanager
    def operation(self, label: str):
        """Publish and clean up an operation even when its body fails."""
        operation_id = self.begin(label)
        try:
            yield operation_id
        finally:
            self.finish(operation_id)

    def finish(self, operation_id: str) -> None:
        with self._lock:
            if self._operation_id != operation_id:
                return
            self._operation_id = None
            self._event.clear()
        self.events.publish("operation_end", id=operation_id)

    def cancel(self, operation_id: str | None = None) -> bool:
        with self._lock:
            if self._operation_id is None:
                return False
            if operation_id is not None and operation_id != self._operation_id:
                return False
            active = self._operation_id
            self._event.set()
        self.events.publish("operation_cancelled", id=active)
        return True

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def operation_id(self) -> str | None:
        with self._lock:
            return self._operation_id

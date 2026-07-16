"""Agent-side client that links a terminal ``kia`` to the shared web hub.

A terminal ``kia`` process stays terminal-first. This client runs in a daemon
thread, connects to the hub's loopback ``/internal/agent`` endpoint, registers
a session, forwards every :class:`~kiui.agent.io.EventHub` event, and injects
browser actions back into the same brokers the terminal uses — so terminal and
web stay perfectly in sync.
"""

from __future__ import annotations

import asyncio
import json
import threading

from kiui.agent.io import (
    CancellationToken,
    EventHub,
    InputBroker,
    PromptBroker,
)


RECONNECT_DELAY = 2.0       # seconds between reconnect attempts
EVENT_WAIT_TIMEOUT = 1.0    # max block per EventHub.wait_after call
WS_MAX_SIZE = 256 * 1024


class HubClient:
    def __init__(
        self,
        events: EventHub,
        inputs: InputBroker,
        prompts: PromptBroker,
        cancellation: CancellationToken,
        *,
        host: str,
        port: int,
        secret: str,
        session_id: str,
        meta: dict,
    ):
        self.events = events
        self.inputs = inputs
        self.prompts = prompts
        self.cancellation = cancellation
        self.host = host
        self.port = port
        self.secret = secret
        self.session_id = session_id
        self.meta = meta
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None
        self._async_state_lock = threading.Lock()
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._async_stopped: asyncio.Event | None = None

    @property
    def uri(self) -> str:
        return f"ws://{self.host}:{self.port}/internal/agent"

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        with self._async_state_lock:
            loop = self._async_loop
            stopped = self._async_stopped
        if loop is not None and stopped is not None and loop.is_running():
            try:
                loop.call_soon_threadsafe(stopped.set)
            except RuntimeError:
                # The loop can close between is_running() and scheduling.
                pass
        if self._thread is not None:
            self._thread.join(timeout=3)

    # -- thread body --------------------------------------------------------

    def _run(self) -> None:
        try:
            asyncio.run(self._loop())
        except Exception:
            # The web link is best-effort; the terminal agent keeps working.
            pass

    async def _loop(self) -> None:
        import websockets

        loop = asyncio.get_running_loop()
        stopped = asyncio.Event()
        if self._stopped.is_set():
            stopped.set()
        with self._async_state_lock:
            self._async_loop = loop
            self._async_stopped = stopped
        try:
            while not self._stopped.is_set():
                try:
                    async with websockets.connect(
                        self.uri, max_size=WS_MAX_SIZE, ping_interval=20
                    ) as ws:
                        await self._session(ws)
                except Exception:
                    pass
                if self._stopped.is_set():
                    break
                try:
                    await asyncio.wait_for(stopped.wait(), RECONNECT_DELAY)
                except asyncio.TimeoutError:
                    pass
        finally:
            with self._async_state_lock:
                if self._async_loop is loop:
                    self._async_loop = None
                    self._async_stopped = None

    async def _session(self, ws) -> None:
        await ws.send(json.dumps({
            "type": "register",
            "secret": self.secret,
            "session_id": self.session_id,
            "meta": self.meta,
        }))
        # Wait for the registration ack before streaming.
        ack = json.loads(await ws.recv())
        if not (isinstance(ack, dict) and ack.get("type") == "registered"):
            return
        assigned_session_id = ack.get("session_id")
        if not isinstance(assigned_session_id, str) or not assigned_session_id:
            return
        self.session_id = assigned_session_id

        loop = asyncio.get_running_loop()

        # Replay retained history so the hub rebuilds the full timeline and
        # derived state, then stream everything after it.
        snapshot = self.events.after(0)
        for event in snapshot:
            await ws.send(json.dumps({"type": "event", "event": event.to_dict()}))
        seq = snapshot[-1].seq if snapshot else 0

        async def forward():
            nonlocal seq
            while not self._stopped.is_set():
                pending = await loop.run_in_executor(
                    None, self.events.wait_after, seq, EVENT_WAIT_TIMEOUT
                )
                for event in pending:
                    await ws.send(json.dumps({"type": "event", "event": event.to_dict()}))
                    seq = event.seq

        async def receive():
            async for raw in ws:
                try:
                    message = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(message, dict) or message.get("type") != "action":
                    continue
                action = message.get("action")
                if isinstance(action, dict):
                    self._apply(action)

        async def watch_stop():
            with self._async_state_lock:
                stopped = self._async_stopped
            if stopped is None:
                return
            await stopped.wait()

        tasks = [
            asyncio.create_task(forward()),
            asyncio.create_task(receive()),
            asyncio.create_task(watch_stop()),
        ]
        try:
            done, _ = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                task.result()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    # -- action injection ---------------------------------------------------

    def _apply(self, action: dict) -> None:
        """Route a browser action into the local brokers (thread-safe).

        The hub acks delivery optimistically; the real accept/reject happens
        here, in the agent process. On failure we publish an ``error`` event
        into the agent's stream so every browser viewing this session gets
        truthful feedback instead of a silently dropped action.
        """
        kind = action.get("type")
        if kind == "submit":
            if self.cancellation.operation_id is not None:
                self.events.publish("error", text="Message not sent: agent is working.")
                return
            try:
                self.inputs.submit(str(action.get("text", "")), "web")
            except ValueError as exc:
                self.events.publish("error", text=f"Message not sent: {exc}")
        elif kind == "prompt_response":
            ok = self.prompts.resolve(
                str(action.get("id", "")), str(action.get("answer", "")), source="web"
            )
            if not ok:
                self.events.publish(
                    "error",
                    text="Prompt response was not accepted "
                    "(it may have already been answered, or the choice was invalid).",
                )
        elif kind == "cancel":
            self.cancellation.cancel(action.get("operation_id"))

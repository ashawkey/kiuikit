"""Shared web hub for kia terminal agents.

A single ``kia --hub`` process owns the public port (the one exposed via a
Cloudflare tunnel). Every auto-linked terminal agent stays terminal-first and,
in addition, connects to this hub over a loopback WebSocket to register a
*session*. The hub multiplexes all registered sessions into one browser UI,
each shown as a separate tab.

Two client surfaces:

* **Browsers** authenticate with the shared token (``/api/login``) and open
  ``/api/ws`` — either as a control channel (session list) or, with a
  ``session=<id>`` query param, as a per-session event stream. This half of
  the server generalizes a single authenticated event stream to a registry of
  per-session streams.
* **Agents** connect to ``/internal/agent`` (loopback only, shared-secret
  authed), register a session, stream their :class:`~kiui.agent.io.EventHub`
  events, and receive browser actions to inject into their local brokers.

Note: no ``from __future__ import annotations`` here — FastAPI must evaluate
endpoint annotations eagerly, and ``fastapi`` types are only imported inside
``_create_app``.
"""

import asyncio
import hmac
import json
import os
import secrets
import socket
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from starlette.websockets import WebSocketDisconnect

from kiui.agent.io import EventHub


COOKIE_NAME = "kia_web_session"
SESSION_TTL = 12 * 60 * 60
MAX_SESSIONS = 32               # browser login sessions (not agent sessions)
# Concurrent browser websockets. Each open browser page holds one control
# socket plus one socket per agent session (all kept open so tab switching is
# instant), so this must comfortably exceed (agents + 1) * pages.
MAX_CLIENTS = 128
LOGIN_RATE_WINDOW = 60          # seconds
LOGIN_RATE_LIMIT = 8            # attempts per window per IP
EVENT_WAIT_TIMEOUT = 1.0        # fallback re-check interval for browser readers
MAX_BODY_BYTES = 4096
MAX_ANSWER_BYTES = 128 * 1024
LOOPBACK_HOST = "127.0.0.1"
DEFAULT_HUB_PORT = 8765
DISCOVERY_INFO_RETRIES = 10
DISCOVERY_INFO_RETRY_DELAY = 0.05

# Discovery file: written by the hub, read by terminal agents so they can find
# the hub's port and the shared internal secret without config edits.
HUB_INFO_PATH = Path.home() / ".kia" / "hub.json"


def read_hub_info() -> dict | None:
    """Return the hub info file's contents, or ``None`` if absent/corrupt.

    This is a plain file read with no liveness check — the file can be stale if
    a previous hub crashed without cleaning up. Agents should use
    :func:`discover_hub`; the hub itself uses this to reclaim its own file.
    """
    try:
        info = json.loads(HUB_INFO_PATH.read_text(encoding="utf-8"))
        return info if isinstance(info, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _hub_reachable(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except (OSError, ValueError, OverflowError):
        return False


def discover_hub(port: int = DEFAULT_HUB_PORT) -> dict | None:
    """Return a *reachable* hub's connection info, or ``None``.

    Guards against a stale ``hub.json`` left by a crashed hub: the recorded
    endpoint is probed and ignored if nothing is listening. A cross-platform
    TCP probe is used deliberately — ``os.kill(pid, 0)`` is not a safe liveness
    check on Windows, where it terminates the target process.
    """
    info = read_hub_info()
    if not info:
        # Uvicorn becomes reachable just before Hub.start writes hub.json.
        # Only wait when the expected port is already accepting connections,
        # keeping the normal "no hub" path fast.
        if not _hub_reachable(LOOPBACK_HOST, port):
            return None
        for _ in range(DISCOVERY_INFO_RETRIES):
            time.sleep(DISCOVERY_INFO_RETRY_DELAY)
            info = read_hub_info()
            if info:
                break
        if not info:
            return None
    if not _hub_reachable(info.get("host", LOOPBACK_HOST), info.get("port", 0)):
        return None
    return info


class RemoteSession:
    """A single agent registered with the hub.

    Holds a hub-local :class:`EventHub` that browsers subscribe to. Incoming
    agent events are *re-published* here (re-sequenced with hub-local seqs), so
    all of the browser-facing replay/reconnect logic is reused unchanged. The
    agent's own seq/stream_id never reach the browser.
    """

    def __init__(self, session_id: str, meta: dict):
        self.id = session_id
        self.meta = meta                 # {title, cwd, model, pid, host}
        self.events = EventHub()
        # Derived UI state, tracked from the event stream (mirrors what the
        # single-agent server pulls from its brokers).
        self.pending = 0
        self.prompt: dict | None = None
        self.operation_id: str | None = None
        self.agent_ws = None             # starlette WebSocket to the agent
        self.agent_send_lock = asyncio.Lock()
        # Async wakeup for browser readers. Events are published on the event
        # loop (via ingest), so browsers wait on this instead of parking a
        # thread-pool worker per open session.
        self._notify = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    def reset(self) -> None:
        """Start a fresh event stream (new stream_id) on agent (re)connect."""
        self.events = EventHub()
        self.pending = 0
        self.prompt = None
        self.operation_id = None

    def touch(self) -> None:
        """Wake browser readers (safe to call from any thread)."""
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._notify.set)

    async def wait_events(self, after_seq: int, timeout: float) -> list:
        """Return events with ``seq > after_seq``, waiting up to *timeout*.

        Uses a clear-then-recheck sequence so a publish that races the wait is
        never lost; the *timeout* is only a fallback re-check, not the norm.
        """
        events = self.events.after(after_seq)
        if events:
            return events
        self._notify.clear()
        events = self.events.after(after_seq)
        if events:
            return events
        try:
            await asyncio.wait_for(self._notify.wait(), timeout)
        except asyncio.TimeoutError:
            pass
        return self.events.after(after_seq)

    def ingest(self, event: dict) -> None:
        """Consume one agent event: update derived state and re-publish it."""
        etype = event.get("type", "")
        data = event.get("data", {}) or {}
        if etype == "queue_changed":
            self.pending = int(data.get("pending", 0) or 0)
        elif etype == "prompt_open":
            self.prompt = {
                "id": data.get("id", ""),
                "kind": data.get("kind", "text"),
                "message": data.get("message", ""),
                "choices": list(data.get("choices", []) or []),
                "default": data.get("default", ""),
            }
        elif etype == "prompt_resolved":
            self.prompt = None
        elif etype == "operation_start":
            self.operation_id = data.get("id")
        elif etype == "operation_end":
            if self.operation_id == data.get("id"):
                self.operation_id = None
        self.events.publish(etype, **data)
        self.touch()

    def summary(self) -> dict:
        return {
            "id": self.id,
            "title": self.meta.get("title", self.id),
            "cwd": self.meta.get("cwd", ""),
            "model": self.meta.get("model", ""),
            "host": self.meta.get("host", ""),
        }


class Hub:
    def __init__(
        self,
        *,
        port: int = 8765,
        token: str | None = None,
        secret: str | None = None,
        console=None,
    ):
        self.host = LOOPBACK_HOST
        self.port = port
        self.token = token or secrets.token_urlsafe(32)
        self.secret = secret or secrets.token_urlsafe(32)
        self.console = console

        # Agent session registry.
        self._sessions: dict[str, RemoteSession] = {}
        self._registry_lock = threading.Lock()
        # Published on every registry change so control-channel browsers refresh.
        self._control = EventHub()
        self._control_notify = asyncio.Event()
        # The uvicorn event loop, captured at startup; used to wake async
        # browser readers from any thread.
        self._loop: asyncio.AbstractEventLoop | None = None

        # Browser auth state.
        self._logins: dict[str, tuple[float, str]] = {}
        self._login_attempts: dict[str, deque] = defaultdict(deque)
        self._login_lock = threading.Lock()
        self._clients = 0
        self._client_lock = threading.Lock()

        self._thread: threading.Thread | None = None
        self._server = None
        self._lifecycle_lock = threading.Lock()
        self.app = self._create_app()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    # -- logging ------------------------------------------------------------

    def _log(self, msg: str) -> None:
        # Logging must never break a connection (e.g. a console encoding error).
        if self.console is not None:
            try:
                self.console.system(msg)
            except Exception:
                pass

    def _log_dim(self, msg: str) -> None:
        if self.console is not None:
            try:
                self.console.debug(msg)
            except Exception:
                pass

    # -- registry -----------------------------------------------------------

    def _session_list(self) -> list[dict]:
        with self._registry_lock:
            return [s.summary() for s in self._sessions.values()]

    def register(self, session_id: str, meta: dict) -> RemoteSession:
        with self._registry_lock:
            session = self._sessions.get(session_id)
            new = session is None
            if new:
                session = RemoteSession(session_id, meta)
                self._sessions[session_id] = session
            else:
                # Re-registration (agent reconnect): fresh stream, updated meta.
                session.meta = meta
                session.reset()
            session._loop = self._loop
            count = len(self._sessions)
        self._notify_control()
        session.touch()  # in case a browser is already streaming this session
        verb = "connected" if new else "reconnected"
        title = meta.get("title", session_id)
        cwd = meta.get("cwd", "")
        detail = f" [{cwd}]" if cwd else ""
        self._log(f"agent {verb}: {title}{detail} (session {session_id[:8]}, {count} total)")
        return session

    def drop(self, session_id: str) -> None:
        with self._registry_lock:
            session = self._sessions.pop(session_id, None)
            count = len(self._sessions)
        if session is not None:
            session.touch()  # unblock its browser readers so they re-check
            self._log(
                f"agent disconnected: {session.meta.get('title', session_id)} "
                f"(session {session_id[:8]}, {count} total)"
            )
        self._notify_control()

    def _notify_control(self) -> None:
        self._control.publish("sessions_changed")
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._control_notify.set)

    def get_session(self, session_id: str) -> RemoteSession | None:
        with self._registry_lock:
            return self._sessions.get(session_id)

    # -- app ----------------------------------------------------------------

    def _create_app(self):
        try:
            from fastapi import FastAPI, HTTPException, Request, WebSocket
            from fastapi.responses import FileResponse, JSONResponse
            from fastapi.staticfiles import StaticFiles
        except ImportError as exc:
            raise RuntimeError(
                "Web UI dependencies are missing. Install with: pip install 'kiui[kia]'"
            ) from exc

        @asynccontextmanager
        async def lifespan(_app):
            # Cross-thread wakeups (from the agent event stream or the registry)
            # are scheduled onto this loop via call_soon_threadsafe.
            self._loop = asyncio.get_running_loop()
            with self._registry_lock:
                for session in self._sessions.values():
                    session._loop = self._loop
            yield

        app = FastAPI(
            docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan
        )
        assets = Path(__file__).with_name("frontend") / "dist"

        @app.middleware("http")
        async def security_headers(request: Request, call_next):
            response = await call_next(request)
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Referrer-Policy"] = "no-referrer"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; connect-src 'self'; "
                "img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; "
                "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
            )
            return response

        def client_ip(request: Request) -> str:
            return request.client.host if request.client else "unknown"

        def check_rate_limit(ip: str) -> None:
            now = time.time()
            stale = [
                key for key, value in self._login_attempts.items()
                if not value or value[-1] < now - LOGIN_RATE_WINDOW
            ]
            for key in stale:
                self._login_attempts.pop(key, None)
            attempts = self._login_attempts[ip]
            while attempts and attempts[0] < now - LOGIN_RATE_WINDOW:
                attempts.popleft()
            if len(attempts) >= LOGIN_RATE_LIMIT:
                raise HTTPException(status_code=429, detail="Too many login attempts.")
            attempts.append(now)

        def new_login() -> tuple[str, str]:
            login_id = secrets.token_urlsafe(32)
            csrf = secrets.token_urlsafe(24)
            with self._login_lock:
                now = time.time()
                self._logins = {
                    key: value for key, value in self._logins.items()
                    if value[0] >= now
                }
                while len(self._logins) >= MAX_SESSIONS:
                    self._logins.pop(next(iter(self._logins)))
                self._logins[login_id] = (time.time() + SESSION_TTL, csrf)
            return login_id, csrf

        def authenticate_cookie(
            login_id: str | None, *, refresh: bool = True
        ) -> str | None:
            if not login_id:
                return None
            with self._login_lock:
                record = self._logins.get(login_id)
                if record is None:
                    return None
                expires, csrf = record
                if expires < time.time():
                    self._logins.pop(login_id, None)
                    return None
                if refresh:
                    self._logins.pop(login_id)
                    self._logins[login_id] = (time.time() + SESSION_TTL, csrf)
                return csrf

        def valid_origin(origin: str | None, host: str | None) -> bool:
            if not origin or not host:
                return False
            parsed = urlsplit(origin)
            return parsed.scheme in {"http", "https"} and parsed.netloc == host

        @app.get("/")
        async def index():
            return FileResponse(assets / "index.html")

        @app.get("/api/health")
        async def health():
            return {"ok": True}

        @app.post("/api/login")
        async def login(request: Request):
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > MAX_BODY_BYTES:
                        raise HTTPException(status_code=413, detail="Request too large.")
                except ValueError as exc:
                    raise HTTPException(
                        status_code=400, detail="Invalid Content-Length."
                    ) from exc
            ip = client_ip(request)
            check_rate_limit(ip)
            raw_body = await request.body()
            if len(raw_body) > MAX_BODY_BYTES:
                raise HTTPException(status_code=413, detail="Request too large.")
            try:
                body = json.loads(raw_body)
                if not isinstance(body, dict):
                    raise ValueError
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail="Invalid JSON.") from exc
            supplied = str(body.get("token", ""))
            if not hmac.compare_digest(
                supplied.encode("utf-8"), self.token.encode("utf-8")
            ):
                raise HTTPException(status_code=401, detail="Invalid token.")
            self._login_attempts.pop(ip, None)
            login_id, csrf = new_login()
            self._log_dim(f"web login from {ip}")
            response = JSONResponse({"ok": True, "csrf": csrf})
            response.set_cookie(
                COOKIE_NAME,
                login_id,
                httponly=True,
                secure=request.url.scheme == "https",
                samesite="strict",
                max_age=SESSION_TTL,
                path="/",
            )
            return response

        @app.post("/api/logout")
        async def logout(request: Request):
            login_id = request.cookies.get(COOKIE_NAME)
            csrf = authenticate_cookie(login_id, refresh=False)
            if csrf is None or not hmac.compare_digest(
                request.headers.get("x-csrf-token", ""), csrf
            ):
                raise HTTPException(status_code=403, detail="Forbidden.")
            with self._login_lock:
                self._logins.pop(login_id, None)
            response = JSONResponse({"ok": True})
            response.delete_cookie(COOKIE_NAME, path="/")
            return response

        @app.get("/api/sessions")
        async def sessions(request: Request):
            if authenticate_cookie(request.cookies.get(COOKIE_NAME)) is None:
                raise HTTPException(status_code=403, detail="Forbidden.")
            return {"sessions": self._session_list()}

        # -- agent-facing internal endpoint (loopback + secret) -------------

        @app.websocket("/internal/agent")
        async def agent_endpoint(websocket: WebSocket):
            # Loopback-only. Uvicorn's forwarded_allow_ips keeps client.host as
            # the real peer for non-loopback hops, so this rejects anything the
            # tunnel might forward to the internal path.
            peer = websocket.client.host if websocket.client else ""
            if peer not in ("127.0.0.1", "::1"):
                await websocket.close(code=4403)
                return
            await websocket.accept()
            try:
                raw = await websocket.receive_json()
            except Exception:
                await websocket.close(code=4400)
                return
            if not isinstance(raw, dict) or raw.get("type") != "register":
                await websocket.close(code=4400)
                return
            if not hmac.compare_digest(
                str(raw.get("secret", "")).encode("utf-8"),
                self.secret.encode("utf-8"),
            ):
                await websocket.close(code=4403)
                return
            session_id = str(raw.get("session_id", "")) or secrets.token_urlsafe(12)
            meta = raw.get("meta", {})
            if not isinstance(meta, dict):
                meta = {}
            session = self.register(session_id, meta)
            session.agent_ws = websocket
            await websocket.send_json({"type": "registered", "session_id": session_id})
            try:
                await self._serve_agent(websocket, session)
            except Exception:
                pass
            finally:
                # Only drop if this websocket is still the active one (a newer
                # reconnect may have replaced it).
                if session.agent_ws is websocket:
                    self.drop(session_id)

        # -- browser websocket: control channel or per-session stream -------

        @app.websocket("/api/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            csrf = authenticate_cookie(websocket.cookies.get(COOKIE_NAME))
            if csrf is None or not valid_origin(
                websocket.headers.get("origin"), websocket.headers.get("host")
            ):
                await websocket.close(code=4403)
                return
            with self._client_lock:
                if self._clients >= MAX_CLIENTS:
                    await websocket.close(code=4429)
                    return
                self._clients += 1
                count = self._clients
            session_id = websocket.query_params.get("session", "")
            channel = f"session {session_id[:8]}" if session_id else "control"
            self._log_dim(f"web client connected ({channel}, {count} sockets)")
            try:
                if session_id:
                    await self._serve_browser_session(websocket, csrf, session_id)
                else:
                    await self._serve_browser_control(websocket, csrf)
            except Exception:
                pass
            finally:
                with self._client_lock:
                    self._clients -= 1
                    count = self._clients
                self._log_dim(f"web client disconnected ({channel}, {count} sockets)")

        app.mount("/assets", StaticFiles(directory=assets / "assets"), name="assets")
        return app

    # -- agent stream -------------------------------------------------------

    async def _serve_agent(self, websocket, session: RemoteSession) -> None:
        """Pump agent events into the session and forward nothing back here.

        Browser actions are pushed to ``session.agent_ws`` from the browser
        handlers directly (same event loop), so this coroutine only needs to
        drain inbound events until the agent disconnects.
        """
        while True:
            message = await websocket.receive_json()
            if not isinstance(message, dict):
                continue
            if message.get("type") == "event":
                event = message.get("event")
                if isinstance(event, dict):
                    session.ingest(event)

    # -- browser: control channel (session list) ---------------------------

    async def _serve_browser_control(self, websocket, csrf: str) -> None:
        await websocket.send_json({
            "type": "sessions",
            "csrf": csrf,
            "sessions": self._session_list(),
        })
        seq = self._control.latest_seq

        async def push_updates():
            nonlocal seq
            while True:
                if self._control.latest_seq != seq:
                    seq = self._control.latest_seq
                    await websocket.send_json({
                        "type": "sessions",
                        "sessions": self._session_list(),
                    })
                    continue
                self._control_notify.clear()
                if self._control.latest_seq != seq:
                    continue
                try:
                    await asyncio.wait_for(
                        self._control_notify.wait(), EVENT_WAIT_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    pass

        async def drain():
            # Control channel takes no client actions; just detect disconnect.
            while True:
                await websocket.receive_text()

        await self._race(push_updates(), drain())

    # -- browser: per-session event stream ---------------------------------

    async def _serve_browser_session(
        self, websocket, csrf: str, session_id: str
    ) -> None:
        session = self.get_session(session_id)
        if session is None:
            await websocket.close(code=4404)
            return

        try:
            raw_seq = websocket.query_params.get("after", "0")
            seq = max(0, int(raw_seq))
        except ValueError:
            seq = 0
        # Clamp an out-of-range cursor (e.g. a reconnect against a fresh stream)
        # so replay always starts from a valid point.
        if seq > session.events.latest_seq:
            seq = 0

        def state_frame(s: RemoteSession, after_seq: int) -> dict:
            return {
                "type": "state",
                "csrf": csrf,
                "session": session_id,
                "stream_id": s.events.stream_id,
                "latest_seq": s.events.latest_seq,
                "pending": s.pending,
                "operation_id": s.operation_id,
                "prompt": s.prompt,
                "oldest_seq": s.events.oldest_seq,
                "replay_truncated": s.events.has_replay_gap(after_seq),
            }

        stream = session.events.stream_id
        await websocket.send_json(state_frame(session, seq))

        async def send_events():
            nonlocal seq, stream
            while True:
                current = self.get_session(session_id)
                if current is None:
                    await websocket.close(code=4404)
                    return
                # The event stream is replaced on agent reconnect (new
                # stream_id, seq reset to 0). Re-issue state and restart from
                # the head so the client rebuilds this session's timeline.
                if current.events.stream_id != stream:
                    stream = current.events.stream_id
                    seq = 0
                    await websocket.send_json(state_frame(current, seq))
                pending = await current.wait_events(seq, EVENT_WAIT_TIMEOUT)
                for event in pending:
                    await websocket.send_json(event.to_dict())
                    seq = event.seq

        async def receive_actions():
            while True:
                try:
                    payload = await websocket.receive_json()
                except WebSocketDisconnect:
                    return
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "rejected", "error": "Invalid JSON message."
                    })
                    continue
                if not isinstance(payload, dict):
                    continue
                action = payload.get("type")
                if action not in {"submit", "prompt_response", "cancel"}:
                    await websocket.send_json({
                        "type": "rejected", "error": "Unknown action type."
                    })
                    continue
                if action == "prompt_response":
                    answer = str(payload.get("answer", ""))
                    if len(answer.encode("utf-8")) > MAX_ANSWER_BYTES:
                        await websocket.send_json({"type": "prompt_ack", "ok": False})
                        continue
                ok = await self._forward_to_agent(session_id, payload)
                if action == "submit":
                    if ok:
                        await websocket.send_json({"type": "accepted"})
                    else:
                        await websocket.send_json({
                            "type": "rejected", "error": "Agent is offline."
                        })
                elif action == "prompt_response":
                    await websocket.send_json({"type": "prompt_ack", "ok": ok})
                elif action == "cancel":
                    await websocket.send_json({"type": "cancel_ack", "ok": ok})

        await self._race(send_events(), receive_actions())

    async def _forward_to_agent(self, session_id: str, payload: dict) -> bool:
        session = self.get_session(session_id)
        if session is None or session.agent_ws is None:
            return False
        try:
            async with session.agent_send_lock:
                await session.agent_ws.send_json({"type": "action", "action": payload})
            return True
        except Exception:
            return False

    @staticmethod
    async def _race(*coros) -> None:
        tasks = [asyncio.create_task(coro) for coro in coros]
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

    # -- lifecycle ----------------------------------------------------------

    def _write_info(self) -> None:
        HUB_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
        HUB_INFO_PATH.write_text(
            json.dumps({
                "host": self.host,
                "port": self.port,
                "secret": self.secret,
                "pid": os.getpid(),
                "started": time.time(),
            }),
            encoding="utf-8",
        )
        try:
            os.chmod(HUB_INFO_PATH, 0o600)
        except OSError:
            pass

    def _remove_info(self) -> None:
        try:
            if HUB_INFO_PATH.exists():
                info = read_hub_info()
                if info and info.get("pid") == os.getpid():
                    HUB_INFO_PATH.unlink()
        except OSError:
            pass

    def start(self) -> None:
        import uvicorn

        with self._lifecycle_lock:
            if self._thread is not None:
                raise RuntimeError("The hub is already started.")
            config = uvicorn.Config(
                self.app,
                host=self.host,
                port=self.port,
                log_level="warning",
                proxy_headers=True,
                forwarded_allow_ips="127.0.0.1,::1",
                ws_max_size=256 * 1024,
            )
            self._server = uvicorn.Server(config)
            self._thread = threading.Thread(target=self._server.run, daemon=True)
            self._thread.start()
            deadline = time.time() + 10
            while (
                not self._server.started
                and self._thread.is_alive()
                and time.time() < deadline
            ):
                time.sleep(0.05)
            if not self._server.started:
                self._server.should_exit = True
                self._thread.join(timeout=5)
                self._server = None
                self._thread = None
                raise RuntimeError("The hub failed to start.")
            try:
                self._write_info()
            except BaseException:
                self._server.should_exit = True
                self._thread.join(timeout=5)
                self._server = None
                self._thread = None
                raise

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._remove_info()
            if self._server is not None:
                self._server.should_exit = True
            if self._thread is not None:
                self._thread.join(timeout=5)
            self._server = None
            self._thread = None

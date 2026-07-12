"""Optional authenticated FastAPI companion for the kia terminal UI.

Note: no ``from __future__ import annotations`` here — FastAPI must be able
to evaluate endpoint annotations eagerly, since ``fastapi`` types are only
imported inside ``_create_app``.
"""

import asyncio
import hmac
import json
import secrets
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from urllib.parse import urlsplit

from kiui.agent.io import CancellationToken, EventHub, InputBroker, PromptBroker


COOKIE_NAME = "kia_web_session"
SESSION_TTL = 12 * 60 * 60
MAX_SESSIONS = 32
MAX_CLIENTS = 8
LOGIN_RATE_WINDOW = 60          # seconds
LOGIN_RATE_LIMIT = 8            # attempts per window per IP
EVENT_WAIT_TIMEOUT = 1.0        # max block per EventHub.wait_after call
MAX_BODY_BYTES = 4096
MAX_ANSWER_BYTES = 128 * 1024
LOOPBACK_HOST = "127.0.0.1"


class WebServer:
    def __init__(
        self,
        events: EventHub,
        inputs: InputBroker,
        prompts: PromptBroker,
        cancellation: CancellationToken,
        *,
        port: int = 8765,
        token: str | None = None,
    ):
        self.events = events
        self.inputs = inputs
        self.prompts = prompts
        self.cancellation = cancellation
        self.host = LOOPBACK_HOST
        self.port = port
        self.token = token or secrets.token_urlsafe(32)
        self._sessions: dict[str, tuple[float, str]] = {}
        self._login_attempts: dict[str, deque[float]] = defaultdict(deque)
        self._session_lock = threading.Lock()
        self._clients = 0
        self._client_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._server = None
        self.app = self._create_app()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _create_app(self):
        try:
            from fastapi import FastAPI, HTTPException, Request, WebSocket
            from fastapi.responses import FileResponse, JSONResponse
            from fastapi.staticfiles import StaticFiles
        except ImportError as exc:
            raise RuntimeError(
                "Web UI dependencies are missing. Install with: pip install 'kiui[kia]'"
            ) from exc

        app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
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
                # ansi-to-react uses numeric inline styles for exact terminal
                # colors. Scripts remain restricted to packaged assets.
                "img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; "
                "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
            )
            return response

        def client_ip(request: Request) -> str:
            return request.client.host if request.client else "unknown"

        def check_rate_limit(ip: str) -> None:
            now = time.time()
            # Drop IPs whose window has fully expired so the map stays bounded.
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

        def new_session() -> tuple[str, str]:
            session_id = secrets.token_urlsafe(32)
            csrf = secrets.token_urlsafe(24)
            with self._session_lock:
                now = time.time()
                self._sessions = {
                    key: value for key, value in self._sessions.items()
                    if value[0] >= now
                }
                while len(self._sessions) >= MAX_SESSIONS:
                    self._sessions.pop(next(iter(self._sessions)))
                self._sessions[session_id] = (time.time() + SESSION_TTL, csrf)
            return session_id, csrf

        def authenticate_cookie(session_id: str | None) -> str | None:
            if not session_id:
                return None
            with self._session_lock:
                record = self._sessions.get(session_id)
                if record is None:
                    return None
                expires, csrf = record
                if expires < time.time():
                    self._sessions.pop(session_id, None)
                    return None
                # Re-insert so dict order tracks last use and eviction is LRU.
                self._sessions.pop(session_id)
                self._sessions[session_id] = (time.time() + SESSION_TTL, csrf)
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
            # Compare as bytes: compare_digest raises TypeError on non-ASCII str.
            if not hmac.compare_digest(
                supplied.encode("utf-8"), self.token.encode("utf-8")
            ):
                raise HTTPException(status_code=401, detail="Invalid token.")
            self._login_attempts.pop(ip, None)
            session_id, csrf = new_session()
            response = JSONResponse({"ok": True, "csrf": csrf})
            response.set_cookie(
                COOKIE_NAME,
                session_id,
                httponly=True,
                # Uvicorn trusts proxy headers only from loopback (configured
                # in start()), so tunnel-provided HTTPS is safe to honor here.
                secure=request.url.scheme == "https",
                samesite="strict",
                max_age=SESSION_TTL,
                path="/",
            )
            return response

        @app.post("/api/logout")
        async def logout(request: Request):
            session_id = request.cookies.get(COOKIE_NAME)
            csrf = authenticate_cookie(session_id)
            if csrf is None or not hmac.compare_digest(
                request.headers.get("x-csrf-token", ""), csrf
            ):
                raise HTTPException(status_code=403, detail="Forbidden.")
            with self._session_lock:
                self._sessions.pop(session_id, None)
            response = JSONResponse({"ok": True})
            response.delete_cookie(COOKIE_NAME, path="/")
            return response

        @app.websocket("/api/ws")
        async def websocket_endpoint(websocket: WebSocket):
            # Accept first so rejections deliver a real application close code.
            # Closing before accept rejects the handshake at the HTTP layer,
            # which browsers surface only as an opaque 1006, hiding 4403/4429.
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
            # Everything after the slot claim runs under try/finally so an
            # early disconnect (e.g. during accept) can never leak the slot.
            try:
                await self._serve_websocket(websocket, csrf)
            except Exception:
                # Disconnects surface as different exception types depending
                # on the ws backend; the session is over either way.
                pass
            finally:
                with self._client_lock:
                    self._clients -= 1

        app.mount("/assets", StaticFiles(directory=assets / "assets"), name="assets")
        return app

    async def _serve_websocket(self, websocket, csrf: str) -> None:
        # The endpoint has already accepted the handshake.
        try:
            raw_seq = websocket.query_params.get("after", "0")
            seq = max(0, int(raw_seq))
        except ValueError:
            seq = 0
        client_stream = websocket.query_params.get("stream", "")
        if client_stream and client_stream != self.events.stream_id:
            seq = 0
        elif seq > self.events.latest_seq:
            seq = 0
        active_prompt = self.prompts.active
        await websocket.send_json({
            "type": "state",
            # Re-issue the CSRF token: a new tab shares the session cookie but
            # not the sessionStorage copy, and logout needs it.
            "csrf": csrf,
            "stream_id": self.events.stream_id,
            "latest_seq": self.events.latest_seq,
            "pending": self.inputs.pending,
            "operation_id": self.cancellation.operation_id,
            "prompt": (
                {
                    "id": active_prompt.id,
                    "kind": active_prompt.kind,
                    "message": active_prompt.message,
                    "choices": active_prompt.choices,
                    "default": active_prompt.default,
                }
                if active_prompt is not None else None
            ),
        })

        loop = asyncio.get_running_loop()

        async def send_events():
            nonlocal seq
            while True:
                pending = await loop.run_in_executor(
                    None, self.events.wait_after, seq, EVENT_WAIT_TIMEOUT
                )
                for event in pending:
                    await websocket.send_json(event.to_dict())
                    seq = event.seq

        async def receive_actions():
            while True:
                try:
                    payload = await websocket.receive_json()
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "rejected", "error": "Invalid JSON message."
                    })
                    continue
                if not isinstance(payload, dict):
                    continue
                action = payload.get("type")
                if action == "submit":
                    try:
                        item = self.inputs.submit(
                            str(payload.get("text", "")), "web"
                        )
                    except ValueError as exc:
                        await websocket.send_json({
                            "type": "rejected", "error": str(exc)
                        })
                    else:
                        await websocket.send_json({"type": "accepted", "id": item.id})
                elif action == "prompt_response":
                    answer = str(payload.get("answer", ""))
                    if len(answer.encode("utf-8")) > MAX_ANSWER_BYTES:
                        await websocket.send_json({"type": "prompt_ack", "ok": False})
                        continue
                    ok = self.prompts.resolve(
                        str(payload.get("id", "")),
                        answer,
                        source="web",
                    )
                    await websocket.send_json({"type": "prompt_ack", "ok": ok})
                elif action == "cancel":
                    ok = self.cancellation.cancel(payload.get("operation_id"))
                    await websocket.send_json({"type": "cancel_ack", "ok": ok})

        sender = asyncio.create_task(send_events())
        receiver = asyncio.create_task(receive_actions())
        try:
            done, _ = await asyncio.wait(
                {sender, receiver}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                task.result()  # surface disconnects/errors to the caller
        finally:
            sender.cancel()
            receiver.cancel()
            await asyncio.gather(sender, receiver, return_exceptions=True)

    def start(self) -> None:
        import uvicorn

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
            raise RuntimeError("The web server failed to start.")

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)

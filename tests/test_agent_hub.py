import inspect
import json
import re
import socket
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from kiui.agent.hub import Hub, RemoteSession
from kiui.agent.io import CancellationToken, EventHub, InputBroker, PromptBroker


def make_hub():
    return Hub(token="correct-token", secret="internal-secret")


def receive_type(sock, event_type, limit=10):
    for _ in range(limit):
        message = sock.receive_json()
        if message.get("type") == event_type:
            return message
    raise AssertionError(f"Did not receive {event_type!r}")


def add_session(hub, session_id="s1", **meta):
    meta.setdefault("title", "proj · model")
    meta.setdefault("cwd", "/proj")
    meta.setdefault("model", "model")
    meta.setdefault("host", "box")
    return hub.register(session_id, meta)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def ws_header_kwargs(websockets, headers):
    params = inspect.signature(websockets.connect).parameters
    if "additional_headers" in params:
        return {"additional_headers": headers}
    return {"extra_headers": headers}


# -- static / auth (browser surface) ---------------------------------------

def test_hub_is_loopback_only():
    hub = make_hub()
    assert hub.host == "127.0.0.1"


def test_built_frontend_and_security_policy_are_served():
    hub = make_hub()
    with TestClient(hub.app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert '<div id="root"></div>' in response.text
        asset = re.search(r'(?:src|href)="(/assets/[^"]+)"', response.text)
        assert asset is not None
        assert client.get(asset.group(1)).status_code == 200
        policy = response.headers["content-security-policy"]
        assert "script-src 'self'" in policy
        assert "style-src 'self' 'unsafe-inline'" in policy


def test_login_rejects_bad_tokens():
    hub = make_hub()
    with TestClient(hub.app) as client:
        assert client.post("/api/login", json={"token": "wrong"}).status_code == 401
        # Non-ASCII tokens must be a clean 401, not a compare_digest TypeError.
        assert client.post("/api/login", json={"token": "秘密"}).status_code == 401
        response = client.post("/api/login", json={"token": "correct-token"})
        assert response.status_code == 200
        assert response.cookies.get("kia_web_session")
        assert "Secure" not in response.headers["set-cookie"]


def test_https_proxy_marks_session_cookie_secure():
    hub = make_hub()
    with TestClient(hub.app, base_url="https://kia.example") as client:
        response = client.post("/api/login", json={"token": "correct-token"})
        assert response.status_code == 200
        assert "Secure" in response.headers["set-cookie"]


def test_sessions_endpoint_requires_auth_and_lists():
    hub = make_hub()
    add_session(hub, "s1")
    with TestClient(hub.app) as client:
        assert client.get("/api/sessions").status_code == 403
        client.post("/api/login", json={"token": "correct-token"})
        listed = client.get("/api/sessions").json()["sessions"]
        assert [s["id"] for s in listed] == ["s1"]


def test_logout_removes_login_without_refreshing_it():
    hub = make_hub()
    with TestClient(hub.app) as client:
        response = client.post("/api/login", json={"token": "correct-token"})
        login_id = response.cookies["kia_web_session"]
        csrf = response.json()["csrf"]

        class TrackingDict(dict):
            writes = 0

            def __setitem__(self, key, value):
                self.writes += 1
                super().__setitem__(key, value)

        hub._logins = TrackingDict(hub._logins)

        response = client.post("/api/logout", headers={"x-csrf-token": csrf})
        assert response.status_code == 200
        assert login_id not in hub._logins
        assert hub._logins.writes == 0


# -- browser websockets ----------------------------------------------------

def test_control_channel_lists_sessions():
    hub = make_hub()
    add_session(hub, "s1")
    with TestClient(hub.app) as client:
        client.post("/api/login", json={"token": "correct-token"})
        with client.websocket_connect(
            "/api/ws", headers={"origin": "http://testserver"}
        ) as sock:
            message = receive_type(sock, "sessions")
            assert [s["id"] for s in message["sessions"]] == ["s1"]


def test_per_session_state_and_event_replay():
    hub = make_hub()
    session = add_session(hub, "s1")
    session.events.publish("system", text="ready")
    with TestClient(hub.app) as client:
        response = client.post("/api/login", json={"token": "correct-token"})
        csrf = response.json()["csrf"]
        with client.websocket_connect(
            "/api/ws?session=s1&after=0", headers={"origin": "http://testserver"}
        ) as sock:
            state = receive_type(sock, "state")
            assert state["csrf"] == csrf
            assert state["session"] == "s1"
            assert receive_type(sock, "system")["data"]["text"] == "ready"


def test_per_session_reports_truncated_replay():
    hub = make_hub()
    session = add_session(hub, "s1")
    session.events = EventHub(max_events=2)
    session.events.publish("system", text="evicted")
    session.events.publish("system", text="kept-1")
    session.events.publish("system", text="kept-2")
    with TestClient(hub.app) as client:
        client.post("/api/login", json={"token": "correct-token"})
        with client.websocket_connect(
            "/api/ws?session=s1&after=0", headers={"origin": "http://testserver"}
        ) as sock:
            state = receive_type(sock, "state")
            assert state["replay_truncated"] is True
            assert state["oldest_seq"] == 2
            assert receive_type(sock, "system")["data"]["text"] == "kept-1"


def test_per_session_state_reflects_derived_prompt():
    hub = make_hub()
    session = add_session(hub, "s1")
    session.ingest({
        "type": "prompt_open",
        "data": {"id": "p1", "kind": "select", "message": "ok?",
                 "choices": ["Yes", "No"], "default": "Yes"},
    })
    with TestClient(hub.app) as client:
        client.post("/api/login", json={"token": "correct-token"})
        with client.websocket_connect(
            "/api/ws?session=s1&after=0", headers={"origin": "http://testserver"}
        ) as sock:
            state = receive_type(sock, "state")
            assert state["prompt"]["id"] == "p1"


def test_per_session_restarts_stream_on_agent_reconnect():
    hub = make_hub()
    session = add_session(hub, "s1")
    session.events.publish("system", text="first")
    with TestClient(hub.app) as client:
        client.post("/api/login", json={"token": "correct-token"})
        with client.websocket_connect(
            "/api/ws?session=s1&after=0", headers={"origin": "http://testserver"}
        ) as sock:
            first_stream = receive_type(sock, "state")["stream_id"]
            assert receive_type(sock, "system")["data"]["text"] == "first"

            # Agent reconnects: register swaps in a fresh event stream.
            hub.register("s1", session.meta)
            reconnected = hub.get_session("s1")
            reconnected.events.publish("system", text="second")

            # The browser must be re-issued state on the new stream, then the
            # new event — not silently stall on the stale sequence cursor.
            new_state = receive_type(sock, "state")
            assert new_state["stream_id"] != first_stream
            assert receive_type(sock, "system")["data"]["text"] == "second"


def test_unknown_session_is_rejected():
    hub = make_hub()
    with TestClient(hub.app) as client:
        client.post("/api/login", json={"token": "correct-token"})
        with client.websocket_connect(
            "/api/ws?session=missing", headers={"origin": "http://testserver"}
        ) as sock:
            with pytest.raises(WebSocketDisconnect) as exc:
                sock.receive_json()
        assert exc.value.code == 4404


def test_submit_without_agent_is_rejected():
    hub = make_hub()
    add_session(hub, "s1")  # registered but no live agent websocket
    with TestClient(hub.app) as client:
        client.post("/api/login", json={"token": "correct-token"})
        with client.websocket_connect(
            "/api/ws?session=s1", headers={"origin": "http://testserver"}
        ) as sock:
            receive_type(sock, "state")
            sock.send_json({"type": "submit", "text": "hi"})
            assert receive_type(sock, "rejected")["error"] == "Agent is offline."


def test_unknown_browser_action_is_rejected_without_forwarding():
    hub = make_hub()
    session = add_session(hub, "s1")

    class FakeAgentSocket:
        def __init__(self):
            self.sent = []

        async def send_json(self, payload):
            self.sent.append(payload)

    agent = FakeAgentSocket()
    session.agent_ws = agent
    with TestClient(hub.app) as client:
        client.post("/api/login", json={"token": "correct-token"})
        with client.websocket_connect(
            "/api/ws?session=s1", headers={"origin": "http://testserver"}
        ) as sock:
            receive_type(sock, "state")
            sock.send_json({"type": "delete_all_files"})
            assert receive_type(sock, "rejected")["error"] == "Unknown action type."
    assert agent.sent == []


def test_websocket_requires_same_origin():
    hub = make_hub()
    add_session(hub, "s1")
    with TestClient(hub.app) as client:
        client.post("/api/login", json={"token": "correct-token"})
        with client.websocket_connect(
            "/api/ws?session=s1", headers={"origin": "https://attacker.example"}
        ) as sock:
            with pytest.raises(WebSocketDisconnect) as exc:
                sock.receive_json()
        assert exc.value.code == 4403


def test_malformed_json_rejected_without_disconnect():
    hub = make_hub()
    add_session(hub, "s1")
    with TestClient(hub.app) as client:
        client.post("/api/login", json={"token": "correct-token"})
        with client.websocket_connect(
            "/api/ws?session=s1", headers={"origin": "http://testserver"}
        ) as sock:
            receive_type(sock, "state")
            sock.send_text("not-json")
            assert receive_type(sock, "rejected")["error"] == "Invalid JSON message."


def test_client_slot_released_after_disconnect():
    hub = make_hub()
    add_session(hub, "s1")
    with TestClient(hub.app) as client:
        client.post("/api/login", json={"token": "correct-token"})
        for _ in range(3):
            with client.websocket_connect(
                "/api/ws?session=s1", headers={"origin": "http://testserver"}
            ) as sock:
                receive_type(sock, "state")
                assert hub._clients == 1
    deadline = time.time() + 2
    while hub._clients != 0 and time.time() < deadline:
        time.sleep(0.01)
    assert hub._clients == 0


# -- RemoteSession derived-state unit --------------------------------------

def test_remote_session_ingest_tracks_derived_state():
    session = RemoteSession("s", {})
    session.ingest({"type": "operation_start", "data": {"id": "op1"}})
    assert session.operation_id == "op1"
    session.ingest({"type": "operation_end", "data": {"id": "op1"}})
    assert session.operation_id is None
    session.ingest({"type": "prompt_open", "data": {
        "id": "p", "kind": "text", "message": "m", "choices": [], "default": ""}})
    assert session.prompt["id"] == "p"
    session.ingest({"type": "prompt_resolved", "data": {"id": "p"}})
    assert session.prompt is None
    # Every ingested event is re-published for browser replay.
    assert session.events.latest_seq == 4


# -- action feedback / discovery -------------------------------------------

def test_apply_reports_broker_failures():
    from kiui.agent.hubclient import HubClient

    events = EventHub()
    inputs = InputBroker(events)
    prompts = PromptBroker(events)
    cancellation = CancellationToken(events)
    client = HubClient(
        events, inputs, prompts, cancellation,
        host="127.0.0.1", port=1, secret="", session_id="x", meta={},
    )

    inputs.submit("first", "web")  # occupy the single input slot
    before = events.latest_seq
    client._apply({"type": "submit", "text": "overflow"})  # busy -> rejected
    errors = [e for e in events.after(before) if e.type == "error"]
    assert errors and "not sent" in errors[0].data["text"].lower()

    inputs.get_nowait()
    operation_id = cancellation.begin("working")
    before = events.latest_seq
    client._apply({"type": "submit", "text": "while busy"})
    errors = [e for e in events.after(before) if e.type == "error"]
    assert errors and "working" in errors[0].data["text"].lower()
    cancellation.finish(operation_id)

    before = events.latest_seq
    client._apply({"type": "prompt_response", "id": "missing", "answer": "x"})
    errors = [e for e in events.after(before) if e.type == "error"]
    assert errors and "not accepted" in errors[0].data["text"].lower()


def test_discover_hub_ignores_stale_file(tmp_path, monkeypatch):
    import kiui.agent.hub as hubmod

    info_path = tmp_path / "hub.json"
    monkeypatch.setattr(hubmod, "HUB_INFO_PATH", info_path)

    assert hubmod.discover_hub() is None  # no file

    dead = free_port()  # bound then released -> nothing listening
    info_path.write_text(json.dumps({"host": "127.0.0.1", "port": dead, "secret": "s"}))
    assert hubmod.discover_hub() is None  # stale: unreachable

    port = free_port()
    hub = Hub(port=port, token="t", secret="s")
    hub.start()  # writes the (monkeypatched) info file
    try:
        got = hubmod.discover_hub()
        assert got is not None and got["port"] == port
    finally:
        hub.stop()


def test_discover_hub_retries_info_after_port_becomes_reachable(monkeypatch):
    import kiui.agent.hub as hubmod

    info = {"host": "127.0.0.1", "port": 8765, "secret": "s"}
    reads = iter([None, info])
    monkeypatch.setattr(hubmod, "read_hub_info", lambda: next(reads))
    monkeypatch.setattr(hubmod, "_hub_reachable", lambda *args, **kwargs: True)
    monkeypatch.setattr(hubmod, "DISCOVERY_INFO_RETRY_DELAY", 0)

    assert hubmod.discover_hub(8765) == info


def test_hub_rejects_double_start(tmp_path, monkeypatch):
    import kiui.agent.hub as hubmod

    monkeypatch.setattr(hubmod, "HUB_INFO_PATH", tmp_path / "hub.json")
    hub = Hub(port=free_port(), token="t", secret="s")
    hub.start()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            hub.start()
    finally:
        hub.stop()


def test_hubclient_uses_server_assigned_session_id():
    import asyncio
    from kiui.agent.hubclient import HubClient

    events = EventHub()
    client = HubClient(
        events,
        InputBroker(events),
        PromptBroker(events),
        CancellationToken(events),
        host="127.0.0.1",
        port=1,
        secret="s",
        session_id="",
        meta={},
    )

    class FakeWebSocket:
        async def send(self, _payload):
            pass

        async def recv(self):
            return json.dumps({"type": "registered", "session_id": "assigned"})

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    async def run():
        client._stopped.set()
        client._async_stopped = asyncio.Event()
        client._async_stopped.set()
        await client._session(FakeWebSocket())

    asyncio.run(run())
    assert client.session_id == "assigned"


# -- live agent link (real loopback; internal endpoint is 127.0.0.1 only) ---

def test_agent_link_end_to_end():
    websockets = pytest.importorskip("websockets")
    import asyncio

    port = free_port()
    hub = Hub(port=port, token="correct-token", secret="internal-secret")
    hub.start()

    events = EventHub()
    inputs = InputBroker(events)
    prompts = PromptBroker(events)
    cancellation = CancellationToken(events)
    events.publish("system", text="agent online")

    from kiui.agent.hubclient import HubClient

    client = HubClient(
        events, inputs, prompts, cancellation,
        host="127.0.0.1", port=port, secret="internal-secret",
        session_id="live", meta={"title": "t", "cwd": "/c", "model": "m", "host": "h"},
    )
    try:
        client.start()
        deadline = time.time() + 5
        while time.time() < deadline and not hub.get_session("live"):
            time.sleep(0.05)
        session = hub.get_session("live")
        assert session is not None

        # Replayed history reaches the hub.
        deadline = time.time() + 5
        while time.time() < deadline and session.events.latest_seq < 1:
            time.sleep(0.05)
        assert session.events.latest_seq >= 1

        # Browser action forwarded to the agent's InputBroker.
        async def forward():
            return await hub._forward_to_agent("live", {"type": "submit", "text": "yo"})

        assert asyncio.new_event_loop().run_until_complete(forward()) is True
        deadline = time.time() + 5
        got = None
        while time.time() < deadline:
            if inputs.pending:
                got = inputs.get_nowait()
                break
            time.sleep(0.05)
        assert got is not None and got.text == "yo" and got.source == "web"
    finally:
        client.stop()
        hub.stop()


def test_two_sessions_stream_concurrently():
    websockets = pytest.importorskip("websockets")
    import asyncio

    import httpx

    from kiui.agent.hubclient import HubClient

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    hub = Hub(port=port, token="correct-token", secret="internal-secret")
    hub.start()

    def make_agent(sid):
        events = EventHub()
        inputs = InputBroker(events)
        prompts = PromptBroker(events)
        cancellation = CancellationToken(events)
        client = HubClient(
            events, inputs, prompts, cancellation,
            host="127.0.0.1", port=port, secret="internal-secret",
            session_id=sid, meta={"title": sid, "cwd": "/", "model": "m", "host": "h"},
        )
        client.start()
        return client, events

    client_a, events_a = make_agent("a")
    client_b, events_b = make_agent("b")

    async def run():
        deadline = time.time() + 5
        while time.time() < deadline and not (
            hub.get_session("a") and hub.get_session("b")
        ):
            await asyncio.sleep(0.05)
        assert hub.get_session("a") and hub.get_session("b")

        with httpx.Client(base_url=base) as c:
            r = c.post("/api/login", json={"token": "correct-token"})
            cookie = r.cookies.get("kia_web_session")
        headers = {"Origin": base, "Cookie": f"kia_web_session={cookie}"}

        # Two browser sockets open at once, one per session.
        async with websockets.connect(
            f"ws://127.0.0.1:{port}/api/ws?session=a&after=0",
            **ws_header_kwargs(websockets, headers),
        ) as ws_a, websockets.connect(
            f"ws://127.0.0.1:{port}/api/ws?session=b&after=0",
            **ws_header_kwargs(websockets, headers),
        ) as ws_b:
            for ws in (ws_a, ws_b):
                state = json.loads(await asyncio.wait_for(ws.recv(), 5))
                assert state["type"] == "state"

            # Each agent publishes; each browser must receive only its own event.
            events_a.publish("assistant_message", text="from-a")
            events_b.publish("assistant_message", text="from-b")

            async def next_message(ws):
                for _ in range(10):
                    frame = json.loads(await asyncio.wait_for(ws.recv(), 5))
                    if frame.get("type") == "assistant_message":
                        return frame["data"]["text"]
                raise AssertionError("no assistant_message")

            got_a = await asyncio.wait_for(next_message(ws_a), 5)
            got_b = await asyncio.wait_for(next_message(ws_b), 5)
            assert got_a == "from-a", got_a
            assert got_b == "from-b", got_b

    try:
        asyncio.new_event_loop().run_until_complete(run())
    finally:
        client_a.stop()
        client_b.stop()
        hub.stop()


def test_agent_link_rejects_wrong_secret():
    websockets = pytest.importorskip("websockets")
    import asyncio

    port = free_port()
    hub = Hub(port=port, token="correct-token", secret="internal-secret")
    hub.start()

    async def attempt():
        uri = f"ws://127.0.0.1:{port}/internal/agent"
        async with websockets.connect(uri) as ws:
            await ws.send(json.dumps({
                "type": "register", "secret": "WRONG",
                "session_id": "x", "meta": {},
            }))
            await ws.recv()  # should close before a 'registered' ack

    try:
        with pytest.raises(websockets.exceptions.ConnectionClosed) as exc:
            asyncio.new_event_loop().run_until_complete(attempt())
        assert exc.value.code == 4403
        assert hub.get_session("x") is None
    finally:
        hub.stop()

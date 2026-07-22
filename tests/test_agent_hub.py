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
from kiui.agent.utils.io import CancellationToken, EventHub, InputBroker, PromptBroker


def make_hub():
    return Hub(token="correct-token")


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


def test_sessions_endpoint_requires_auth_and_lists():
    hub = make_hub()
    add_session(hub, "s1")
    with TestClient(hub.app) as client:
        assert client.get("/api/sessions").status_code == 403
        client.post("/api/login", json={"token": "correct-token"})
        listed = client.get("/api/sessions").json()["sessions"]
        assert [s["id"] for s in listed] == ["s1"]


# -- browser websockets ----------------------------------------------------

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


# -- RemoteSession derived-state unit --------------------------------------

# -- action feedback / discovery -------------------------------------------

# -- live agent link (real loopback; internal endpoint is 127.0.0.1 only) ---

def test_agent_link_end_to_end():
    websockets = pytest.importorskip("websockets")
    import asyncio

    port = free_port()
    hub = Hub(port=port, token="correct-token")
    hub.start()

    events = EventHub()
    inputs = InputBroker(events)
    prompts = PromptBroker(events)
    cancellation = CancellationToken(events)
    events.publish("system", text="agent online")

    from kiui.agent.hubclient import HubClient

    client = HubClient(
        events, inputs, prompts, cancellation,
        host="127.0.0.1", port=port, token="correct-token",
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
    hub = Hub(port=port, token="correct-token")
    hub.start()

    def make_agent(sid):
        events = EventHub()
        inputs = InputBroker(events)
        prompts = PromptBroker(events)
        cancellation = CancellationToken(events)
        client = HubClient(
            events, inputs, prompts, cancellation,
            host="127.0.0.1", port=port, token="correct-token",
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


def test_agent_link_rejects_wrong_token():
    websockets = pytest.importorskip("websockets")
    import asyncio

    port = free_port()
    hub = Hub(port=port, token="correct-token")
    hub.start()

    async def attempt():
        uri = f"ws://127.0.0.1:{port}/internal/agent"
        async with websockets.connect(uri) as ws:
            await ws.send(json.dumps({
                "type": "register", "token": "WRONG",
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

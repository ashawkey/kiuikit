import threading
import time
import re

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from kiui.agent.io import CancellationToken, EventHub, InputBroker, PromptBroker
from kiui.agent.web import WebServer


def make_server():
    events = EventHub()
    inputs = InputBroker(events)
    prompts = PromptBroker(events)
    cancellation = CancellationToken(events)
    server = WebServer(
        events, inputs, prompts, cancellation, token="correct-token"
    )
    return server, events, inputs, prompts, cancellation


def receive_type(socket, event_type, limit=10):
    for _ in range(limit):
        message = socket.receive_json()
        if message.get("type") == event_type:
            return message
    raise AssertionError(f"Did not receive {event_type!r}")


def test_server_is_loopback_only():
    server, *_ = make_server()
    assert server.host == "127.0.0.1"
    assert server.url == "http://127.0.0.1:8765"


def test_built_frontend_and_security_policy_are_served():
    server, *_ = make_server()
    with TestClient(server.app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert '<div id="root"></div>' in response.text
        asset = re.search(r'(?:src|href)="(/assets/[^"]+)"', response.text)
        assert asset is not None
        assert client.get(asset.group(1)).status_code == 200
        policy = response.headers["content-security-policy"]
        assert "script-src 'self'" in policy
        assert "style-src 'self' 'unsafe-inline'" in policy


def test_login_and_authenticated_event_replay():
    server, events, *_ = make_server()
    with TestClient(server.app) as client:
        assert client.post("/api/login", json={"token": "wrong"}).status_code == 401
        # Non-ASCII tokens must be a clean 401, not a compare_digest TypeError.
        assert client.post("/api/login", json={"token": "秘密"}).status_code == 401
        response = client.post("/api/login", json={"token": "correct-token"})
        assert response.status_code == 200
        assert response.cookies.get("kia_web_session")
        assert "Secure" not in response.headers["set-cookie"]
        csrf = response.json()["csrf"]

        events.publish("system", text="ready")
        with client.websocket_connect(
            "/api/ws?after=0", headers={"origin": "http://testserver"}
        ) as socket:
            # state re-issues the CSRF token so fresh tabs can log out.
            assert receive_type(socket, "state")["csrf"] == csrf
            event = receive_type(socket, "system")
            assert event["data"]["text"] == "ready"


def test_https_proxy_marks_session_cookie_secure():
    server, *_ = make_server()
    with TestClient(server.app, base_url="https://kia.example") as client:
        response = client.post("/api/login", json={"token": "correct-token"})
        assert response.status_code == 200
        assert "Secure" in response.headers["set-cookie"]


def test_websocket_submits_messages_and_cancels():
    server, _, inputs, _, cancellation = make_server()
    with TestClient(server.app) as client:
        client.post("/api/login", json={"token": "correct-token"})
        operation_id = cancellation.begin("model")
        with client.websocket_connect(
            "/api/ws", headers={"origin": "http://testserver"}
        ) as socket:
            socket.send_json({"type": "submit", "text": "hello"})
            accepted = receive_type(socket, "accepted")
            assert accepted["id"]
            assert inputs.get_nowait().text == "hello"

            socket.send_json({"type": "cancel", "operation_id": operation_id})
            assert receive_type(socket, "cancel_ack")["ok"] is True
            assert cancellation.cancelled


def test_websocket_requires_same_origin():
    server, *_ = make_server()
    with TestClient(server.app) as client:
        client.post("/api/login", json={"token": "correct-token"})
        # The handshake is accepted, then closed with an application code so
        # the browser can distinguish an auth failure from a dropped socket.
        with client.websocket_connect(
            "/api/ws", headers={"origin": "https://attacker.example"}
        ) as socket:
            with pytest.raises(WebSocketDisconnect) as exc:
                socket.receive_json()
        assert exc.value.code == 4403


def test_web_client_can_resolve_mobile_prompt():
    server, _, _, prompts, _ = make_server()
    result = []
    thread = threading.Thread(
        target=lambda: result.append(
            prompts.ask("select", "Allow?", choices=["Yes", "No"])
        )
    )
    thread.start()
    deadline = time.time() + 2
    while prompts.active is None and time.time() < deadline:
        time.sleep(0.01)

    with TestClient(server.app) as client:
        client.post("/api/login", json={"token": "correct-token"})
        with client.websocket_connect(
            "/api/ws", headers={"origin": "http://testserver"}
        ) as socket:
            state = receive_type(socket, "state")
            prompt = state["prompt"]
            socket.send_json({
                "type": "prompt_response", "id": prompt["id"], "answer": "Yes"
            })
            assert receive_type(socket, "prompt_ack")["ok"] is True
    thread.join(timeout=2)
    assert result == ["Yes"]


def test_websocket_releases_client_slot_after_disconnect():
    server, *_ = make_server()
    with TestClient(server.app) as client:
        client.post("/api/login", json={"token": "correct-token"})
        for _ in range(3):  # reconnect repeatedly; slots must not leak
            with client.websocket_connect(
                "/api/ws", headers={"origin": "http://testserver"}
            ) as socket:
                receive_type(socket, "state")
                assert server._clients == 1
    deadline = time.time() + 2
    while server._clients != 0 and time.time() < deadline:
        time.sleep(0.01)
    assert server._clients == 0


def test_websocket_rejects_malformed_json_without_disconnect():
    server, *_ = make_server()
    with TestClient(server.app) as client:
        client.post("/api/login", json={"token": "correct-token"})
        with client.websocket_connect(
            "/api/ws", headers={"origin": "http://testserver"}
        ) as socket:
            receive_type(socket, "state")
            socket.send_text("not-json")
            rejected = receive_type(socket, "rejected")
            assert rejected["error"] == "Invalid JSON message."

"""Tests for OpenAI Codex OAuth storage and Responses transport."""

from __future__ import annotations

import base64
import json
import os
import time
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from kiui.agent.providers import (
    AuthInteraction,
    CompletionRequest,
    ProviderError,
    ProviderSettings,
)
from kiui.agent.providers.auth import CredentialStore, OAuthCredential
from kiui.agent.providers.openai_codex import OpenAICodexProvider, _build_body
from kiui.agent.providers.openai_codex_oauth import (
    JWT_CLAIM_PATH,
    _exchange_code,
    _authorization_url,
    _create_pkce,
    _decode_account_id,
    _device_login,
    _parse_authorization_input,
)


def _jwt(account_id: str = "account-1") -> str:
    def encode(value):
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")

    return f"{encode({'alg': 'none'})}.{encode({JWT_CLAIM_PATH: {'chatgpt_account_id': account_id}})}.sig"


def _credential(*, expires: float | None = None, account_id: str = "account-1") -> OAuthCredential:
    return OAuthCredential(
        access=_jwt(account_id),
        refresh="refresh-token",
        expires=expires if expires is not None else time.time() + 3600,
        metadata={"account_id": account_id},
    )


def test_oauth_pkce_url_and_manual_callback_parsing():
    verifier, challenge = _create_pkce()
    assert len(verifier) >= 43
    assert "=" not in verifier
    assert "=" not in challenge

    url = _authorization_url(challenge, "state-1")
    query = parse_qs(urlparse(url).query)
    assert query["client_id"]
    assert query["code_challenge"] == [challenge]
    assert query["state"] == ["state-1"]
    assert query["originator"] == ["kia"]

    code, state = _parse_authorization_input(
        "http://localhost:1455/auth/callback?code=abc&state=state-1"
    )
    assert (code, state) == ("abc", "state-1")
    assert _parse_authorization_input("abc#state-1") == ("abc", "state-1")
    assert _decode_account_id(_jwt()) == "account-1"


def test_oauth_code_exchange_builds_stored_credential(monkeypatch):
    seen = {}

    def post(url, **kwargs):
        seen["url"] = url
        seen["data"] = kwargs["data"]
        return httpx.Response(200, json={
            "access_token": _jwt("account-2"),
            "refresh_token": "rotating-refresh",
            "expires_in": 3600,
        })

    monkeypatch.setattr("kiui.agent.providers.openai_codex_oauth.httpx.post", post)
    credential = _exchange_code("authorization-code", "verifier", "http://localhost/callback")

    assert seen["data"]["grant_type"] == "authorization_code"
    assert seen["data"]["code_verifier"] == "verifier"
    assert credential.refresh == "rotating-refresh"
    assert credential.metadata["account_id"] == "account-2"


def test_device_login_wait_can_be_cancelled(monkeypatch):
    cancelled = False
    calls = 0

    def post(url, **kwargs):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={
            "device_auth_id": "device-1",
            "user_code": "ABCD-EFGH",
            "interval": 10,
        })

    def notify(message):
        nonlocal cancelled
        cancelled = True

    monkeypatch.setattr("kiui.agent.providers.openai_codex_oauth.httpx.post", post)
    monkeypatch.setattr(
        "kiui.agent.providers.openai_codex_oauth.webbrowser.open", lambda url: True
    )
    interaction = AuthInteraction(
        select=lambda message, choices: choices[0],
        prompt=lambda message: None,
        notify=notify,
        cancelled=lambda: cancelled,
    )

    with pytest.raises(ProviderError) as exc_info:
        _device_login(interaction)

    assert exc_info.value.code == "cancelled"
    assert calls == 1


def test_login_removes_credentials_when_cancelled_during_write(monkeypatch):
    cancelled = False
    deleted = []

    class Store:
        def write_oauth(self, provider_id, credential):
            nonlocal cancelled
            cancelled = True

        def delete(self, provider_id):
            deleted.append(provider_id)

    monkeypatch.setattr(
        "kiui.agent.providers.openai_codex.login_openai_codex",
        lambda interaction: _credential(),
    )
    provider = object.__new__(OpenAICodexProvider)
    provider._store = Store()
    interaction = AuthInteraction(
        select=lambda message, choices: choices[0],
        prompt=lambda message: None,
        notify=lambda message: None,
        cancelled=lambda: cancelled,
    )

    with pytest.raises(ProviderError) as exc_info:
        provider.login(interaction)

    assert exc_info.value.code == "cancelled"
    assert deleted == ["openai-codex"]


def test_credential_store_round_trip_and_delete(tmp_path):
    path = tmp_path / "auth.json"
    store = CredentialStore(path)
    credential = _credential()

    assert store.read_oauth("openai-codex") is None
    store.write_oauth("openai-codex", credential)
    assert store.read_oauth("openai-codex") == credential
    assert "refresh-token" in path.read_text()
    if os.name == "posix":
        assert path.stat().st_mode & 0o077 == 0
    assert store.delete("openai-codex") is True
    assert store.read_oauth("openai-codex") is None


def test_codex_request_converts_chat_messages_and_tools():
    request = CompletionRequest(
        model="gpt-5.6-sol",
        messages=[
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "inspect"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"file":"a.py"}'},
                }],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "contents"},
        ],
        tools=[{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"file": {"type": "string"}}},
            },
        }],
        reasoning_effort="high",
        session_id="session/one",
    )

    body = _build_body(request)

    assert body["instructions"] == "system prompt"
    assert body["store"] is False
    assert body["include"] == ["reasoning.encrypted_content"]
    assert body["reasoning"] == {"effort": "high", "summary": "auto"}
    assert body["prompt_cache_key"] == "session_one"
    assert body["input"][1]["type"] == "function_call"
    assert body["input"][2] == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": "contents",
    }
    assert body["tools"][0]["name"] == "read_file"


def test_codex_stream_round_trip_and_provider_state(monkeypatch, tmp_path):
    terminal_output = [
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "considering"}],
            "encrypted_content": "encrypted",
        },
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "done", "annotations": []}],
        },
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call-1",
            "name": "read_file",
            "arguments": '{"file":"a.py"}',
        },
    ]
    events = [
        {"type": "response.output_text.delta", "output_index": 1, "delta": "done"},
        {"type": "response.reasoning_summary_text.delta", "output_index": 0, "delta": "considering"},
        {
            "type": "response.done",
            "response": {
                "id": "resp_1",
                "status": "completed",
                "output": terminal_output,
                "usage": {
                    "input_tokens": 20,
                    "output_tokens": 8,
                    "total_tokens": 28,
                    "input_tokens_details": {"cached_tokens": 5},
                    "output_tokens_details": {"reasoning_tokens": 3},
                },
            },
        },
    ]
    payload = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
    seen_request = {}

    def handler(request: httpx.Request):
        seen_request["headers"] = request.headers
        seen_request["body"] = json.loads(request.content)
        return httpx.Response(200, content=payload.encode(), headers={"content-type": "text/event-stream"})

    store = CredentialStore(tmp_path / "auth.json")
    store.write_oauth("openai-codex", _credential())
    provider = OpenAICodexProvider(ProviderSettings(), store=store)
    monkeypatch.setattr(
        provider,
        "_new_client",
        lambda timeout: httpx.Client(transport=httpx.MockTransport(handler), timeout=timeout),
    )
    text = []
    thinking = []
    stream = provider.open_stream(CompletionRequest(
        model="gpt-5.6-sol",
        messages=[{"role": "system", "content": "system"}, {"role": "user", "content": "go"}],
        stream=True,
        reasoning_effort="high",
        session_id="session-one",
    ))
    try:
        result = stream.consume(on_content=text.append, on_thinking=thinking.append)
    finally:
        stream.close()

    assert seen_request["headers"]["authorization"].startswith("Bearer ")
    assert seen_request["headers"]["chatgpt-account-id"] == "account-1"
    assert seen_request["headers"]["originator"] == "kia"
    assert seen_request["headers"]["session-id"] == "session-one"
    assert seen_request["body"]["model"] == "gpt-5.6-sol"
    assert text == ["done"]
    assert thinking == ["considering"]
    assert result.message["content"] == "done"
    assert result.message["tool_calls"][0]["id"] == "call-1"
    assert result.message["provider_state"]["openai-codex"]["output"] == terminal_output
    assert result.usage.cached_prompt_tokens == 5
    assert result.usage.reasoning_tokens == 3
    assert result.finish_reason == "tool_calls"

    replay = _build_body(CompletionRequest(
        model="gpt-5.6-sol",
        messages=[result.message, {"role": "tool", "tool_call_id": "call-1", "content": "ok"}],
    ))
    assert replay["input"][:3] == terminal_output
    assert replay["input"][3]["type"] == "function_call_output"


def test_expired_oauth_token_is_refreshed_once(monkeypatch, tmp_path):
    store = CredentialStore(tmp_path / "auth.json")
    store.write_oauth("openai-codex", _credential(expires=time.time() - 1))
    provider = OpenAICodexProvider(ProviderSettings(), store=store)
    refreshed = _credential(expires=time.time() + 7200, account_id="account-2")
    calls = []

    def refresh(current):
        calls.append(current)
        return refreshed

    monkeypatch.setattr("kiui.agent.providers.openai_codex.refresh_openai_codex", refresh)

    assert provider._resolve_credential() == refreshed
    assert provider._resolve_credential() == refreshed
    assert len(calls) == 1
    assert store.read_oauth("openai-codex") == refreshed


def test_codex_subscription_limit_is_not_retried(tmp_path):
    provider = OpenAICodexProvider(
        ProviderSettings(), store=CredentialStore(tmp_path / "auth.json")
    )
    response = httpx.Response(
        429,
        json={"error": {"code": "usage_limit_reached", "resets_at": time.time() + 600}},
    )

    error = provider._http_error(response)

    assert error.retryable is False
    assert "usage limit" in str(error).lower()


def test_codex_rejects_api_key_and_custom_endpoint():
    with pytest.raises(ValueError, match="api_key"):
        OpenAICodexProvider(ProviderSettings(api_key="secret"))
    with pytest.raises(ValueError, match="base_url"):
        OpenAICodexProvider(ProviderSettings(base_url="https://example.com"))

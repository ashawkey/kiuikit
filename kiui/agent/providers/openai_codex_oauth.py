"""OpenAI Codex OAuth flows for ChatGPT Plus/Pro subscriptions."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from .auth import OAuthCredential
from .types import AuthInteraction, ProviderError

# Public client used by OpenAI's Codex OAuth flow. PKCE public clients do not
# carry a client secret.
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTH_BASE_URL = "https://auth.openai.com"
AUTHORIZE_URL = f"{AUTH_BASE_URL}/oauth/authorize"
TOKEN_URL = f"{AUTH_BASE_URL}/oauth/token"
REDIRECT_URI = "http://localhost:1455/auth/callback"
DEVICE_USER_CODE_URL = f"{AUTH_BASE_URL}/api/accounts/deviceauth/usercode"
DEVICE_TOKEN_URL = f"{AUTH_BASE_URL}/api/accounts/deviceauth/token"
DEVICE_VERIFICATION_URI = f"{AUTH_BASE_URL}/codex/device"
DEVICE_REDIRECT_URI = f"{AUTH_BASE_URL}/deviceauth/callback"
SCOPE = "openid profile email offline_access"
JWT_CLAIM_PATH = "https://api.openai.com/auth"
DEVICE_TIMEOUT_SECONDS = 15 * 60
CALLBACK_TIMEOUT_SECONDS = 5 * 60

_BROWSER = "Browser login (local callback)"
_MANUAL = "Browser login (paste redirect URL)"
_DEVICE = "Device code login (headless/SSH)"


def _check_cancelled(interaction: AuthInteraction) -> None:
    if interaction.cancelled():
        raise ProviderError(
            "OpenAI Codex login cancelled", code="cancelled", retryable=False
        )


def _base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _create_pkce() -> tuple[str, str]:
    verifier = _base64url(secrets.token_bytes(64))
    challenge = _base64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def _authorization_url(verifier_challenge: str, state: str) -> str:
    return AUTHORIZE_URL + "?" + urlencode({
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "code_challenge": verifier_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": "kia",
    })


def _parse_authorization_input(value: str) -> tuple[str | None, str | None]:
    value = value.strip()
    if not value:
        return None, None
    try:
        parsed = urlparse(value)
        if parsed.scheme and parsed.netloc:
            query = parse_qs(parsed.query)
            return query.get("code", [None])[0], query.get("state", [None])[0]
    except ValueError:
        pass
    if "#" in value:
        code, state = value.split("#", 1)
        return code or None, state or None
    if "code=" in value:
        query = parse_qs(value)
        return query.get("code", [None])[0], query.get("state", [None])[0]
    return value, None


def _decode_account_id(access_token: str) -> str:
    try:
        payload = access_token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        account_id = claims[JWT_CLAIM_PATH]["chatgpt_account_id"]
    except (IndexError, KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
        raise ProviderError("OpenAI Codex token does not contain a ChatGPT account ID", retryable=False) from e
    if not isinstance(account_id, str) or not account_id:
        raise ProviderError("OpenAI Codex token contains an invalid ChatGPT account ID", retryable=False)
    return account_id


def _credential_from_token(data: object) -> OAuthCredential:
    if not isinstance(data, dict):
        raise ProviderError("OpenAI Codex token response is not a JSON object", retryable=False)
    access = data.get("access_token")
    refresh = data.get("refresh_token")
    expires_in = data.get("expires_in")
    if not isinstance(access, str) or not isinstance(refresh, str) or not isinstance(expires_in, (int, float)):
        raise ProviderError("OpenAI Codex token response is missing required fields", retryable=False)
    return OAuthCredential(
        access=access,
        refresh=refresh,
        expires=time.time() + float(expires_in),
        metadata={"account_id": _decode_account_id(access)},
    )


def _token_request(data: dict[str, str], operation: str) -> OAuthCredential:
    try:
        response = httpx.post(
            TOKEN_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
    except httpx.HTTPError as e:
        raise ProviderError(f"OpenAI Codex token {operation} failed: {e}", retryable=True) from e
    if not response.is_success:
        raise ProviderError(
            f"OpenAI Codex token {operation} failed (HTTP {response.status_code})",
            status_code=response.status_code,
            retryable=False,
        )
    try:
        payload = response.json()
    except json.JSONDecodeError as e:
        raise ProviderError("OpenAI Codex token response is not valid JSON", retryable=False) from e
    return _credential_from_token(payload)


def _exchange_code(code: str, verifier: str, redirect_uri: str) -> OAuthCredential:
    return _token_request({
        "grant_type": "authorization_code",
        "client_id": CLIENT_ID,
        "code": code,
        "code_verifier": verifier,
        "redirect_uri": redirect_uri,
    }, "exchange")


def refresh_openai_codex(credential: OAuthCredential) -> OAuthCredential:
    return _token_request({
        "grant_type": "refresh_token",
        "refresh_token": credential.refresh,
        "client_id": CLIENT_ID,
    }, "refresh")


class _CallbackState:
    def __init__(self, expected_state: str):
        self.expected_state = expected_state
        self.event = threading.Event()
        self.code: str | None = None
        self.error: str | None = None


def _callback_handler(state: _CallbackState):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path != "/auth/callback":
                self.send_error(404)
                return
            query = parse_qs(parsed.query)
            returned_state = query.get("state", [None])[0]
            code = query.get("code", [None])[0]
            error = query.get("error_description", query.get("error", [None]))[0]
            if returned_state != state.expected_state:
                state.error = "OAuth state mismatch"
                status = 400
            elif error:
                state.error = f"OpenAI authorization failed: {error}"
                status = 400
            elif not code:
                state.error = "OpenAI authorization callback contained no code"
                status = 400
            else:
                state.code = code
                status = 200
            state.event.set()
            body = (
                "OpenAI authentication completed. You can close this window."
                if status == 200
                else state.error or "Authentication failed."
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    return Handler


def _wait_for_browser_callback(url: str, state: str, interaction: AuthInteraction) -> str:
    callback = _CallbackState(state)
    host = os.environ.get("KIA_OAUTH_CALLBACK_HOST", "127.0.0.1")
    try:
        server = ThreadingHTTPServer((host, 1455), _callback_handler(callback))
    except OSError as e:
        interaction.notify(f"Could not start the local OAuth callback server: {e}")
        return _prompt_for_code(url, state, interaction)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        interaction.notify(f"Open this URL to authenticate with OpenAI:\n{url}")
        webbrowser.open(url)
        deadline = time.monotonic() + CALLBACK_TIMEOUT_SECONDS
        while not callback.event.is_set():
            _check_cancelled(interaction)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                interaction.notify("Timed out waiting for the local OAuth callback.")
                return _prompt_for_code(url, state, interaction)
            callback.event.wait(min(0.1, remaining))
        _check_cancelled(interaction)
        if callback.error:
            raise ProviderError(callback.error, retryable=False)
        if not callback.code:
            raise ProviderError("OpenAI authorization returned no code", retryable=False)
        return callback.code
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _prompt_for_code(url: str, state: str, interaction: AuthInteraction) -> str:
    _check_cancelled(interaction)
    interaction.notify(f"Open this URL to authenticate with OpenAI:\n{url}")
    value = interaction.prompt("Paste the authorization code or full redirect URL")
    if value is None:
        raise ProviderError(
            "OpenAI Codex login cancelled", code="cancelled", retryable=False
        )
    _check_cancelled(interaction)
    code, returned_state = _parse_authorization_input(value)
    if returned_state is not None and returned_state != state:
        raise ProviderError("OAuth state mismatch", retryable=False)
    if not code:
        raise ProviderError("No authorization code was provided", retryable=False)
    return code


def _browser_login(interaction: AuthInteraction, *, manual: bool) -> OAuthCredential:
    _check_cancelled(interaction)
    verifier, challenge = _create_pkce()
    state = secrets.token_hex(16)
    url = _authorization_url(challenge, state)
    if manual:
        webbrowser.open(url)
        code = _prompt_for_code(url, state, interaction)
    else:
        code = _wait_for_browser_callback(url, state, interaction)
    credential = _exchange_code(code, verifier, REDIRECT_URI)
    _check_cancelled(interaction)
    return credential


def _device_login(interaction: AuthInteraction) -> OAuthCredential:
    _check_cancelled(interaction)
    try:
        response = httpx.post(
            DEVICE_USER_CODE_URL,
            json={"client_id": CLIENT_ID},
            timeout=30,
        )
    except httpx.HTTPError as e:
        raise ProviderError(f"OpenAI Codex device login failed: {e}", retryable=True) from e
    _check_cancelled(interaction)
    if not response.is_success:
        raise ProviderError(
            f"OpenAI Codex device login failed (HTTP {response.status_code})",
            status_code=response.status_code,
            retryable=False,
        )
    try:
        data = response.json()
    except json.JSONDecodeError as e:
        raise ProviderError("OpenAI Codex device response is not valid JSON", retryable=False) from e
    try:
        device_auth_id = data["device_auth_id"]
        user_code = data["user_code"]
        interval = float(data["interval"])
    except (KeyError, TypeError, ValueError) as e:
        raise ProviderError("OpenAI Codex device response is invalid", retryable=False) from e
    if not isinstance(device_auth_id, str) or not isinstance(user_code, str) or interval < 0:
        raise ProviderError("OpenAI Codex device response is invalid", retryable=False)

    interaction.notify(
        f"Open {DEVICE_VERIFICATION_URI} and enter this code:\n{user_code}"
    )
    webbrowser.open(DEVICE_VERIFICATION_URI)
    deadline = time.monotonic() + DEVICE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        poll_at = min(time.monotonic() + interval, deadline)
        while True:
            remaining = poll_at - time.monotonic()
            if remaining <= 0:
                break
            _check_cancelled(interaction)
            time.sleep(min(0.1, remaining))
        _check_cancelled(interaction)
        try:
            poll = httpx.post(
                DEVICE_TOKEN_URL,
                json={"device_auth_id": device_auth_id, "user_code": user_code},
                timeout=30,
            )
        except httpx.HTTPError as e:
            raise ProviderError(f"OpenAI Codex device polling failed: {e}", retryable=True) from e
        _check_cancelled(interaction)
        if poll.is_success:
            try:
                result = poll.json()
            except json.JSONDecodeError as e:
                raise ProviderError("OpenAI Codex device token response is not valid JSON", retryable=False) from e
            try:
                code = result["authorization_code"]
                verifier = result["code_verifier"]
            except (KeyError, TypeError) as e:
                raise ProviderError("OpenAI Codex device token response is invalid", retryable=False) from e
            credential = _exchange_code(code, verifier, DEVICE_REDIRECT_URI)
            _check_cancelled(interaction)
            return credential
        if poll.status_code in (403, 404):
            continue
        try:
            error = poll.json().get("error")
            code = error.get("code") if isinstance(error, dict) else error
        except (json.JSONDecodeError, TypeError):
            code = None
        if code == "deviceauth_authorization_pending":
            continue
        if code == "slow_down":
            interval += 5
            continue
        raise ProviderError(
            f"OpenAI Codex device authorization failed (HTTP {poll.status_code})",
            status_code=poll.status_code,
            retryable=False,
        )
    _check_cancelled(interaction)
    raise ProviderError("OpenAI Codex device authorization timed out", retryable=False)


def login_openai_codex(interaction: AuthInteraction) -> OAuthCredential:
    _check_cancelled(interaction)
    method = interaction.select(
        "Select OpenAI Codex login method",
        [_BROWSER, _MANUAL, _DEVICE],
    )
    if method is None:
        raise ProviderError(
            "OpenAI Codex login cancelled", code="cancelled", retryable=False
        )
    _check_cancelled(interaction)
    if method == _BROWSER:
        return _browser_login(interaction, manual=False)
    if method == _MANUAL:
        return _browser_login(interaction, manual=True)
    if method == _DEVICE:
        return _device_login(interaction)
    raise ProviderError(f"Unknown OpenAI Codex login method: {method}", retryable=False)

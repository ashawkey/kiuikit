"""OpenAI Codex Responses provider using ChatGPT subscription OAuth."""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable, Iterator

import httpx

from .auth import CredentialStore, OAuthCredential
from .openai_codex_oauth import (
    _decode_account_id,
    login_openai_codex,
    refresh_openai_codex,
)
from .registry import ProviderSettings
from .types import (
    AuthInteraction,
    CompletionRequest,
    CompletionResult,
    CompletionStream,
    LLMProvider,
    ProviderError,
    ProviderUsage,
)

CODEX_BASE_URL = "https://chatgpt.com/backend-api"
CODEX_RESPONSES_URL = f"{CODEX_BASE_URL}/codex/responses"
_RETRYABLE_STATUS = frozenset({408, 409, 425, 500, 502, 503, 504})
_USAGE_LIMIT_CODES = frozenset({"usage_limit_reached", "usage_not_included", "rate_limit_exceeded"})


def _package_version() -> str:
    try:
        return version("kiui")
    except PackageNotFoundError:
        return "dev"


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        item.get("text", "")
        for item in content
        if isinstance(item, dict) and item.get("type") == "text"
    )


def _user_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    if not isinstance(content, list):
        raise ProviderError("Codex user message has invalid content", retryable=False)
    result: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            raise ProviderError("Codex user content item is invalid", retryable=False)
        if item.get("type") == "text":
            result.append({"type": "input_text", "text": item.get("text", "")})
        elif item.get("type") == "image_url":
            image = item.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            if not isinstance(url, str):
                raise ProviderError("Codex image content has no URL", retryable=False)
            result.append({"type": "input_image", "detail": "auto", "image_url": url})
        else:
            raise ProviderError(f"Unsupported Codex user content type: {item.get('type')!r}", retryable=False)
    return result


def _call_id(value: Any) -> str:
    raw = str(value or "").split("|", 1)[0]
    normalized = re.sub(r"[^A-Za-z0-9_-]", "_", raw)[:64].rstrip("_")
    return normalized or "call_kia"


def _messages_to_input(messages: list[dict[str, Any]], model: str) -> tuple[str, list[dict[str, Any]]]:
    instructions: list[str] = []
    items: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "system":
            text = _message_text(message.get("content"))
            if text:
                instructions.append(text)
        elif role == "user":
            content = _user_content(message.get("content"))
            if content:
                items.append({"role": "user", "content": content})
        elif role == "assistant":
            provider_state = message.get("provider_state")
            codex_state = provider_state.get("openai-codex") if isinstance(provider_state, dict) else None
            if (
                isinstance(codex_state, dict)
                and codex_state.get("model") == model
                and isinstance(codex_state.get("output"), list)
            ):
                items.extend(codex_state["output"])
                continue

            text = _message_text(message.get("content"))
            if text:
                items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                    "status": "completed",
                    "id": f"msg_kia_{index}",
                })
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                items.append({
                    "type": "function_call",
                    "call_id": _call_id(tool_call.get("id")),
                    "name": function.get("name", ""),
                    "arguments": function.get("arguments") or "{}",
                })
        elif role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": _call_id(message.get("tool_call_id")),
                "output": _message_text(message.get("content")) or "(no tool output)",
            })
        else:
            raise ProviderError(f"Unsupported Codex message role: {role!r}", retryable=False)
    return "\n\n".join(instructions) or "You are a helpful assistant.", items


def _responses_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for tool in tools:
        function = tool["function"]
        result.append({
            "type": "function",
            "name": function["name"],
            "description": function.get("description", ""),
            "parameters": function.get("parameters", {"type": "object", "properties": {}}),
            "strict": None,
        })
    return result


def _prompt_cache_key(session_id: str | None) -> str | None:
    if not session_id:
        return None
    value = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id)[:64]
    return value or None


def _build_body(request: CompletionRequest) -> dict[str, Any]:
    instructions, items = _messages_to_input(request.messages, request.model)
    body: dict[str, Any] = {
        "model": request.model,
        "store": False,
        "stream": True,
        "instructions": instructions,
        "input": items,
        "text": {"verbosity": "low"},
        "include": ["reasoning.encrypted_content"],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }
    if request.tools:
        body["tools"] = _responses_tools(request.tools)
    if request.reasoning_effort is not None:
        body["reasoning"] = {
            "effort": request.reasoning_effort,
            "summary": "auto",
        }
    cache_key = _prompt_cache_key(request.session_id)
    if cache_key:
        body["prompt_cache_key"] = cache_key
    return body


def _iter_sse(response: Any) -> Iterator[dict[str, Any]]:
    data_lines: list[str] = []
    for line in response.iter_lines():
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line == "":
            if data_lines:
                payload = "\n".join(data_lines).strip()
                data_lines.clear()
                if payload and payload != "[DONE]":
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError as e:
                        raise ProviderError("Invalid JSON in Codex event stream", retryable=False) from e
                    if not isinstance(event, dict):
                        raise ProviderError("Invalid Codex stream event", retryable=False)
                    yield event
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].strip())
    if data_lines:
        payload = "\n".join(data_lines).strip()
        if payload and payload != "[DONE]":
            try:
                event = json.loads(payload)
            except json.JSONDecodeError as e:
                raise ProviderError("Invalid JSON in Codex event stream", retryable=False) from e
            if not isinstance(event, dict):
                raise ProviderError("Invalid Codex stream event", retryable=False)
            yield event


def _response_usage(response: dict[str, Any]) -> ProviderUsage | None:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt = int(usage.get("input_tokens") or 0)
    completion = int(usage.get("output_tokens") or 0)
    input_details = usage.get("input_tokens_details")
    input_details = input_details if isinstance(input_details, dict) else {}
    output_details = usage.get("output_tokens_details")
    output_details = output_details if isinstance(output_details, dict) else {}
    return ProviderUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=int(usage.get("total_tokens") or prompt + completion),
        cached_prompt_tokens=int(input_details.get("cached_tokens") or 0),
        reasoning_tokens=int(output_details.get("reasoning_tokens") or 0),
    )


def _canonical_message(
    output: list[dict[str, Any]],
    model: str,
    streamed_text: str,
    streamed_reasoning: str,
) -> dict[str, Any]:
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for item in output:
        kind = item.get("type")
        if kind == "message":
            for content in item.get("content") or []:
                if content.get("type") == "output_text":
                    text_parts.append(content.get("text") or "")
                elif content.get("type") == "refusal":
                    text_parts.append(content.get("refusal") or "")
        elif kind == "reasoning":
            blocks = item.get("summary") or item.get("content") or []
            reasoning_parts.extend(block.get("text") or "" for block in blocks)
        elif kind == "function_call":
            tool_calls.append({
                "id": item.get("call_id") or item.get("id") or "",
                "type": "function",
                "function": {
                    "name": item.get("name") or "",
                    "arguments": item.get("arguments") or "{}",
                },
            })
    text = "".join(text_parts) or streamed_text
    reasoning = "\n\n".join(part for part in reasoning_parts if part) or streamed_reasoning
    message: dict[str, Any] = {
        "role": "assistant",
        "content": text or None,
        "provider_state": {
            "openai-codex": {"model": model, "output": output},
        },
    }
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


class _CodexCompletionStream(CompletionStream):
    def __init__(self, response: Any, model: str, release: Callable[[], None]):
        self._response = response
        self._model = model
        self._release = release
        self._closed = False

    def consume(self, *, on_content=None, on_thinking=None, should_stop=None) -> CompletionResult:
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        output_items: dict[int, dict[str, Any]] = {}
        terminal: dict[str, Any] | None = None

        for event in _iter_sse(self._response):
            if should_stop is not None and should_stop():
                from kiui.agent.utils.interrupt import RequestInterrupted

                raise RequestInterrupted()
            kind = event.get("type")
            if kind == "response.output_item.added":
                item = event.get("item")
                if isinstance(item, dict):
                    output_items[int(event.get("output_index", len(output_items)))] = item
            elif kind in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
                delta = event.get("delta") or ""
                reasoning_parts.append(delta)
                if delta and on_thinking is not None:
                    on_thinking(delta)
            elif kind == "response.reasoning_summary_part.done":
                reasoning_parts.append("\n\n")
                if on_thinking is not None:
                    on_thinking("\n\n")
            elif kind in ("response.output_text.delta", "response.refusal.delta"):
                delta = event.get("delta") or ""
                text_parts.append(delta)
                if delta and on_content is not None:
                    on_content(delta)
            elif kind == "response.function_call_arguments.delta":
                index = int(event.get("output_index", 0))
                item = output_items.setdefault(index, {"type": "function_call", "arguments": ""})
                item["arguments"] = (item.get("arguments") or "") + (event.get("delta") or "")
            elif kind == "response.function_call_arguments.done":
                index = int(event.get("output_index", 0))
                item = output_items.setdefault(index, {"type": "function_call"})
                item["arguments"] = event.get("arguments") or item.get("arguments") or "{}"
            elif kind == "response.output_item.done":
                item = event.get("item")
                if isinstance(item, dict):
                    output_items[int(event.get("output_index", len(output_items)))] = item
            elif kind in ("response.completed", "response.done", "response.incomplete"):
                value = event.get("response")
                if isinstance(value, dict):
                    terminal = value
                break
            elif kind == "response.failed":
                response = event.get("response")
                response = response if isinstance(response, dict) else {}
                error = response.get("error")
                error = error if isinstance(error, dict) else {}
                raise ProviderError(
                    error.get("message") or "OpenAI Codex response failed",
                    code=error.get("code"),
                    retryable=False,
                )
            elif kind == "error":
                error = event.get("error") if isinstance(event.get("error"), dict) else event
                raise ProviderError(
                    error.get("message") or "OpenAI Codex stream failed",
                    code=error.get("code"),
                    retryable=False,
                )

        if should_stop is not None and should_stop():
            from kiui.agent.utils.interrupt import RequestInterrupted

            raise RequestInterrupted()
        if terminal is None:
            raise ProviderError("OpenAI Codex stream ended without a terminal response", retryable=True)
        output = terminal.get("output")
        if not isinstance(output, list) or (not output and output_items):
            output = [output_items[index] for index in sorted(output_items)]
        message = _canonical_message(
            output,
            self._model,
            "".join(text_parts),
            "".join(reasoning_parts),
        )
        status = terminal.get("status")
        if message.get("tool_calls"):
            finish_reason = "tool_calls"
        elif status == "incomplete":
            finish_reason = "length"
        else:
            finish_reason = "stop"
        return CompletionResult(message, _response_usage(terminal), finish_reason)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._response.close()
        finally:
            self._release()


class OpenAICodexProvider(LLMProvider):
    """ChatGPT subscription provider backed by OpenAI Codex Responses."""

    id = "openai-codex"

    def __init__(self, settings: ProviderSettings, store: CredentialStore | None = None):
        if settings.base_url:
            raise ValueError("openai-codex uses a fixed OpenAI endpoint; base_url is not allowed")
        if settings.api_key:
            raise ValueError("openai-codex uses OAuth; api_key is not allowed")
        self._store = store or CredentialStore()
        self._active_client: httpx.Client | None = None
        self._client_lock = threading.Lock()

    def _resolve_credential(self) -> OAuthCredential:
        try:
            credential = self._store.read_oauth(self.id)
        except Exception as e:
            raise ProviderError(f"Failed to read OpenAI Codex credentials: {e}", retryable=False) from e
        if credential is None:
            raise ProviderError(
                "OpenAI Codex is not logged in. Run /login openai-codex.",
                status_code=401,
                code="auth_required",
                retryable=False,
            )
        if credential.expires > time.time() + 60:
            return credential

        def refresh(current: OAuthCredential | None) -> OAuthCredential:
            if current is None:
                raise ProviderError("OpenAI Codex was logged out during refresh", retryable=False)
            if current.expires > time.time() + 60:
                return current
            try:
                return refresh_openai_codex(current)
            except ProviderError as e:
                raise ProviderError(
                    "OpenAI Codex OAuth refresh failed; run /login openai-codex again.",
                    status_code=e.status_code,
                    code="oauth_refresh_failed",
                    retryable=False,
                ) from e

        try:
            return self._store.modify_oauth(self.id, refresh)
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"Failed to update OpenAI Codex credentials: {e}", retryable=False) from e

    def _headers(self, credential: OAuthCredential) -> dict[str, str]:
        account_id = credential.metadata.get("account_id") or _decode_account_id(credential.access)
        return {
            "Authorization": f"Bearer {credential.access}",
            "chatgpt-account-id": account_id,
            "originator": "kia",
            "User-Agent": f"kia/{_package_version()}",
            "OpenAI-Beta": "responses=experimental",
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        }

    def _new_client(self, timeout: httpx.Timeout) -> httpx.Client:
        return httpx.Client(timeout=timeout)

    def _activate(self, client: httpx.Client) -> None:
        with self._client_lock:
            if self._active_client is not None:
                raise RuntimeError("Provider already has an active request")
            self._active_client = client

    def _release(self, client: httpx.Client) -> None:
        with self._client_lock:
            if self._active_client is client:
                self._active_client = None
        client.close()

    def open_stream(self, request: CompletionRequest) -> CompletionStream:
        credential = self._resolve_credential()
        timeout = httpx.Timeout(request.timeout or 600, connect=30)
        client = self._new_client(timeout)
        self._activate(client)
        try:
            headers = self._headers(credential)
            cache_key = _prompt_cache_key(request.session_id)
            if cache_key:
                headers["session-id"] = cache_key
                headers["x-client-request-id"] = cache_key
            outbound = client.build_request(
                "POST",
                CODEX_RESPONSES_URL,
                headers=headers,
                json=_build_body(request),
            )
            response = client.send(outbound, stream=True)
            if not response.is_success:
                response.read()
                error = self._http_error(response)
                response.close()
                raise error
        except BaseException:
            self._release(client)
            raise
        return _CodexCompletionStream(response, request.model, lambda: self._release(client))

    def complete(self, request: CompletionRequest) -> CompletionResult:
        stream = self.open_stream(replace(request, stream=True))
        try:
            return stream.consume()
        finally:
            stream.close()

    def _http_error(self, response: httpx.Response) -> ProviderError:
        code = None
        message = response.reason_phrase or "OpenAI Codex request failed"
        reset = None
        try:
            payload = response.json()
            error = payload.get("error") if isinstance(payload, dict) else {}
            if isinstance(error, dict):
                code = error.get("code") or error.get("type")
                message = error.get("message") or message
                reset = error.get("resets_at")
        except (json.JSONDecodeError, TypeError):
            pass
        if code in _USAGE_LIMIT_CODES or response.status_code == 429:
            suffix = ""
            if isinstance(reset, (int, float)):
                minutes = max(0, round((reset - time.time()) / 60))
                suffix = f" Try again in about {minutes} minutes."
            return ProviderError(
                f"ChatGPT Codex usage limit reached.{suffix}",
                status_code=response.status_code,
                code=code,
                retryable=False,
            )
        return ProviderError(
            message,
            status_code=response.status_code,
            code=code,
            retryable=response.status_code in _RETRYABLE_STATUS,
        )

    def login(self, interaction: AuthInteraction) -> None:
        credential = login_openai_codex(interaction)
        if interaction.cancelled():
            raise ProviderError(
                "OpenAI Codex login cancelled", code="cancelled", retryable=False
            )
        self._store.write_oauth(self.id, credential)
        if interaction.cancelled():
            self._store.delete(self.id)
            raise ProviderError(
                "OpenAI Codex login cancelled", code="cancelled", retryable=False
            )

    def logout(self) -> None:
        self._store.delete(self.id)

    def auth_status(self) -> str:
        try:
            credential = self._store.read_oauth(self.id)
        except Exception as e:
            return f"credential error: {e}"
        if credential is None:
            return "not logged in"
        if credential.expires <= time.time():
            return "OAuth token expired (will refresh on use)"
        return "logged in with ChatGPT OAuth"

    def cancel(self) -> None:
        with self._client_lock:
            client = self._active_client
            self._active_client = None
        if client is not None:
            client.close()

"""OpenAI-compatible Chat Completions provider."""

from __future__ import annotations

import threading
from typing import Any, Callable

from kiui.agent.models import reasoning_kwargs
from kiui.agent.utils.streaming import consume_stream, message_to_dict

from .registry import ProviderSettings
from .types import (
    CompletionRequest,
    CompletionResult,
    CompletionStream,
    LLMProvider,
    ProviderUsage,
)


def _detail_tokens(details: Any, name: str) -> int:
    if details is None:
        return 0
    return int(getattr(details, name, 0) or 0)


def normalize_usage(usage: Any) -> ProviderUsage | None:
    """Normalize an OpenAI SDK usage object."""
    if usage is None:
        return None
    prompt = int(usage.prompt_tokens or 0)
    completion = int(usage.completion_tokens or 0)
    return ProviderUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=int(usage.total_tokens or prompt + completion),
        cached_prompt_tokens=_detail_tokens(
            getattr(usage, "prompt_tokens_details", None), "cached_tokens"
        ),
        reasoning_tokens=_detail_tokens(
            getattr(usage, "completion_tokens_details", None), "reasoning_tokens"
        ),
    )


class _OpenAICompletionStream(CompletionStream):
    def __init__(self, raw_stream: Any, release: Callable[[], None]):
        self._raw_stream = raw_stream
        self._release = release
        self._closed = False

    def consume(self, *, on_content=None, on_thinking=None, should_stop=None) -> CompletionResult:
        message, usage, finish_reason = consume_stream(
            self._raw_stream,
            on_content=on_content,
            on_thinking=on_thinking,
            should_stop=should_stop,
        )
        return CompletionResult(message, normalize_usage(usage), finish_reason)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._raw_stream.close()
        finally:
            self._release()


class OpenAICompatibleProvider(LLMProvider):
    """Adapter for servers implementing OpenAI Chat Completions."""

    id = "openai"

    def __init__(self, settings: ProviderSettings):
        self._api_key = settings.api_key
        self._base_url = settings.base_url
        self._reasoning_style = settings.reasoning_style
        self._active_client: Any = None
        self._client_lock = threading.Lock()

    def _new_client(self):
        from openai import OpenAI

        return OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            max_retries=0,
        )

    def _activate(self, client: Any) -> None:
        with self._client_lock:
            if self._active_client is not None:
                raise RuntimeError("Provider already has an active request")
            self._active_client = client

    def _release(self, client: Any) -> None:
        with self._client_lock:
            if self._active_client is client:
                self._active_client = None
        client.close()

    def _kwargs(self, request: CompletionRequest) -> dict[str, Any]:
        messages = [
            {key: value for key, value in message.items() if key != "provider_state"}
            for message in request.messages
        ]
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": request.stream,
        }
        if request.max_output_tokens is not None:
            kwargs["max_tokens"] = request.max_output_tokens
        if request.stream:
            kwargs["stream_options"] = {"include_usage": True}
        if request.tools:
            kwargs["tools"] = request.tools
        if request.reasoning_effort is not None:
            kwargs.update(reasoning_kwargs(self._reasoning_style, request.reasoning_effort))
        if request.timeout is not None:
            kwargs["timeout"] = request.timeout
        return kwargs

    def complete(self, request: CompletionRequest) -> CompletionResult:
        if request.stream:
            raise ValueError("complete() requires stream=False")
        client = self._new_client()
        self._activate(client)
        try:
            response = client.chat.completions.create(**self._kwargs(request))
            choice = response.choices[0]
            return CompletionResult(
                message=message_to_dict(choice.message),
                usage=normalize_usage(response.usage),
                finish_reason=choice.finish_reason,
            )
        finally:
            self._release(client)

    def open_stream(self, request: CompletionRequest) -> CompletionStream:
        if not request.stream:
            raise ValueError("open_stream() requires stream=True")
        client = self._new_client()
        self._activate(client)
        try:
            raw_stream = client.chat.completions.create(**self._kwargs(request))
        except BaseException:
            self._release(client)
            raise
        return _OpenAICompletionStream(raw_stream, lambda: self._release(client))

    def cancel(self) -> None:
        with self._client_lock:
            client = self._active_client
            self._active_client = None
        if client is not None:
            client.close()

    def auth_status(self) -> str:
        return "API key configured" if self._api_key else "API key missing"

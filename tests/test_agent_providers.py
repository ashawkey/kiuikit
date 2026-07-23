"""Tests for provider-neutral completion adapters."""

from types import SimpleNamespace as NS

import pytest

from kiui.agent.providers import (
    CompletionRequest,
    OpenAICompatibleProvider,
    ProviderSettings,
    create_provider,
    provider_names,
)


def _message(content="done"):
    return NS(role="assistant", content=content, tool_calls=None, model_extra={})


def _usage():
    return NS(
        prompt_tokens=10,
        completion_tokens=4,
        total_tokens=14,
        prompt_tokens_details=NS(cached_tokens=3),
        completion_tokens_details=NS(reasoning_tokens=2),
    )


class _Client:
    def __init__(self, response):
        self.response = response
        self.kwargs = None
        self.close_calls = 0
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response

    def close(self):
        self.close_calls += 1


def test_registry_defaults_to_openai_adapter():
    assert set(provider_names()) >= {"openai", "openai-codex"}
    provider = create_provider("openai", ProviderSettings(api_key="key", base_url="url"))
    assert isinstance(provider, OpenAICompatibleProvider)


def test_registry_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider 'missing'"):
        create_provider("missing", ProviderSettings())


def test_openai_provider_builds_chat_request_and_normalizes_usage(monkeypatch):
    response = NS(
        choices=[NS(message=_message(), finish_reason="stop")],
        usage=_usage(),
    )
    client = _Client(response)
    provider = OpenAICompatibleProvider(
        ProviderSettings(api_key="key", base_url="url", reasoning_style="openai")
    )
    monkeypatch.setattr(provider, "_new_client", lambda: client)

    request = CompletionRequest(
        model="gpt-test",
        messages=[{
            "role": "user",
            "content": "hello",
            "provider_state": {"openai-codex": {"output": []}},
        }],
        tools=[{"type": "function", "function": {"name": "read_file"}}],
        stream=False,
        max_output_tokens=1234,
        reasoning_effort="high",
        timeout=60,
    )
    result = provider.complete(request)

    assert client.kwargs == {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
        "max_tokens": 1234,
        "tools": request.tools,
        "reasoning_effort": "high",
        "timeout": 60,
    }
    assert result.message == {"role": "assistant", "content": "done"}
    assert result.usage.prompt_tokens == 10
    assert result.usage.cached_prompt_tokens == 3
    assert result.usage.reasoning_tokens == 2
    assert result.finish_reason == "stop"
    assert client.close_calls == 1


def test_openai_provider_stream_is_closed_and_normalized(monkeypatch):
    chunks = [
        NS(
            choices=[NS(
                finish_reason=None,
                delta=NS(
                    role="assistant",
                    content="hello",
                    reasoning=None,
                    reasoning_content=None,
                    tool_calls=None,
                    model_extra={},
                ),
            )],
            usage=None,
        ),
        NS(choices=[], usage=_usage()),
    ]

    class RawStream(list):
        close_calls = 0

        def close(self):
            self.close_calls += 1

    raw_stream = RawStream(chunks)
    client = _Client(raw_stream)
    provider = OpenAICompatibleProvider(ProviderSettings(api_key="key", base_url="url"))
    monkeypatch.setattr(provider, "_new_client", lambda: client)
    parts = []

    stream = provider.open_stream(CompletionRequest(
        model="gpt-test",
        messages=[],
        stream=True,
    ))
    try:
        result = stream.consume(on_content=parts.append)
    finally:
        stream.close()

    assert client.kwargs["stream"] is True
    assert client.kwargs["stream_options"] == {"include_usage": True}
    assert result.message["content"] == "hello"
    assert result.usage.total_tokens == 14
    assert parts == ["hello"]
    assert raw_stream.close_calls == 1
    assert client.close_calls == 1

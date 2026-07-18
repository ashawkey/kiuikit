"""Streaming response accumulation for the kia agent.

The OpenAI-compatible streaming API yields a sequence of chunks, each carrying
a ``delta`` with partial content, partial reasoning ("thinking"), and partial
tool-call arguments. :func:`consume_stream` folds those chunks back into a
single :class:`~openai.types.chat.ChatCompletionMessage`-shaped object plus a
usage object, so the rest of the agent (context, tool dispatch) can treat a
streamed turn exactly like a non-streamed one.

While folding, it invokes callbacks so the UI can render tokens as they arrive:

* ``on_content(text)``  — a chunk of visible assistant text.
* ``on_thinking(text)`` — a chunk of reasoning/thinking text (model-dependent).

Reasoning is exposed under different keys across providers; we probe the common
ones (``reasoning_content`` for DeepSeek-style, ``reasoning`` for OpenAI/Gemini
proxies) on both the typed delta and its ``model_extra`` bag.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from kiui.agent.interrupt import RequestInterrupted
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)


# Delta attribute names that carry reasoning/thinking text, most specific first.
_REASONING_KEYS = ("reasoning_content", "reasoning")


def _extract_reasoning(delta: Any) -> str | None:
    """Return the reasoning text on *delta*, probing known provider keys."""
    for key in _REASONING_KEYS:
        value = getattr(delta, key, None)
        if value:
            return value
        extra = getattr(delta, "model_extra", None)
        if isinstance(extra, dict) and extra.get(key):
            return extra[key]
    return None


class _ToolCallAccumulator:
    """Reassemble a tool call streamed as a sequence of argument fragments."""

    def __init__(self) -> None:
        self.id: str | None = None
        self.name: str = ""
        self.arguments: str = ""

    def update(self, delta_tool_call: Any) -> None:
        if delta_tool_call.id:
            self.id = delta_tool_call.id
        function = delta_tool_call.function
        if function is not None:
            if function.name:
                self.name = function.name
            if function.arguments:
                self.arguments += function.arguments

    def build(self) -> ChatCompletionMessageToolCall:
        return ChatCompletionMessageToolCall(
            id=self.id or "",
            type="function",
            function=Function(name=self.name, arguments=self.arguments),
        )


def consume_stream(
    stream: Iterable[Any],
    on_content: Callable[[str], None] | None = None,
    on_thinking: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[ChatCompletionMessage, Any]:
    """Fold a streamed chat completion into ``(message, usage)``.

    Callbacks fire synchronously as chunks arrive. If *should_stop* returns
    true, cancellation is raised so a partial response cannot enter context.
    """
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, _ToolCallAccumulator] = {}
    usage: Any = None
    role = "assistant"

    for chunk in stream:
        if should_stop is not None and should_stop():
            raise RequestInterrupted()

        # The final usage-only chunk (stream_options.include_usage) has no
        # choices; capture it and continue.
        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        if delta is None:
            continue
        if delta.role:
            role = delta.role

        thinking = _extract_reasoning(delta)
        if thinking:
            reasoning_parts.append(thinking)
            if on_thinking is not None:
                on_thinking(thinking)

        if delta.content:
            content_parts.append(delta.content)
            if on_content is not None:
                on_content(delta.content)

        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                acc = tool_calls.get(tc_delta.index)
                if acc is None:
                    acc = _ToolCallAccumulator()
                    tool_calls[tc_delta.index] = acc
                acc.update(tc_delta)

    if should_stop is not None and should_stop():
        raise RequestInterrupted()

    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    built_tool_calls = [tool_calls[i].build() for i in sorted(tool_calls)]

    message = ChatCompletionMessage(
        role=role,
        content=content or None,
        tool_calls=built_tool_calls or None,
    )
    # Preserve reasoning so /context and session dumps can show it; harmless to
    # send back to the API (unknown fields are ignored by extra="allow").
    if reasoning:
        message.reasoning_content = reasoning

    return message, usage

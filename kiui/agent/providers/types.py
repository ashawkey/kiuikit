"""Provider-neutral completion contracts for the kia agent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from kiui.agent.models import ReasoningEffort


class ProviderError(RuntimeError):
    """Provider failure with retry semantics understood by the agent loop."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        retryable: bool | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class AuthInteraction:
    """Provider-neutral callbacks used by interactive authentication flows."""

    select: Callable[[str, list[str]], str | None]
    prompt: Callable[[str], str | None]
    notify: Callable[[str], None]
    cancelled: Callable[[], bool] = lambda: False


@dataclass(frozen=True)
class CompletionRequest:
    """One model request expressed in kia's canonical message format."""

    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)
    stream: bool = True
    max_output_tokens: int | None = None
    reasoning_effort: ReasoningEffort | None = None
    session_id: str | None = None
    timeout: float | None = None


@dataclass(frozen=True)
class ProviderUsage:
    """Token usage normalized across provider APIs."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_prompt_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass(frozen=True)
class CompletionResult:
    """A completed assistant turn in kia's canonical message format."""

    message: dict[str, Any]
    usage: ProviderUsage | None
    finish_reason: str | None


class CompletionStream(ABC):
    """An opened provider stream that can be consumed exactly once."""

    @abstractmethod
    def consume(
        self,
        *,
        on_content: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> CompletionResult:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class LLMProvider(ABC):
    """Transport adapter from kia requests to one provider API."""

    id: str

    @abstractmethod
    def complete(self, request: CompletionRequest) -> CompletionResult:
        """Execute a non-streaming request."""

    @abstractmethod
    def open_stream(self, request: CompletionRequest) -> CompletionStream:
        """Open a streaming request without consuming its response body."""

    @abstractmethod
    def cancel(self) -> None:
        """Cancel the provider's active request, if any."""

    def login(self, interaction: AuthInteraction) -> None:
        raise ProviderError(f"Provider '{self.id}' does not support interactive login")

    def logout(self) -> None:
        raise ProviderError(f"Provider '{self.id}' does not store login credentials")

    def auth_status(self) -> str:
        return "configured"

    def close(self) -> None:
        self.cancel()

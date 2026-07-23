"""LLM provider interfaces and built-in provider registry."""

from .openai_codex import OpenAICodexProvider
from .openai_compatible import OpenAICompatibleProvider
from .registry import (
    ProviderSettings,
    create_provider,
    provider_names,
    register_provider,
)
from .types import (
    AuthInteraction,
    CompletionRequest,
    CompletionResult,
    CompletionStream,
    LLMProvider,
    ProviderError,
    ProviderUsage,
)

register_provider("openai", OpenAICompatibleProvider)
register_provider("openai-codex", OpenAICodexProvider)

__all__ = [
    "AuthInteraction",
    "CompletionRequest",
    "CompletionResult",
    "CompletionStream",
    "LLMProvider",
    "OpenAICodexProvider",
    "OpenAICompatibleProvider",
    "ProviderError",
    "ProviderSettings",
    "ProviderUsage",
    "create_provider",
    "provider_names",
    "register_provider",
]

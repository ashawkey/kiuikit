"""Provider registry and construction from one configured model entry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .types import LLMProvider


@dataclass(frozen=True)
class ProviderSettings:
    """Connection settings shared by the current built-in provider factory."""

    api_key: str = ""
    base_url: str = ""
    reasoning_style: str | None = None


ProviderFactory = Callable[[ProviderSettings], LLMProvider]
_FACTORIES: dict[str, ProviderFactory] = {}


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Register a provider factory, rejecting accidental replacement."""
    if name in _FACTORIES:
        raise ValueError(f"Provider already registered: {name}")
    _FACTORIES[name] = factory


def create_provider(name: str, settings: ProviderSettings) -> LLMProvider:
    """Create a configured provider or fail with the available provider names."""
    factory = _FACTORIES.get(name)
    if factory is None:
        available = ", ".join(sorted(_FACTORIES))
        raise ValueError(f"Unknown provider '{name}'. Available: {available}")
    return factory(settings)


def provider_names() -> tuple[str, ...]:
    return tuple(sorted(_FACTORIES))

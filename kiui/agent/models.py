"""Model catalog: maps model identifiers to context-window sizes and
thinking-budget styles so the agent can auto-configure itself."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelProfile:
    """Properties of a model family that affect API behaviour."""

    context_window: int = 128_000
    thinking: str | None = None  # "openai" | "gemini" | None


# Ordered most-specific → least-specific within each family.
# Matching is case-insensitive substring; first hit wins.
MODEL_CATALOG: list[tuple[str, ModelProfile]] = [
    # --- OpenAI / o-series ---
    ("gpt-5",    ModelProfile(context_window=1_000_000, thinking="openai")),
    ("gpt-4o",   ModelProfile(context_window=128_000)),

    # --- Google Gemini ---
    ("gemini-3",   ModelProfile(context_window=1_000_000, thinking="gemini")),
    ("gemini",     ModelProfile(context_window=1_000_000, thinking="gemini")),

    # --- Anthropic Claude ---
    ("claude-opus-4",   ModelProfile(context_window=200_000)),
    ("claude-sonnet-4", ModelProfile(context_window=200_000)),
    ("claude",          ModelProfile(context_window=200_000)),

    # --- DeepSeek ---
    ("deepseek-reasoner", ModelProfile(context_window=128_000)),
    ("deepseek-chat",     ModelProfile(context_window=128_000)),
    ("deepseek",          ModelProfile(context_window=128_000)),
]

DEFAULT_PROFILE = ModelProfile()


def resolve_model_profile(model_id: str, model_key: str = "") -> ModelProfile:
    """Match a model identifier against the catalog.

    Tries *model_id* first, then *model_key* as a fallback.
    Returns ``DEFAULT_PROFILE`` when nothing matches.
    """
    for candidate in (model_id, model_key):
        if not candidate:
            continue
        lower = candidate.lower()
        for pattern, profile in MODEL_CATALOG:
            if pattern in lower:
                return profile
    return DEFAULT_PROFILE

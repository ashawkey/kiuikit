"""Model capabilities and provider-specific reasoning configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")


@dataclass(frozen=True)
class ModelProfile:
    """Properties of a model family that affect API behaviour."""

    context_length: int = 128_000
    reasoning: str | None = None  # "openai" | "anthropic" | "gemini" | "deepseek"


# Ordered most-specific → least-specific within each family.
# Matching is case-insensitive substring; first hit wins.
MODEL_CATALOG: list[tuple[str, ModelProfile]] = [
    ("gpt-5", ModelProfile(context_length=1_000_000, reasoning="openai")),
    ("o1", ModelProfile(reasoning="openai")),
    ("o3", ModelProfile(reasoning="openai")),
    ("o4", ModelProfile(reasoning="openai")),
    ("gpt-4o", ModelProfile(context_length=128_000)),
    ("gemini", ModelProfile(context_length=1_000_000, reasoning="gemini")),
    ("claude", ModelProfile(context_length=1_000_000, reasoning="anthropic")),
    ("deepseek", ModelProfile(context_length=1_000_000, reasoning="deepseek")),
]

DEFAULT_PROFILE = ModelProfile()


def resolve_model_profile(model_id: str, model_alias: str = "") -> ModelProfile:
    """Resolve a model ID, falling back to its configured alias."""
    for candidate in (model_id, model_alias):
        lower = candidate.lower()
        for pattern, profile in MODEL_CATALOG:
            if pattern in lower:
                return profile
    return DEFAULT_PROFILE


def reasoning_kwargs(style: str | None, effort: ReasoningEffort) -> dict[str, Any]:
    """Translate normalized reasoning effort to OpenAI-compatible API fields."""
    if style is None:
        return {}
    if style == "openai":
        return {"reasoning_effort": effort}
    if style == "anthropic":
        # OpenAI-compatible Anthropic gateways commonly consume both the generic
        # effort field and the native adaptive-thinking controls.
        mapped = "max" if effort == "xhigh" else effort
        if effort == "none":
            return {"extra_body": {"thinking": {"type": "disabled"}}}
        return {
            "reasoning_effort": effort,
            "extra_body": {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": mapped},
            },
        }
    if style == "gemini":
        # Gemini 3 uses qualitative thinking levels; it has no xhigh level.
        mapped = {"none": "minimal", "xhigh": "high"}.get(effort, effort)
        return {
            "extra_body": {
                "google": {
                    "thinking_config": {
                        "thinking_level": mapped,
                        "include_thoughts": True,
                    }
                }
            }
        }
    if style == "deepseek":
        if effort == "none":
            return {"extra_body": {"thinking": {"type": "disabled"}}}
        # DeepSeek officially maps low/medium to high and xhigh to max.
        mapped = "max" if effort == "xhigh" else "high"
        return {
            "reasoning_effort": mapped,
            "extra_body": {"thinking": {"type": "enabled"}},
        }
    raise ValueError(f"Unknown reasoning style: {style}")

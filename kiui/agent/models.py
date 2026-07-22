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
    reasoning: str | None = None  # "openai" | "anthropic" | "gemini" | "deepseek" | "kimi"
    supports_image_input: bool = False
    # Max output tokens per request. Reasoning tokens count against this budget,
    # so reasoning models need generous ceilings to avoid mid-tool-call
    # `finish_reason="length"` truncation. Kept below each provider's hard cap.
    max_output_tokens: int = 32_000


# Ordered most-specific → least-specific within each family.
# Matching is case-insensitive substring; first hit wins.
MODEL_CATALOG: list[tuple[str, ModelProfile]] = [
    ("gpt-5", ModelProfile(context_length=258_000, reasoning="openai", supports_image_input=True, max_output_tokens=128_000)),
    ("gpt", ModelProfile(supports_image_input=True)),
    ("gemini", ModelProfile(context_length=1_000_000, reasoning="gemini", supports_image_input=True, max_output_tokens=64_000)),
    ("claude", ModelProfile(context_length=1_000_000, reasoning="anthropic", supports_image_input=True, max_output_tokens=64_000)),
    ("deepseek", ModelProfile(context_length=1_000_000, reasoning="deepseek", max_output_tokens=64_000)),
    ("glm", ModelProfile(context_length=1_000_000, reasoning="deepseek", max_output_tokens=64_000)),
    ("kimi-k3", ModelProfile(context_length=1_000_000, reasoning="kimi", supports_image_input=True, max_output_tokens=64_000)),
    ("kimi", ModelProfile(supports_image_input=True)),
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
    if style == "kimi":
        # Kimi K3 always reasons (thinking cannot be disabled) and currently
        # only accepts reasoning_effort="max"; all effort levels map to it.
        return {"reasoning_effort": "max"}
    raise ValueError(f"Unknown reasoning style: {style}")

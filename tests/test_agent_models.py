from kiui.agent.models import reasoning_kwargs, resolve_model_profile


def test_reasoning_profiles():
    assert resolve_model_profile("openai/gpt-5.6").reasoning == "openai"
    assert resolve_model_profile("anthropic/claude-opus-4-8").reasoning == "anthropic"
    assert resolve_model_profile("google/gemini-3.1-pro").reasoning == "gemini"
    assert resolve_model_profile("deepseek-v4-pro").reasoning == "deepseek"


def test_openai_reasoning():
    assert reasoning_kwargs("openai", "xhigh") == {"reasoning_effort": "xhigh"}


def test_anthropic_reasoning():
    kwargs = reasoning_kwargs("anthropic", "xhigh")
    assert kwargs["reasoning_effort"] == "xhigh"
    assert kwargs["extra_body"]["thinking"] == {"type": "adaptive"}
    assert kwargs["extra_body"]["output_config"] == {"effort": "max"}


def test_gemini_reasoning():
    config = reasoning_kwargs("gemini", "xhigh")["extra_body"]["google"]["thinking_config"]
    assert config == {"thinking_level": "high", "include_thoughts": True}


def test_deepseek_reasoning():
    assert reasoning_kwargs("deepseek", "high")["reasoning_effort"] == "high"
    assert reasoning_kwargs("deepseek", "xhigh")["reasoning_effort"] == "max"
    assert reasoning_kwargs("deepseek", "none") == {
        "extra_body": {"thinking": {"type": "disabled"}}
    }

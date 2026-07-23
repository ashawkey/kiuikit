"""Tests for CLI model configuration."""

from kiui.config import conf
from kiui.agent.cli import Args, get_agent


def test_get_agent_passes_token_limits(monkeypatch):
    created = []

    class FakeAgent:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setitem(conf, "openai", {
        "test": {
            "model": "test-model",
            "api_key": "key",
            "base_url": "url",
            "context_length": 200_000,
            "max_output_tokens": 16_000,
        }
    })
    monkeypatch.setattr("kiui.agent.backend.LLMAgent", FakeAgent)
    monkeypatch.setattr("kiui.agent.hub.discover_hub", lambda port: None)

    agent, hub_client = get_agent(Args(model="test"))

    assert agent is not None
    assert hub_client is None
    assert created[0]["provider_name"] == "openai"
    assert created[0]["context_length"] == 200_000
    assert created[0]["max_output_tokens"] == 16_000

"""Tests for CLI model configuration."""

from kiui.config import conf
from kiui.agent import cli
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


def test_clean_accepts_entry_names(monkeypatch):
    cleaned = []
    monkeypatch.setattr("sys.argv", ["kia", "--clean", "pdf-cache", "sessions"])
    monkeypatch.setattr(cli.tyro, "cli", lambda *args, **kwargs: Args())
    monkeypatch.setattr(cli, "cmd_clean", lambda names: cleaned.extend(names))

    cli.main()

    assert cleaned == ["pdf-cache", "sessions"]

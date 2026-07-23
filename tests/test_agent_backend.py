"""Tests for backend helpers that don't need a live API."""

from contextlib import nullcontext

import pytest

from kiui.agent.backend import _is_fatal_api_error
from kiui.agent.backend.commands import AgentCommandsMixin
from kiui.agent.providers import ProviderError


class _StatusError(Exception):
    """Mimics an openai.APIStatusError instance carrying ``status_code``."""

    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_fatal_client_errors_are_not_retried(status):
    assert _is_fatal_api_error(_StatusError(status)) is True


@pytest.mark.parametrize("status", [408, 409, 425, 429, 500, 502, 503])
def test_transient_errors_are_retried(status):
    assert _is_fatal_api_error(_StatusError(status)) is False


def test_oauth_commands_use_current_provider():
    output = []

    class Console:
        def select(self, message, choices):
            return choices[0]

        def ask_text(self, message):
            return "code"

        def print(self, message, **kwargs):
            output.append(message)

        def system(self, message):
            output.append(message)

        def error(self, message):
            pytest.fail(message)

        def thinking(self, **kwargs):
            output.append(kwargs["label"])
            return nullcontext()

    class Provider:
        def __init__(self):
            self.logged_in = False

        def login(self, interaction):
            assert interaction.select("method", ["browser"]) == "browser"
            assert interaction.prompt("code") == "code"
            interaction.notify("open URL")
            self.logged_in = True

        def logout(self):
            self.logged_in = False

        def auth_status(self):
            return "logged in" if self.logged_in else "not logged in"

    agent = type("Agent", (AgentCommandsMixin,), {})()
    agent.console = Console()
    agent.provider = Provider()
    agent.provider_name = "openai-codex"
    agent.model_alias = "codex"
    agent.cancellation = None
    agent._operation = lambda label: nullcontext()

    agent._cmd_login("/login")
    agent._cmd_auth("/auth")
    agent._cmd_logout("/logout")

    assert "Authenticating" in output
    assert any("Logged in to openai-codex" in line for line in output)
    assert output[-1] == "Logged out of openai-codex."


def test_provider_retry_classification_overrides_http_status():
    assert _is_fatal_api_error(
        ProviderError("subscription limit", status_code=429, retryable=False)
    ) is True
    assert _is_fatal_api_error(
        ProviderError("temporary", status_code=400, retryable=True)
    ) is False

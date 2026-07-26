"""Tests for backend helpers that don't need a live API."""

import asyncio
import queue
import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest

import kiui.agent.backend as backend
from kiui.agent.backend import LLMAgent, _is_fatal_api_error
from kiui.agent.backend.commands import AgentCommandsMixin
from kiui.agent.providers import ProviderError
from kiui.agent.utils.io import EventHub, InputBroker


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


@pytest.mark.parametrize(
    "query, instant",
    [
        ("/usage", True),
        ("/context", True),
        ("/perm auto", True),
        ("/goal clear", True),
        ("/model", True),        # bare form only lists
        ("/model gpt-5", False),  # switching swaps the provider mid-round
        ("/skills", True),
        ("/skills reload", False),
        ("/persona reload", False),
        ("/clear", False),
        ("/compact", False),
        ("/rewind", False),
        ("/exit", False),
        ("/nonsense", True),     # a typo is reported straight away
    ],
)
def test_instant_command_classification(query, instant):
    agent = type("Agent", (AgentCommandsMixin,), {})()
    assert agent.is_instant_command(query) is instant


def _busy_agent(broker):
    """Agent stub standing in for one whose round runs on the worker thread."""
    dispatched = []
    echoed = []

    class Agent(AgentCommandsMixin):
        input_broker = broker
        console = NS(user_input=lambda text, **kwargs: echoed.append((text, kwargs)))

        def _run_command(self, query):
            dispatched.append(query)

    return Agent(), dispatched, echoed


def test_instant_command_runs_while_a_round_is_in_flight():
    broker = InputBroker(EventHub())
    broker.submit("/usage", source="web")
    agent, dispatched, echoed = _busy_agent(broker)

    assert LLMAgent._run_instant_command(agent) is True
    assert dispatched == ["/usage"]
    assert echoed[0][0] == "/usage"
    assert echoed[0][1]["source"] == "web"
    # Consumed, so the prompt stops advertising it as queued.
    assert broker.submission is None


@pytest.mark.parametrize("query", ["/clear", "steer the agent", "!git status"])
def test_non_instant_input_stays_queued_while_busy(query):
    broker = InputBroker(EventHub())
    submission = broker.submit(query)
    agent, dispatched, _ = _busy_agent(broker)

    assert LLMAgent._run_instant_command(agent) is False
    assert dispatched == []
    assert broker.submission == submission


def _wait_until(condition, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


class _ScriptedTerminal:
    """Terminal stub whose prompt returns lines the test feeds from outside."""

    def __init__(self):
        self.lines: queue.Queue[str] = queue.Queue()
        self.busy: list[bool] = []
        self.app = NS(invalidate=lambda: None, is_running=True)
        self.text = ""

    async def prompt_async(self, default: str = "") -> str:
        # A default means the loop handed a rejected draft back to the editor;
        # the script feeds one line at a time, so that must never happen.
        assert not default, f"unexpected draft returned to the editor: {default!r}"
        while True:
            try:
                return self.lines.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.005)

    def set_runtime_state(self, **kwargs) -> None:
        pass

    def set_busy(self, busy: bool) -> None:
        self.busy.append(busy)

    def set_status(self, status) -> None:
        pass


def test_terminal_loop_answers_a_command_without_waiting_for_the_round(monkeypatch):
    """A command typed mid-round runs now; anything else waits its turn."""
    # Routing, not rendering, is under test: stdout patching needs a real console.
    monkeypatch.setattr(backend, "patch_stdout", lambda **kwargs: nullcontext())
    broker = InputBroker(EventHub())
    round_started = threading.Event()
    release_round = threading.Event()
    dispatched: list[str] = []

    class Agent(AgentCommandsMixin):
        input_broker = broker
        console = NS(user_input=lambda text, **kwargs: None, warn=lambda *a, **k: None)
        cancellation = None
        prompt_broker = None
        verbose = False
        _run_instant_command = LLMAgent._run_instant_command
        _run_round = LLMAgent._run_round

        def _process_next_submission(self) -> bool:
            text = broker.get_nowait().text
            if text == "exit":
                return True
            if text.startswith("/"):  # as _process_query routes a queued command
                self._run_command(text)
                return False
            round_started.set()
            release_round.wait(5)
            return False

        def _queue_pending_auto(self) -> None:
            pass

        def _run_command(self, query: str) -> bool:
            dispatched.append(query)
            return False

    agent = Agent()
    terminal = _ScriptedTerminal()
    loop = threading.Thread(
        target=lambda: LLMAgent._run_terminal_loop(agent, terminal), daemon=True
    )
    loop.start()

    terminal.lines.put("do some work")
    assert round_started.wait(5)

    # Answered with the round still blocked on the worker thread.
    terminal.lines.put("/usage")
    assert _wait_until(lambda: dispatched == ["/usage"])
    assert not release_round.is_set()

    # A command that touches the conversation queues instead.
    terminal.lines.put("/clear")
    assert _wait_until(lambda: broker.submission is not None)
    assert broker.submission.text == "/clear"
    assert dispatched == ["/usage"]

    release_round.set()
    assert _wait_until(lambda: dispatched == ["/usage", "/clear"])

    terminal.lines.put("exit")
    loop.join(timeout=5)
    assert not loop.is_alive()


def test_terminal_loop_survives_a_failing_round(monkeypatch):
    """One bad turn is reported and the session stays at the prompt.

    Regression: an exception escaping the worker thread propagated out of the
    UI loop, killing the whole chat session (and raising a second time from the
    loop's own cleanup).
    """
    monkeypatch.setattr(backend, "patch_stdout", lambda **kwargs: nullcontext())
    broker = InputBroker(EventHub())
    warnings: list[str] = []

    class Agent(AgentCommandsMixin):
        input_broker = broker
        console = NS(
            user_input=lambda text, **kwargs: None,
            warn=lambda msg, **kwargs: warnings.append(msg),
        )
        cancellation = None
        prompt_broker = None
        verbose = False
        _run_instant_command = LLMAgent._run_instant_command
        _run_round = LLMAgent._run_round

        def _process_next_submission(self) -> bool:
            if broker.get_nowait().text == "exit":
                return True
            raise RuntimeError("round blew up")

        def _queue_pending_auto(self) -> None:
            pass

        def _run_command(self, query: str) -> bool:
            return False

    terminal = _ScriptedTerminal()
    loop = threading.Thread(
        target=lambda: LLMAgent._run_terminal_loop(Agent(), terminal), daemon=True
    )
    loop.start()

    terminal.lines.put("do some work")
    assert _wait_until(lambda: any("round blew up" in w for w in warnings))
    assert loop.is_alive()  # the session outlived the failure

    terminal.lines.put("exit")  # and still accepts input
    loop.join(timeout=5)
    assert not loop.is_alive()


def test_provider_retry_classification_overrides_http_status():
    assert _is_fatal_api_error(
        ProviderError("subscription limit", status_code=429, retryable=False)
    ) is True
    assert _is_fatal_api_error(
        ProviderError("temporary", status_code=400, retryable=True)
    ) is False

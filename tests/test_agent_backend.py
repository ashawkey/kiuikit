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
from kiui.agent.context import CompactionState, ContextManager
from kiui.agent.providers import ProviderError
from kiui.agent.utils.interrupt import RequestInterrupted
from kiui.agent.utils.io import EventHub, InputBroker, UserSubmission


def test_agent_close_releases_resources_once():
    calls = []
    agent = LLMAgent.__new__(LLMAgent)
    agent._closed = False
    agent.changes = NS(close=lambda: calls.append("changes"))
    agent.provider = NS(close=lambda: calls.append("provider"))
    agent.tool_executor = NS(
        shutdown_processes=lambda: calls.append("processes"),
        shutdown_tool_resources=lambda clear=False: calls.append(("resources", clear)),
    )

    agent.close()
    agent.close()

    assert calls == ["changes", "provider", "processes", ("resources", True)]


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
        ("/ps", True),
        ("/ps p-12345678", True),
        ("/context", True),
        ("/wait 1h later", False),
        ("/model", True),        # bare form only lists
        ("/model gpt-5", False),  # switching swaps the provider mid-round
        ("/skills", True),
        ("/skills reload", False),
        ("/persona reload", False),
        ("/clear", False),
        ("/compact", False),
        ("/continue", False),
        ("/rewind", False),
        ("/exit", False),
        ("/nonsense", True),     # a typo is reported straight away
    ],
)
def test_instant_command_classification(query, instant):
    agent = type("Agent", (AgentCommandsMixin,), {})()
    assert agent.is_instant_command(query) is instant


def test_process_status_callback_publishes_without_inspection(tmp_path):
    events = EventHub()
    agent = LLMAgent.__new__(LLMAgent)
    agent.events = events
    agent._process_status_sink = None
    agent.tool_executor = backend.ToolExecutor(work_dir=str(tmp_path))
    agent.tool_executor.set_process_status_callback(agent._process_status_changed)

    started = agent.tool_executor.execute(
        "start_process", {"command": "python -c 'import time; time.sleep(.2)'"}
    )
    try:
        assert _wait_until(lambda: agent.tool_executor.process_counts() == (0, 1))
        statuses = [event for event in events.after(0) if event.type == "process_status"]
        assert any(event.data["running"] == 1 for event in statuses)
        assert statuses[-1].data["finished"] == 1
        assert statuses[-1].data["text"] == ""
        assert started["process_id"]
    finally:
        agent.tool_executor.shutdown_processes()


def test_ps_lists_processes_and_shows_detail_tail():
    output = []
    calls = []

    class Console:
        def print(self, message):
            output.append(str(message))

        def system(self, message):
            output.append(str(message))

        def warn(self, message):
            output.append(str(message))

    process = {
        "process_id": "p-12345678",
        "pid": 42,
        "status": "running",
        "exit_code": None,
        "command": "python worker.py --long-option value",
        "cwd": "/tmp/work",
        "log_path": ".kia/processes/p-12345678.log",
        "log_tail": "ready\n",
    }
    agent = type("Agent", (AgentCommandsMixin,), {})()
    agent.console = Console()
    agent.tool_executor = NS(inspect_processes=lambda **kwargs: (
        calls.append(kwargs) or {"success": True, "processes": [process]}
    ))

    agent._cmd_ps("/ps")
    agent._cmd_ps("/ps p-12345678")

    assert calls == [
        {"process_id": None, "log_tail_chars": 0},
        {"process_id": "p-12345678", "log_tail_chars": 8000},
    ]
    assert "p-12345678" in output[0]
    assert any("Recent output" in item and "ready" in item for item in output)


def test_slash_command_catalog_includes_skills_without_shadowing_builtins():
    agent = type("Agent", (AgentCommandsMixin,), {})()
    agent.skills = {
        "monitor-jobs": {"description": "Monitor jobs."},
        "usage": {"description": "Collides with a built-in."},
    }

    catalog = agent._slash_command_help()

    assert catalog["monitor-jobs"] == "Skill — Monitor jobs."
    assert catalog["usage"] == agent.COMMAND_HELP["usage"]


def test_skill_invocation_is_not_instant_and_cannot_shadow_builtin():
    agent = type("Agent", (AgentCommandsMixin,), {})()
    agent.skills = {
        "monitor-jobs": {"description": "Monitor jobs."},
        "usage": {"description": "Collides with a built-in."},
    }

    assert agent.is_instant_command("/monitor-jobs") is False
    assert agent.is_instant_command("/monitor-jobs training") is False
    assert agent.is_instant_command("/usage") is True


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


@pytest.mark.parametrize("query", ["/clear", "/wait 1h later", "steer the agent", "!git status"])
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


def _skill_invocation_agent(tmp_path, name="general-skill"):
    skill_dir = tmp_path / ".kia" / "skills" / name
    skill_dir.mkdir(parents=True)
    body = "General instructions."
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Use for tests.\n---\n{body}\n",
        encoding="utf-8",
    )
    agent = object.__new__(LLMAgent)
    agent.skills = backend.discover_skills(tmp_path)
    agent.context = ContextManager("system")
    agent.tool_executor = backend.ToolExecutor(work_dir=str(tmp_path), skills=agent.skills)
    agent.console = NS(
        rule=lambda *a, **k: None,
        user_input=lambda *a, **k: None,
        warn=lambda *a, **k: None,
        reset_timeline=lambda: None,
    )
    agent.round_id = 0
    agent._session_id = "test"
    agent._session_revision_id = None
    agent._compaction_floor_tokens = None
    agent._pending_images = []
    agent._last_interrupted = False
    agent.verbose = False
    agent._operation = lambda _label: nullcontext()
    agent.save_session = lambda *a, **k: None
    agent.get_response = lambda: None
    return agent, body


def test_explicit_skill_invocation_without_task_asks_model_not_to_infer(tmp_path):
    agent, body = _skill_invocation_agent(tmp_path)

    agent._process_query(UserSubmission("/general-skill", "terminal", "s1"))

    call, result, message = agent.context.messages
    tool_call = call["tool_calls"][0]
    assert call["role"] == "assistant"
    assert tool_call["function"]["name"] == "load_skill"
    assert tool_call["function"]["arguments"] == '{"name": "general-skill"}'
    assert result["role"] == "tool"
    assert result["tool_call_id"] == tool_call["id"]
    assert body in result["content"]
    assert message["display_content"] == "/general-skill"
    assert body not in message["content"]
    assert "Default invocation" in message["content"]
    assert "do not infer or start a task" in message["content"]
    state = CompactionState().absorb(agent.context.messages)
    assert state.original_request == message["content"]
    assert state.skills == ("general-skill",)
    assert agent.round_id == 1


def test_explicit_skill_invocation_with_task_and_when_already_loaded(tmp_path):
    agent, body = _skill_invocation_agent(tmp_path)

    agent._process_query(UserSubmission("/general-skill first task", "web", "s1"))
    agent._process_query(UserSubmission("/general-skill second task", "web", "s2"))

    first_call, first_result, first, second_call, second_result, second = (
        agent.context.messages
    )
    assert first_call["tool_calls"][0]["function"]["name"] == "load_skill"
    assert body in first_result["content"]
    assert first["content"] == "first task"
    assert first["display_content"] == "/general-skill first task"
    assert second_call["tool_calls"][0]["function"]["name"] == "load_skill"
    assert body in second_result["content"]
    assert second["content"] == "second task"
    assert second["display_content"] == "/general-skill second task"
    state = CompactionState().absorb(agent.context.messages)
    assert state.original_request == "first task"
    assert state.skills == ("general-skill",)
    assert agent.round_id == 2


def test_manual_skill_load_records_tool_pair_without_user_message(tmp_path):
    agent, body = _skill_invocation_agent(tmp_path)
    agent.console.system = lambda *a, **k: None

    agent._cmd_skills("/skills general-skill")

    call, result = agent.context.messages
    tool_call = call["tool_calls"][0]
    assert call["role"] == "assistant"
    assert tool_call["function"]["name"] == "load_skill"
    assert result["role"] == "tool"
    assert result["tool_call_id"] == tool_call["id"]
    assert body in result["content"]
    assert not any(message["role"] == "user" for message in agent.context.messages)
    assert agent.round_id == 0


def test_cancelled_skill_invocation_restores_skill_state(tmp_path):
    agent, _ = _skill_invocation_agent(tmp_path)
    agent.events = EventHub()
    agent.console.system = lambda *a, **k: None
    agent._set_rewind_draft = lambda *a, **k: None

    def interrupt():
        agent._last_interrupted = True
        agent._interrupt_reverts_prompt = True

    agent.get_response = interrupt
    agent._process_query(UserSubmission("/general-skill", "terminal", "s1"))

    assert agent.context.messages == []
    assert agent.tool_executor._loaded_skills == set()
    assert agent.tool_executor._skill_loads == {}


def test_cancelled_initial_request_restores_context_and_message_draft():
    context = ContextManager("system")
    context.add({"role": "user", "content": "u1"})
    context.add({"role": "assistant", "content": "a1"})
    messages_before = list(context.messages)
    state_before = CompactionState(original_request="u1")
    context.compaction_state = state_before

    events = EventHub()
    resets = []
    saved = []
    console = NS(
        rule=lambda: None,
        user_input=lambda *args, **kwargs: None,
        response=lambda *args, **kwargs: None,
        system=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        warn=lambda *args, **kwargs: None,
        reset_timeline=lambda: resets.append(True),
    )

    agent = object.__new__(LLMAgent)
    agent.context = context
    agent.events = events
    agent.console = console
    agent.round_id = 1
    agent._session_id = "test"
    agent._session_revision_id = "before-round"
    agent._compaction_floor_tokens = 123
    agent._pending_images = []
    agent._last_interrupted = False
    agent.verbose = False
    agent.tool_executor = NS()
    agent._operation = lambda _label: nullcontext()
    agent.save_session = lambda *args, **kwargs: saved.append(
        (list(context.messages), agent.round_id, agent._session_revision_id, kwargs)
    )

    def cancel_after_context_management():
        context.replace_messages([{"role": "user", "content": "compacted u2"}])
        context.compaction_state = CompactionState(original_request="u2")
        agent._session_revision_id = "cancelled-round-snapshot"
        agent._compaction_floor_tokens = 999
        raise RequestInterrupted()

    agent.call_api = cancel_after_context_management

    agent._process_query(
        UserSubmission("u2 @config.py", "terminal", "submission-2")
    )

    assert context.messages == messages_before
    assert context.compaction_state == state_before
    assert agent.round_id == 1
    assert agent._session_revision_id == "before-round"
    assert agent._compaction_floor_tokens == 123
    assert agent._rewind_draft == "u2 @config.py"
    assert resets == [True]
    assert saved == [(messages_before, 1, "before-round", {"reason": "round"})]
    draft_events = [event for event in events.after(0) if event.type == "draft_set"]
    assert [event.data["text"] for event in draft_events] == ["u2 @config.py"]


def test_cancelled_exec_preserves_its_partial_result_in_round_context(tmp_path):
    context = ContextManager("system")
    context.add({"role": "user", "content": "u1"})
    context.add({"role": "assistant", "content": "a1"})
    messages_before = list(context.messages)
    tool_call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "exec_command", "arguments": '{"command": "slow"}'},
    }

    events = EventHub()
    resets = []
    systems = []
    saved = []
    console = NS(
        rule=lambda: None,
        user_input=lambda *args, **kwargs: None,
        response=lambda *args, **kwargs: None,
        system=lambda text, **kwargs: systems.append(text),
        error=lambda *args, **kwargs: None,
        warn=lambda *args, **kwargs: None,
        tool_result=lambda *args, **kwargs: None,
        thinking=lambda *args, **kwargs: nullcontext(),
        reset_timeline=lambda: resets.append(True),
    )

    agent = object.__new__(LLMAgent)
    agent.context = context
    agent.events = events
    agent.console = console
    agent.round_id = 1
    agent._session_id = "test"
    agent._session_revision_id = "before-round"
    agent._compaction_floor_tokens = None
    agent._pending_images = []
    agent._last_interrupted = False
    agent.verbose = False
    agent.stream = False
    agent.input_broker = None
    agent.cancellation = None
    agent.context_length = 16_000
    agent.token_estimator = NS(chars_per_token=3.3)
    agent.work_dir = str(tmp_path)
    agent.tool_compaction_totals = {
        "calls": 0,
        "original_chars": 0,
        "retained_chars": 0,
    }
    agent.tool_executor = NS(
        execute=lambda *args: {
            "stdout": "partial output\n",
            "exit_code": 143,
            "success": False,
            "streamed": True,
            "interrupted": True,
            "error": "Command was interrupted by user.",
        },
    )
    agent._operation = lambda _label: nullcontext()
    agent.save_session = lambda *args, **kwargs: saved.append(list(context.messages))

    def call_api():
        message = {"role": "assistant", "content": None, "tool_calls": [tool_call]}
        context.add(message)
        return message

    agent.call_api = call_api
    agent._process_query(UserSubmission("run it", "terminal", "submission-2"))

    expected = messages_before + [
        {"role": "user", "content": "run it"},
        {"role": "assistant", "content": None, "tool_calls": [tool_call]},
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": (
                "partial output\n"
                "[exit_code: 143, interrupted: true, timed_out: false]"
            ),
        },
    ]
    assert context.messages == expected
    assert agent.round_id == 2
    assert not hasattr(agent, "_rewind_draft")
    assert resets == []
    assert saved == [expected]
    assert systems[-1:] == ["Turn interrupted."]
    assert not [event for event in events.after(0) if event.type == "draft_set"]


def _continue_agent(messages):
    warnings = []
    calls = []
    saved = []
    agent = type("Agent", (AgentCommandsMixin,), {})()
    agent.context = ContextManager("system")
    agent.context.replace_messages(messages)
    agent.console = NS(
        warn=lambda message: warnings.append(message),
        rule=lambda: calls.append("rule"),
    )
    agent._operation = lambda label: nullcontext()
    agent.get_response = lambda: calls.append("response")
    agent._session_id = "test"
    agent.save_session = lambda *args, **kwargs: saved.append((args, kwargs))
    return agent, warnings, calls, saved


def test_continue_resumes_after_tool_result_without_adding_user_message():
    messages = [
        {"role": "user", "content": "do it"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "load_skill", "arguments": '{"name":"lean"}'},
            }],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "loaded"},
    ]
    agent, warnings, calls, saved = _continue_agent(messages)

    agent._cmd_continue()

    assert agent.context.messages == messages
    assert warnings == []
    assert calls == ["rule", "response"]
    assert saved == [(('test',), {"reason": "round"})]


def test_continue_warns_when_round_is_complete():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "done"},
    ]
    agent, warnings, calls, saved = _continue_agent(messages)

    agent._cmd_continue()

    assert warnings == ["The last round is already complete; nothing to continue."]
    assert calls == []
    assert saved == []


def test_continue_resumes_after_empty_assistant_message():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": ""},
    ]
    agent, warnings, calls, saved = _continue_agent(messages)

    agent._cmd_continue()

    assert warnings == []
    assert calls == ["rule", "response"]
    assert saved == [(('test',), {"reason": "round"})]


def test_continue_rejects_partial_tool_results():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call-1", "function": {"name": "one", "arguments": "{}"}},
                {"id": "call-2", "function": {"name": "two", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "done"},
    ]
    agent, warnings, calls, saved = _continue_agent(messages)

    agent._cmd_continue()

    assert warnings == [
        "The last assistant message has unresolved tool calls; cannot continue safely."
    ]
    assert calls == []
    assert saved == []


def test_provider_retry_classification_overrides_http_status():
    assert _is_fatal_api_error(
        ProviderError("subscription limit", status_code=429, retryable=False)
    ) is True
    assert _is_fatal_api_error(
        ProviderError("temporary", status_code=400, retryable=True)
    ) is False

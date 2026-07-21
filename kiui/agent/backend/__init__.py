import asyncio
import json
import os
import queue
import re
import threading
import time
from contextlib import nullcontext
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from prompt_toolkit.patch_stdout import patch_stdout

from kiui.agent.backend.commands import AgentCommandsMixin
from kiui.agent.backend.goals import GoalMixin
from kiui.agent.personas import (
    DEFAULT_PERSONA,
    PersonaContext,
    PersonaInfo,
    get_persona,
    list_personas,
)
from kiui.agent.backend.sessions import SessionMixin
from kiui.agent.backend.skill_commands import SkillCommandsMixin
from kiui.agent.terminal import TerminalInput
from kiui.agent.ui import AgentConsole, ContextStatus
from kiui.agent.utils import get_kia_dir
from kiui.agent.skills import discover_skills
from kiui.agent.tools import (
    MAX_GREP_MATCHES,
    ToolExecutor,
    format_tool_result,
    format_tool_summary,
    get_tool_definitions,
)
from kiui.agent.subagent import SubagentManager
from kiui.agent.tools.results import (
    discard_tool_result_artifact,
    persist_tool_result_artifact,
    read_tool_result_text,
)
from kiui.agent.permissions import PermissionController, PermissionMode
from kiui.agent.utils.streaming import consume_stream, message_to_dict
from kiui.agent.context import (
    ContextManager,
    TokenEstimator,
    ToolResultEnvelope,
    compact_context,
    compact_tool_result_envelope,
    build_tool_name_index,
    estimate_context_chars,
    get_role,
    get_text,
    get_tool_call_id,
    get_tool_calls,
    msg_chars,
    needs_compaction,
    prune_context,
    tool_result_char_budget,
)
from kiui.agent.utils.rewind import ChangeTracker
from kiui.agent.models import (
    REASONING_EFFORTS,
    ReasoningEffort,
    reasoning_kwargs,
    resolve_model_profile,
)
from kiui.agent.utils.interrupt import run_interruptible, RequestInterrupted
from kiui.agent.utils.io import (
    CancellationToken,
    EventHub,
    InputBroker,
    PromptBroker,
    UserSubmission,
    sanitize_unicode,
)

MAX_OUTPUT_TOKENS = 20_000


# HTTP status codes that represent transient failures worth retrying even
# though they are 4xx. Everything else in the 4xx range is a permanent client
# error (bad key, bad request, unknown model, …) and must not be retried.
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429})


def _is_fatal_api_error(exc: Exception) -> bool:
    """Whether *exc* is a permanent client error that retrying cannot fix.

    Only HTTP 4xx responses other than a few transient ones (rate limit,
    timeout, conflict) are fatal. Connection errors, timeouts, and 5xx server
    errors have no ``status_code`` or a retryable one, so they keep retrying.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        return False
    return 400 <= status < 500 and status not in _RETRYABLE_STATUS_CODES


_AT_PATH_RE = re.compile(r"(?<!\S)@([\w./\\~+-]+)")


def _strip_at_marks(query: str) -> str:
    """Strip the ``@`` prefix from file-path references in *query*.

    Only matches ``@`` at a word boundary (preceded by whitespace or
    start-of-string) so email addresses like ``user@host.com`` are left
    untouched.
    """
    return _AT_PATH_RE.sub(r"\1", query)


class LLMAgent(AgentCommandsMixin, GoalMixin, SkillCommandsMixin, SessionMixin):
    INITIAL_BACKOFF = 1.0   # seconds
    MAX_BACKOFF = 64.0      # seconds

    def __init__(
        self, 
        model: str,
        api_key: str,
        base_url: str,
        model_alias: str = "",
        verbose: bool = True,
        stream: bool = True,
        reasoning_effort: ReasoningEffort = "high",
        context_length: int | None = None,
        permission_mode: PermissionMode = PermissionMode.AUTO,
        persona: str | None = None,
        exec_mode: bool = False,
        is_subagent: bool = False,
        work_dir: str | None = None,
        console: AgentConsole | None = None,
        events: EventHub | None = None,
        input_broker: InputBroker | None = None,
        prompt_broker: PromptBroker | None = None,
        cancellation: CancellationToken | None = None,
    ):

        self.model = model
        self.model_alias = model_alias
        self.profile = resolve_model_profile(model, model_alias)
        self._api_key = api_key
        self._base_url = base_url
        self._client = None
        self.verbose = verbose
        self.stream = stream
        self.show_thinking = self.profile.reasoning is not None
        self.reasoning_effort = reasoning_effort
        self.context_length = context_length if context_length is not None else self.profile.context_length
        self.token_estimator = TokenEstimator()

        self.events = events
        if input_broker is None and not is_subagent:
            events = events or EventHub()
            self.events = events
            input_broker = InputBroker(events)
        self.input_broker = input_broker
        if prompt_broker is None and not is_subagent:
            prompt_broker = PromptBroker(events)
        self.prompt_broker = prompt_broker
        if cancellation is None and not is_subagent:
            cancellation = CancellationToken(events, prompt_broker)
        self.cancellation = cancellation
        if self.cancellation is not None and self.prompt_broker is not None:
            self.cancellation.prompts = self.prompt_broker
            self.prompt_broker.cancellation = self.cancellation
        self.console = console or AgentConsole(events=events)

        if self.prompt_broker is not None:
            self.console.prompt_broker = self.prompt_broker

            async def terminal_ask_async(prompt):
                terminal_message = prompt.message.splitlines()[0]
                if prompt.kind == "select":
                    return await self.console.select_terminal_async(
                        terminal_message,
                        choices=prompt.choices,
                        default=prompt.default or None,
                    )
                return await self.console.ask_text_terminal_async(
                    terminal_message, default=prompt.default
                )

            self.prompt_broker.set_terminal_adapter(terminal_ask_async)

        # Discover bundled skills directly from the installed package, followed
        # by project and personal skills.
        skill_issues: dict = {}
        self.skills = discover_skills(work_dir, issues=skill_issues)
        if not is_subagent:
            self._report_skill_issues(skill_issues)

        # persona owns the system prompt and the tool surface; the bundled
        # "coder" persona is the default.
        # A sub-agent must not spawn further sub-agents: spawning stays a
        # single, sequential level deep and always returns.
        self.is_subagent = is_subagent
        self.exec_mode = exec_mode
        self.work_dir = str(Path(work_dir).absolute()) if work_dir else str(Path.cwd())
        self.persona: PersonaInfo = get_persona(persona or DEFAULT_PERSONA)
        self.system_prompt = self._build_system_prompt()
        self.tools = self._get_tool_definitions()

        self.permissions = PermissionController(
            mode=permission_mode,
            console=self.console,
            work_dir=self.work_dir,
        )

        # subagent manager (only for top-level agents with a model_alias;
        # sub-agents get None so they cannot recursively spawn children)
        self.subagent_manager = (
            SubagentManager(
                model_alias=model_alias,
                reasoning_effort=reasoning_effort,
                console=self.console,
                parent_cancellation=cancellation,
            )
            if model_alias and not is_subagent
            else None
        )
        self.changes: ChangeTracker | None = None
        self.tool_executor = ToolExecutor(
            console=self.console,
            subagent_manager=self.subagent_manager,
            work_dir=self.work_dir,
            skills=self.skills,
            cancellation=cancellation,
        )

        self.context = ContextManager(self.system_prompt)
        self._pending_images: list[dict[str, str]] = []

        self.round_id = 0
        self._session_id: str | None = None  # set by chat_loop
        self._last_save_time: float = 0.0  # throttle auto-saves

        # ----- /goal auto-iteration state -----
        self.goal: str | None = None       # standing goal text (persists across rounds)
        self.goal_active: bool = False     # whether the auto-iterate loop is armed
        self.goal_iterations: int = 0      # number of goal-check rounds run so far
        self._pending_auto: str | None = None  # queued auto-injected prompt (goal check)
        self._last_interrupted: bool = False   # set by get_response when a round is cancelled

        self.token_totals = {
            "total": 0,
            "prompt": 0,
            "cached_prompt": 0,
            "completion": 0,
            "reasoning": 0,
        }
        self.tool_compaction_totals = {
            "calls": 0,
            "original_chars": 0,
            "retained_chars": 0,
        }

        # self.console.system(f"System prompt: {self.system_prompt[:100]}...")


    def _get_tool_definitions(self) -> list[dict[str, Any]]:
        return get_tool_definitions(
            include_subagent=not self.is_subagent,
            allowed=self.persona.tools,
            supports_image_input=self.profile.supports_image_input,
        )

    def _messages_with_pending_images(self) -> list[dict[str, Any]]:
        messages = self.context.get()
        if not self._pending_images:
            return messages

        content: list[dict[str, Any]] = []
        for image in self._pending_images:
            content.extend((
                {"type": "text", "text": f"Image returned by read_image: {image['file']}"},
                {"type": "image_url", "image_url": {"url": image["url"]}},
            ))
        messages.append({"role": "user", "content": content})
        return messages

    def _build_system_prompt(self) -> str:
        """Build the system prompt via the active persona."""
        ctx = PersonaContext(
            exec_mode=self.exec_mode,
            is_subagent=self.is_subagent,
            work_dir=self.work_dir,
            skills=self.skills,
        )
        return self.persona.build(ctx)

    def _accumulate_usage(self, usage):
        """Add token counts from an API response to session totals."""
        self.token_totals["total"] += usage.total_tokens
        self.token_totals["prompt"] += usage.prompt_tokens
        self.token_totals["completion"] += usage.completion_tokens
        if usage.prompt_tokens_details and usage.prompt_tokens_details.cached_tokens:
            self.token_totals["cached_prompt"] += usage.prompt_tokens_details.cached_tokens
        if usage.completion_tokens_details and usage.completion_tokens_details.reasoning_tokens:
            self.token_totals["reasoning"] += usage.completion_tokens_details.reasoning_tokens

    def _status_suffix(self) -> ContextStatus:
        """Context-window progress shown in the 'Working...' status bar."""
        ctx_chars = self.context.estimated_chars
        return ContextStatus(
            tokens=self.token_estimator.chars_to_tokens(ctx_chars),
            limit=self.context_length,
            input_tokens=self.token_totals["prompt"],
            output_tokens=self.token_totals["completion"],
        )

    def _interruptible_sleep(self, seconds: float):
        stop = threading.Event()
        run_interruptible(
            lambda: stop.wait(seconds), self.cancellation, on_cancel=stop.set
        )

    def _operation(self, label: str):
        if self.cancellation is None:
            return nullcontext()
        return self.cancellation.operation(label)

    def _create_change_tracker(self, session_id: str, work_dir: str):
        return ChangeTracker(session_id, work_dir, self.console)

    def _kia_dir(self) -> Path:
        return get_kia_dir()

    def _session_timestamp(self) -> str:
        return time.strftime("%Y%m%d_%H%M%S")

    def call_api(self):
        """Call the API using current context, with automatic retry on transient errors.

        Transient errors (connection failures, timeouts, 5xx, rate limits) are
        retried indefinitely with capped exponential backoff. Permanent client
        errors (auth, malformed request, unknown model, …) are re-raised as
        ``RuntimeError`` so ``get_response`` can end the turn gracefully.
        """

        # context management: prune old tool results, then compact if needed
        if self.context_length > 0:
            t_prune = time.monotonic()
            cpt = self.token_estimator.chars_per_token
            # Reuse the incrementally maintained char total to skip a full scan.
            self.context.replace_messages(
                prune_context(
                    self.context.messages, self.context_length, cpt,
                    total_chars=self.context.estimated_chars,
                )
            )
            prune_elapsed = time.monotonic() - t_prune
            if self.verbose and prune_elapsed > 0.1:
                self.console.debug(f"Context pruning took {prune_elapsed:.2f}s")

            if needs_compaction(
                self.context.messages, self.context_length, cpt,
                total_chars=self.context.estimated_chars,
            ):
                before_chars = self.context.estimated_chars
                before_msgs = len(self.context.messages)
                before_tokens = self.token_estimator.chars_to_tokens(before_chars)
                self.console.system("Context window pressure — compacting via LLM summarization")
                t_compact = time.monotonic()
                compact_client = self._request_client()
                try:
                    with self.console.thinking(
                        label="Compacting",
                        progress=True,
                        status_suffix=f"{before_msgs} messages, ~{before_tokens:,} tokens",
                    ):
                        compacted = run_interruptible(
                            lambda: compact_context(
                                self.context.messages, compact_client, self.model,
                                console=self.console,
                                context_length=self.context_length,
                                chars_per_token=cpt,
                            ),
                            self.cancellation,
                            on_cancel=compact_client.close,
                        )
                finally:
                    compact_client.close()
                if self.cancellation is not None and self.cancellation.cancelled:
                    raise RequestInterrupted()
                self.context.replace_messages(compacted)
                compact_elapsed = time.monotonic() - t_compact
                after_chars = self.context.estimated_chars
                after_msgs = len(self.context.messages)
                after_tokens = self.token_estimator.chars_to_tokens(after_chars)
                saved_pct = (1 - after_chars / before_chars) * 100 if before_chars else 0
                self.console.system(
                    f"Compaction complete ({compact_elapsed:.1f}s): "
                    f"{before_msgs} messages → {after_msgs} messages, "
                    f"~{before_tokens:,} tokens → ~{after_tokens:,} tokens "
                    f"(saved {saved_pct:.0f}%)"
                )

        if self.verbose:
            ctx_chars = self.context.estimated_chars
            ctx_tokens = self.token_estimator.chars_to_tokens(ctx_chars)
            ctx_pct = ctx_tokens / self.context_length * 100 if self.context_length else 0
            self.console.debug(
                f"Calling API (round: {self.round_id}, "
                f"context: ~{ctx_tokens}tok / {self.context_length}tok [{ctx_pct:.0f}%])"
            )

        had_pending_images = bool(self._pending_images)
        messages = sanitize_unicode(self._messages_with_pending_images())

        kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": self.stream,
            "max_tokens": MAX_OUTPUT_TOKENS,
        }
        if self.stream:
            # Ask the server to emit a final usage-only chunk so streamed turns
            # still get accurate token accounting.
            kwargs["stream_options"] = {"include_usage": True}

        if self.tools:
            kwargs["tools"] = self.tools
        
        kwargs.update(reasoning_kwargs(self.profile.reasoning, self.reasoning_effort))

        # ---- retry loop with exponential backoff ----
        t_api = time.monotonic()
        retry_count = 0
        wait_time = self.INITIAL_BACKOFF
        while True:
            try:
                if self.stream:
                    message, usage, finish_reason = self._stream_completion(kwargs)
                else:
                    message, usage, finish_reason = self._blocking_completion(kwargs)
                break
            except RequestInterrupted:
                raise  # user cancelled — never retry, let get_response roll back
            except Exception as e:
                if _is_fatal_api_error(e):
                    # Permanent client error — retrying cannot help. Surface it
                    # as RuntimeError so get_response ends the turn gracefully.
                    status = getattr(e, "status_code", "?")
                    raise RuntimeError(f"API request rejected (HTTP {status}): {e}") from e
                retry_count += 1
                self.console.system(
                    f"[Retry {retry_count}] {e} — retrying in {wait_time:.1f}s…"
                )
                self._interruptible_sleep(wait_time)
                wait_time = min(wait_time * 2, self.MAX_BACKOFF)
        api_elapsed = time.monotonic() - t_api

        if self.cancellation is not None and self.cancellation.cancelled:
            raise RequestInterrupted()

        self._pending_images.clear()
        self._accumulate_usage(usage)
        if not had_pending_images:
            self.token_estimator.calibrate(
                estimate_context_chars(messages), usage.prompt_tokens
            )

        if self.verbose:
            cached = usage.prompt_tokens_details.cached_tokens if usage.prompt_tokens_details else None
            reasoning = usage.completion_tokens_details.reasoning_tokens if usage.completion_tokens_details else None
            self.console.debug(
                f"API response in {api_elapsed:.1f}s — finish_reason: {finish_reason or 'N/A'}, "
                f"total_tokens: {usage.total_tokens} = "
                f"output: {usage.completion_tokens} (reasoning: {reasoning or 'N/A'}) "
                f"input: {usage.prompt_tokens} (cached: {cached or 'N/A'})"
            )
            tool_calls = get_tool_calls(message)
            if tool_calls:
                self.console.debug(f"Requested tool calls: {len(tool_calls)}")

        if finish_reason == "length":
            raise RuntimeError(
                "Response truncated by the output-token limit "
                f"(finish_reason='length', output={usage.completion_tokens:,}, "
                f"requested max_tokens={MAX_OUTPUT_TOKENS:,})."
            )

        self.context.add(message)
        return message



    @property
    def client(self):
        if self._client is None:
            self._client = self._request_client()
        return self._client

    @client.setter
    def client(self, client) -> None:
        self._client = client

    def _request_client(self):
        from openai import OpenAI

        return OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            max_retries=0,
        )

    def _blocking_completion(self, kwargs: dict):
        """Non-streaming call: show a spinner, return ``(message, usage)``."""
        client = self._request_client()
        try:
            with self.console.thinking(status_suffix=self._status_suffix()):
                response = run_interruptible(
                    lambda: client.chat.completions.create(**kwargs),
                    self.cancellation,
                    on_cancel=client.close,
                )
        finally:
            client.close()
        choice = response.choices[0]
        return message_to_dict(choice.message), response.usage, choice.finish_reason

    def _stream_completion(self, kwargs: dict):
        """Streaming call: buffer terminal output, return ``(message, usage)``.

        The blocking network request and stream consumption remain
        interruptible. Fragments are sent to web clients immediately, while
        terminal Markdown is rendered once after the response is complete.
        """
        client = self._request_client()

        with self.console.stream_response(show_thinking=self.show_thinking) as sink:
            with self.console.thinking(status_suffix=self._status_suffix()):
                def request():
                    stream = client.chat.completions.create(**kwargs)
                    try:
                        return consume_stream(
                            stream,
                            on_content=sink.on_content,
                            on_thinking=sink.on_thinking,
                            should_stop=lambda: (
                                self.cancellation is not None
                                and self.cancellation.cancelled
                            ),
                        )
                    finally:
                        stream.close()

                try:
                    message, usage, finish_reason = run_interruptible(
                        request, self.cancellation, on_cancel=client.close
                    )
                finally:
                    client.close()
        if usage is None:
            # Some proxies omit the usage chunk; fall back to an estimate so
            # accounting/calibration still works.
            usage = self._estimate_usage(message)
        return message, usage, finish_reason

    def _estimate_usage(self, message) -> Any:
        """Build a rough usage object when the stream omits the usage chunk."""
        from openai.types import CompletionUsage

        prompt_chars = estimate_context_chars(self.context.get())
        prompt_tokens = self.token_estimator.chars_to_tokens(prompt_chars)
        completion_chars = len(message.get("content") or "")
        for tc in message.get("tool_calls") or []:
            completion_chars += len(tc["function"].get("arguments") or "")
        completion_tokens = self.token_estimator.chars_to_tokens(completion_chars)
        return CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )



    def execute_tool_calls(self, tool_calls: list) -> bool:
        """Execute tool calls via the built-in ToolExecutor.

        Returns True if the round was interrupted by the user (ESC / Ctrl+C /
        web cancel) partway through, in which case any remaining tool calls are
        answered with a synthetic "skipped" result so the assistant/tool message
        pairing stays valid, and the caller should return to the prompt.
        """

        t_all = time.monotonic()
        interrupted = False
        for i, tool_call in enumerate(tool_calls):
            function_name = tool_call["function"]["name"]

            # A user cancel aborts the whole round; fill remaining calls with
            # skipped results to keep assistant/tool pairing valid.
            if self.cancellation is not None and self.cancellation.cancelled:
                interrupted = True
            if interrupted:
                self.context.add({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": "Tool call skipped: the user interrupted the turn.",
                })
                continue

            try:
                function_args = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError as e:
                function_args = {}
                self.console.error(f"Failed to parse tool args: {e}")
            
            if self.verbose:
                self.console.debug(f"Tool call {i+1}/{len(tool_calls)}: {function_name}({function_args})")

            if self.cancellation is not None and self.cancellation.cancelled:
                interrupted = True
                self.context.add({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": "Tool call skipped: the user interrupted the turn.",
                })
                continue

            allowed, denial_reason = self.permissions.check(function_name, function_args)
            t_exec = time.monotonic()
            if self.cancellation is not None and self.cancellation.cancelled:
                result = {
                    "error": "Tool call skipped: the user interrupted the turn.",
                    "success": False,
                    "interrupted": True,
                }
            elif not allowed:
                msg = f"Tool call denied: {function_name}"
                if denial_reason:
                    msg += f"\nReason: {denial_reason}"
                result = {"error": msg, "success": False}
            elif function_name == "exec_command":
                with self.console.thinking(label="Running exec_command"):
                    result = self.tool_executor.execute(function_name, function_args)
            elif function_name == "inspect_processes" and function_args.get("wait", 0) > 0:
                with self.console.thinking(label="Waiting for inspect_processes"):
                    result = self.tool_executor.execute(function_name, function_args)
            else:
                result = self.tool_executor.execute(function_name, function_args)
            exec_elapsed = time.monotonic() - t_exec
            image_url = result.pop("image_url", None)
            if image_url and result.get("success"):
                self._pending_images.append({
                    "file": function_args["file"],
                    "url": image_url,
                })
            result_text = format_tool_result(result)

            success = result.get("success", False)
            if result.get("streamed"):
                exit_code = result.get("exit_code", "?")
                self.console.tool_result(f"exit code {exit_code} ({exec_elapsed:.1f}s)", success=success)
            elif function_name in ("edit_file", "write_file") and "diff" in result:
                self.console.diff_edit(**result["diff"], success=success)
            elif function_name == "multi_edit":
                if result.get("diffs"):
                    for d in result["diffs"]:
                        self.console.diff_edit(**d, success=success)
                else:
                    self.console.tool_result(format_tool_summary(result_text), success=success)
            elif function_name == "read_file":
                self._display_read_result(result, success)
            elif function_name in ("glob_files", "grep_files"):
                self._display_search_result(function_name, result, success)
            else:
                self.console.tool_result(format_tool_summary(result_text), success=success)

            envelope = ToolResultEnvelope(function_name, function_args, result, result_text)
            budget = tool_result_char_budget(
                self.context_length,
                self.token_estimator.chars_per_token,
                function_name,
            )
            if envelope.original_chars > budget:
                try:
                    compaction_text = read_tool_result_text(result, result_text)
                except OSError as e:
                    compaction_text = result_text
                    self.console.warn(f"Could not read captured tool output: {e}")

                try:
                    artifact_path = persist_tool_result_artifact(
                        function_name,
                        compaction_text,
                        result,
                        tool_call["id"],
                        self.work_dir,
                        self._session_id,
                        self.round_id,
                    )
                except (OSError, ValueError) as e:
                    artifact_path = None
                    self.console.warn(f"Could not save compacted tool output: {e}")

                compacted = compact_tool_result_envelope(
                    ToolResultEnvelope(function_name, function_args, result, compaction_text),
                    self.context_length,
                    self.token_estimator.chars_per_token,
                    artifact_path=artifact_path,
                )
                result_text = compacted.text
                self.tool_compaction_totals["calls"] += 1
                self.tool_compaction_totals["original_chars"] += compacted.original_chars
                self.tool_compaction_totals["retained_chars"] += compacted.retained_chars
                notice = f"; full captured output: {artifact_path}" if artifact_path else ""
                self.console.system(
                    f"Compacted {function_name} result with {compacted.reducer} "
                    f"({compacted.tier}){notice}"
                )
            else:
                cleanup_error = discard_tool_result_artifact(result)
                if cleanup_error:
                    self.console.warn(f"Could not remove temporary tool output: {cleanup_error}")

            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": result_text,
            }
            self.context.add(tool_message)

            # Detect a user interrupt: either the tool self-reported it
            # (e.g. exec_command killed by ESC) or the shared cancellation
            # token was tripped (web cancel). Abort the remaining calls.
            if result.get("interrupted") or (
                self.cancellation is not None and self.cancellation.cancelled
            ):
                interrupted = True

        total_elapsed = time.monotonic() - t_all
        if self.verbose and len(tool_calls) > 1:
            self.console.debug(f"All {len(tool_calls)} tool calls completed in {total_elapsed:.1f}s")

        return interrupted

    def _display_read_result(self, result: dict[str, Any], success: bool) -> None:
        """Compact read_file result: show path, line count, success."""
        if not success:
            self.console.tool_result(format_tool_summary(format_tool_result(result)), success=False)
            return
        lines_read = result.get("lines_read", "?")
        self.console.tool_result(f"{lines_read} lines read", success=True)

    def _display_search_result(self, tool_name: str, result: dict[str, Any], success: bool) -> None:
        """Compact glob_files/grep_files result: show match count / file count."""
        if not success:
            error_msg = result.get("error", "Unknown error")
            self.console.tool_result(error_msg, success=False)
            return
        count = result.get("count", 0)
        truncated = result.get("truncated", False)
        if tool_name == "glob_files":
            msg = f"{count} files matched"
            if truncated:
                msg += " (truncated to 500)"
        else:
            msg = f"{count} matches"
            if truncated:
                msg += f" (truncated to {MAX_GREP_MATCHES})"
        self.console.tool_result(msg, success=True)

    def get_response(self):
        """Process the context and update the current response.

        Fatal API errors (auth, quota, bad request, …) are caught, displayed
        to the user, and cause the turn to end gracefully rather than crash.
        """

        iteration = 0
        t_turn_start = time.monotonic()
        self._last_interrupted = False

        while True:
            iteration += 1
            t_iter = time.monotonic()
            if self.verbose and iteration > 1:
                self.console.debug(f"--- Agentic loop iteration {iteration} ---")

            snapshot = self.context.checkpoint()

            try:
                message = self.call_api()
            except RequestInterrupted:
                # Roll back to the state before this request was sent.
                self.context.rollback(snapshot)
                self._pending_images.clear()
                self.console.system("Request cancelled.")
                self._last_interrupted = True
                return None
            except RuntimeError as e:
                self.context.rollback(snapshot)
                self._pending_images.clear()
                self.console.error(f"API call failed: {e}")
                return None
            except Exception:
                raise

            content = message.get("content")
            if content:
                # The stream sink renders buffered Markdown and emits the final
                # event on close, so avoid printing it twice here.
                if not self.stream:
                    self.console.response(content)

            if not message.get("tool_calls"):
                if self.verbose:
                    turn_elapsed = time.monotonic() - t_turn_start
                    self.console.debug(f"Turn complete: {iteration} iteration(s) in {turn_elapsed:.1f}s")
                return content or None

            interrupted = self.execute_tool_calls(message["tool_calls"])

            if interrupted:
                # The user cancelled a tool mid-round. Stop the agentic loop
                # and return to the prompt instead of feeding the (partial)
                # tool results back to the model for another iteration.
                self._pending_images.clear()
                self.console.system("Turn interrupted.")
                self._last_interrupted = True
                return None

            if self.verbose:
                iter_elapsed = time.monotonic() - t_iter
                self.console.debug(f"Iteration {iteration} total: {iter_elapsed:.1f}s")

    # ----- bash command (!) ------------------------------------------------

    def _run_bash_command(self, command: str) -> None:
        """Execute a shell command starting with '!' directly, without sending to the model.

        Output is streamed in real-time and the command can be interrupted via Ctrl+C.
        """
        self.console.tool(f"! {command}")
        arguments = {"command": command}
        allowed, reason = self.permissions.check_safety("exec_command", arguments)
        if not allowed:
            self.console.tool_result(reason, success=False)
            return
        t_exec = time.monotonic()
        with self._operation("shell command"):
            with self.console.thinking(label="Running exec_command"):
                result = self.tool_executor.execute("exec_command", arguments)
        exec_elapsed = time.monotonic() - t_exec
        result_text = format_tool_result(result)
        success = result.get("success", False)
        if result.get("streamed", True):
            exit_code = result.get("exit_code", "?")
            self.console.tool_result(f"exit code {exit_code} ({exec_elapsed:.1f}s)", success=success)
        else:
            self.console.tool_result(format_tool_summary(result_text), success=success)
        cleanup_error = discard_tool_result_artifact(result)
        if cleanup_error:
            self.console.warn(f"Could not remove temporary tool output: {cleanup_error}")

    def _restart_session(self):
        """Save the current session and start a fresh one (used by /clear and /persona)."""
        if self._session_id and self.context.messages:
            try:
                self.save_session(self._session_id)
                self.console.system(f"Session '{self._session_id}' saved.")
            except Exception as e:
                self.console.warn(f"Could not save session before clear: {e}")

        # Start a brand-new session without reusing an ID from the same second.
        base_id = self._session_timestamp()
        session_id = base_id
        suffix = 2
        sessions_dir = self._sessions_dir()
        while session_id == self._session_id or (sessions_dir / f"{session_id}.json").exists():
            session_id = f"{base_id}_{suffix}"
            suffix += 1
        self._session_id = session_id
        self._last_save_time = 0.0  # allow immediate save of the new session
        self.context.replace_messages([])
        self._pending_images.clear()
        self.round_id = 0
        self.token_totals = {key: 0 for key in self.token_totals}
        self.tool_compaction_totals = {key: 0 for key in self.tool_compaction_totals}
        self.tool_executor.shutdown_processes(clear=True)
        self.tool_executor._loaded_skills.clear()
        self.tool_executor._skill_loads.clear()
        self.tool_executor.goal_report = None
        self.permissions.reset_session()
        # Drop any standing goal — the new session starts clean.
        self.goal = None
        self.goal_active = False
        self.goal_iterations = 0
        self._pending_auto = None
        self._last_interrupted = False
        # Create a fresh change tracker for the new session
        _wd = self.tool_executor._work_dir or os.getcwd()
        if self.changes is not None:
            self.changes.close()
        self.changes = self._create_change_tracker(self._session_id, _wd)
        self.tool_executor._change_tracker = self.changes
        self.tool_executor._get_round_id = lambda: self.round_id
        self.console.system(f"Started new session '{self._session_id}'.")

    # ----- main loops -------------------------------------------------------

    def _run_terminal_loop(self, terminal: TerminalInput) -> None:
        """Keep the editor active while agent rounds run on a worker thread."""
        assert self.input_broker is not None
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kia-agent")
        active: Future | None = None
        exit_requested = False
        should_exit = False
        prompt_task: asyncio.Task | None = None
        prompt_suspended = False
        prompt_resumed: asyncio.Event | None = None
        draft = ""

        def cancel() -> None:
            if self.cancellation is not None:
                self.cancellation.cancel()

        def pending_text() -> str | None:
            item = self.input_broker.submission
            return item.text if item is not None else None

        def edit_pending() -> str | None:
            item = self.input_broker.withdraw()
            if item is not None:
                terminal.app.invalidate()
            return item.text if item is not None else None

        terminal.set_runtime_state(
            cancel=cancel,
            pending_text=pending_text,
            edit_pending=edit_pending,
        )
        if self.cancellation is not None:
            self.cancellation.watch_keyboard = False
        self.console.interactive_input = True
        self.console.status_sink = terminal.set_status

        async def loop() -> None:
            nonlocal active, exit_requested, should_exit, prompt_task, prompt_suspended, prompt_resumed, draft
            ui_loop = asyncio.get_running_loop()
            input_ready = asyncio.Event()

            def wake_input() -> None:
                ui_loop.call_soon_threadsafe(input_ready.set)

            self.input_broker.add_listener(wake_input)
            prompt_resumed = asyncio.Event()
            prompt_resumed.set()
            prompt_task = asyncio.create_task(terminal.prompt_async())

            async def show_prompt(prompt):
                nonlocal prompt_task, prompt_suspended, draft
                prompt_suspended = True
                prompt_resumed.clear()
                draft = terminal.text
                prompt_task.cancel()
                try:
                    try:
                        await prompt_task
                    except asyncio.CancelledError:
                        pass
                    terminal_message = prompt.message.splitlines()[0]
                    if prompt.kind == "select":
                        return await self.console.select_terminal_async(
                            terminal_message,
                            choices=prompt.choices,
                            default=prompt.default or None,
                        )
                    return await self.console.ask_text_terminal_async(
                        terminal_message, default=prompt.default
                    )
                finally:
                    prompt_task = asyncio.create_task(terminal.prompt_async(draft))
                    draft = ""
                    prompt_suspended = False
                    prompt_resumed.set()

            async def terminal_ask(prompt):
                future = asyncio.run_coroutine_threadsafe(
                    show_prompt(prompt), ui_loop
                )
                return await asyncio.wrap_future(future)

            if self.prompt_broker is not None:
                self.prompt_broker.set_terminal_adapter(terminal_ask)

            try:
                while True:
                    input_task = asyncio.create_task(input_ready.wait())
                    wait = {prompt_task, input_task}
                    if active is not None:
                        wait.add(asyncio.wrap_future(active))
                    done, _ = await asyncio.wait(
                        wait, return_when=asyncio.FIRST_COMPLETED
                    )
                    if input_task in done:
                        input_ready.clear()
                    else:
                        input_task.cancel()

                    if active is not None and active.done():
                        should_exit = active.result()
                        active = None
                        terminal.set_busy(False)
                        if exit_requested or should_exit:
                            return
                        self._queue_pending_auto()
                        if self.input_broker.pending:
                            active = executor.submit(self._process_next_submission)
                            terminal.set_busy(True)

                    if active is None and self.input_broker.pending:
                        active = executor.submit(self._process_next_submission)
                        terminal.set_busy(True)

                    if prompt_task in done:
                        try:
                            text = prompt_task.result()
                        except asyncio.CancelledError:
                            if prompt_suspended:
                                await prompt_resumed.wait()
                                continue
                            raise
                        except (EOFError, KeyboardInterrupt):
                            if active is None:
                                return
                            exit_requested = True
                            cancel()
                            prompt_task = asyncio.create_task(terminal.prompt_async())
                            continue

                        text = text.strip()
                        if not text:
                            prompt_task = asyncio.create_task(terminal.prompt_async())
                            continue
                        try:
                            self.input_broker.submit(text, source="terminal")
                        except ValueError as exc:
                            prompt_task = asyncio.create_task(
                                terminal.prompt_async(text)
                            )
                            self.console.warn(str(exc))
                            continue
                        prompt_task = asyncio.create_task(terminal.prompt_async())
                        terminal.app.invalidate()
                        if active is None:
                            active = executor.submit(self._process_next_submission)
                            terminal.set_busy(True)
            finally:
                self.input_broker.remove_listener(wake_input)
                if prompt_task is not None:
                    prompt_task.cancel()
                if active is not None:
                    await asyncio.wrap_future(active)

        try:
            with patch_stdout(raw=True):
                asyncio.run(loop())
        finally:
            executor.shutdown(wait=True)
            if self.cancellation is not None:
                self.cancellation.watch_keyboard = True
            self.console.interactive_input = False
            self.console.status_sink = None

    def _queue_pending_auto(self) -> None:
        if self._pending_auto is None or self.input_broker is None:
            return
        query = self._pending_auto
        self._pending_auto = None
        try:
            self.input_broker.submit(query, source="goal")
        except ValueError:
            self._pending_auto = query

    def _process_next_submission(self) -> bool:
        assert self.input_broker is not None
        try:
            submission = self.input_broker.get_nowait()
        except queue.Empty:
            return False
        return self._process_query(submission)

    def _process_query(self, submission: UserSubmission) -> bool:
        """Process one user submission. Return True when chat should exit."""
        query = _strip_at_marks(submission.text.strip())
        if not query:
            return False
        self.console.rule()
        self.console.user_input(
            query,
            source=submission.source,
            submission_id=submission.id,
            with_rule=False,
        )
        if query.lower() in ("exit", "quit"):
            return True
        if query.startswith("!"):
            bash_cmd = query[1:].strip()
            if bash_cmd:
                self._run_bash_command(bash_cmd)
            else:
                self.console.warn("Usage: !<shell command>")
            return False
        if query.startswith("/"):
            cmd_word = query.split()[0][1:].lower()
            if cmd_word in self.COMMANDS:
                return self._handle_command(query)
            self.console.warn(
                f"Unknown command: /{cmd_word}. Type /help for available commands."
            )
            return False

        self.context.add({"role": "user", "content": query})
        self.round_id += 1
        self.console.rule()
        self.tool_executor.goal_report = None
        with self._operation("agent response"):
            self.get_response()
        self._maybe_continue_goal()
        try:
            now = time.monotonic()
            if now - self._last_save_time >= 30:
                self.save_session(self._session_id)
                self._last_save_time = now
        except Exception:
            pass
        return False

    def chat_loop(self, resumed_session_id: str | None = None):

        reasoning = self.profile.reasoning or "none"
        if reasoning != "none":
            reasoning += f" · {self.reasoning_effort} effort"
        self.console.startup_panel(
            model=self.model,
            context=f"{self.context_length:,} tokens",
            reasoning=reasoning,
            permission=self.permissions.mode.value,
            persona=self.persona.name,
            skills=self._skills_summary(),
            workspace=self.work_dir,
        )

        # Auto-save session id: reuse resumed session or generate a new one.
        self._session_id = resumed_session_id or self._session_timestamp()

        # Initialize change tracker for code+conversation rollback
        _wd = self.tool_executor._work_dir or os.getcwd()
        if self.changes is None:
            self.changes = self._create_change_tracker(self._session_id, _wd)
        # Wire the tracker into the tool executor for per-operation tracking
        self.tool_executor._change_tracker = self.changes
        self.tool_executor._get_round_id = lambda: self.round_id

        terminal = TerminalInput(history_path=str(self._kia_dir() / "history"), work_dir=_wd)

        try:
            self._run_terminal_loop(terminal)
        finally:
            session_saved = False
            try:
                self.save_session(self._session_id)
                session_saved = True
            except Exception as e:
                self.console.warn(f"Could not save session before exit: {e}")
            if self.changes is not None:
                self.changes.close()
            self.tool_executor.shutdown_processes()
            self._print_token_summary(resume=self._session_id if session_saved else None)

    def execute(self, query: str, *, manage_operation: bool = True):
        self.console.system(f"Executing query: {query}")
        t0 = time.time()

        if self._session_id is None:
            self._session_id = time.strftime("%Y%m%d_%H%M%S")

        # Strip @ prefix from file-path references
        query = _strip_at_marks(query)

        # bash command shortcut: !<command>
        if query.startswith("!"):
            bash_cmd = query[1:].strip()
            if bash_cmd:
                self._run_bash_command(bash_cmd)
            else:
                self.console.warn("Usage: !<shell command>")
            return None

        user_message = {"role": "user", "content": query}
        self.context.add(user_message)

        self.console.rule()

        operation = self._operation("agent response") if manage_operation else nullcontext()
        with operation:
            response = self.get_response()

        t1 = time.time()
        self.console.system(f"Execution time: {t1 - t0:.2f} seconds")
        self._print_token_summary()
        return response

    def _tool_compaction_summary(self) -> str:
        totals = self.tool_compaction_totals
        original = totals["original_chars"]
        retained = totals["retained_chars"]
        if not totals["calls"] or not original:
            return "none"
        saved = max(0, round((1 - retained / original) * 100))
        original_tokens = self.token_estimator.chars_to_tokens(original)
        retained_tokens = self.token_estimator.chars_to_tokens(retained)
        return (
            f"{totals['calls']} result(s), ~{original_tokens:,}→{retained_tokens:,} "
            f"tokens (-{saved}%)"
        )

    def _print_token_summary(self, resume: str | None = None):
        if resume is None:
            self.console.system(
                f"Total tokens used: {self.token_totals['total']} "
                f"(input: {self.token_totals['prompt']}, cached input: {self.token_totals['cached_prompt']}, "
                f"output: {self.token_totals['completion']}, reasoning: {self.token_totals['reasoning']})"
            )
            return
        self.console.session_end_panel(
            total=self.token_totals["total"],
            prompt=self.token_totals["prompt"],
            cached_prompt=self.token_totals["cached_prompt"],
            completion=self.token_totals["completion"],
            reasoning=self.token_totals["reasoning"],
            resume=resume,
        )

import asyncio
import json
import os
import queue
import re
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI

from kiui.agent.ui import AgentConsole
from kiui.agent.terminal import TerminalInput
from kiui.agent.utils import get_kia_dir
from kiui.agent.prompts import build_system_prompt
from kiui.agent.skills import discover_skills
from kiui.agent.tools import (
    MAX_GREP_MATCHES,
    ToolExecutor,
    format_tool_result,
    format_tool_summary,
    get_tool_definitions,
)
from kiui.agent.subagent import SubagentManager
from kiui.agent.permissions import PermissionController, PermissionMode
from kiui.agent.context import (
    truncate_tool_result,
    prune_context,
    needs_compaction,
    compact_context,
    estimate_context_chars,
    TokenEstimator,
    get_role,
    get_text,
    get_tool_calls,
    get_tool_call_id,
    msg_chars,
)
from kiui.agent.rewind import ChangeTracker
from kiui.agent.models import resolve_model_profile
from kiui.agent.interrupt import run_interruptible, RequestInterrupted
from kiui.agent.io import (
    CancellationToken,
    EventHub,
    InputBroker,
    PromptBroker,
    UserSubmission,
)

# How often the input race checks the web queue while the terminal
# prompt is pending.
INPUT_POLL_INTERVAL = 0.05


class ContextManager:
    """Flat conversation history with context-management hooks."""

    def __init__(self, system_prompt: str):
        self.system_prompt: dict[str, str] = {
            "role": "system",
            "content": system_prompt,
        }
        self.messages: list[Any] = []

    def add(self, message: dict[str, Any] | Any) -> None:
        self.messages.append(message)

    def get(self, include_system: bool = True) -> list[Any]:
        res: list[Any] = []
        if include_system:
            res.append(self.system_prompt)
        res.extend(self.messages)
        return res

    def replace_messages(self, new_messages: list[Any]) -> None:
        """Replace all messages (used after compaction)."""
        if new_messages is self.messages:
            return
        self.messages = list(new_messages)

    def checkpoint(self) -> list[Any]:
        """Return a shallow copy of messages for rollback on interruption."""
        return list(self.messages)

    def rollback(self, snapshot: list[Any]) -> None:
        """Restore messages to a previous checkpoint."""
        self.messages = snapshot


_AT_PATH_RE = re.compile(r"(?<!\S)@([\w./\\~+-]+)")


def _strip_at_marks(query: str) -> str:
    """Strip the ``@`` prefix from file-path references in *query*.

    Only matches ``@`` at a word boundary (preceded by whitespace or
    start-of-string) so email addresses like ``user@host.com`` are left
    untouched.
    """
    return _AT_PATH_RE.sub(r"\1", query)


class LLMAgent:
    MAX_RETRIES = 10
    INITIAL_BACKOFF = 1.0   # seconds

    def __init__(
        self, 
        model: str,
        api_key: str,
        base_url: str,
        model_alias: str = "",
        verbose: bool = True,
        thinking_budget: Literal["low", "medium", "high"] = "low",
        context_length: int | None = None,
        permission_mode: PermissionMode = PermissionMode.DEFAULT,
        exec_mode: bool = False,
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
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
        )
        self.verbose = verbose
        self.thinking_budget = thinking_budget
        self.context_length = context_length if context_length is not None else self.profile.context_length
        self.token_estimator = TokenEstimator()

        self.events = events
        self.input_broker = input_broker
        self.prompt_broker = prompt_broker
        self.cancellation = cancellation
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

        # discover skills from .kia/skills/
        self.skills = discover_skills(work_dir)

        # built-in system prompt and tools
        self.system_prompt = build_system_prompt(exec_mode=exec_mode, work_dir=work_dir, skills=self.skills)
        self.tools = get_tool_definitions()

        self.permissions = PermissionController(
            mode=permission_mode,
            console=self.console,
            work_dir=work_dir,
        )

        # subagent manager (only if we have a model_alias for spawning children)
        self.subagent_manager = SubagentManager(model_alias=model_alias, console=self.console) if model_alias else None
        self.changes: ChangeTracker | None = None
        self.tool_executor = ToolExecutor(
            console=self.console,
            subagent_manager=self.subagent_manager,
            work_dir=work_dir,
            skills=self.skills,
            cancellation=cancellation,
        )

        self.context = ContextManager(self.system_prompt)

        self.round_id = 0
        self._session_id: str | None = None  # set by chat_loop
        self._last_save_time: float = 0.0  # throttle auto-saves

        self.token_totals = {
            "total": 0,
            "prompt": 0,
            "cached_prompt": 0,
            "completion": 0,
            "reasoning": 0,
        }

        self.console.system(f"Created Agent with model: {model} (context: {self.context_length:,} tokens), permission: {permission_mode.value}")
        # self.console.system(f"System prompt: {self.system_prompt[:100]}...")


    def _accumulate_usage(self, usage):
        """Add token counts from an API response to session totals."""
        self.token_totals["total"] += usage.total_tokens
        self.token_totals["prompt"] += usage.prompt_tokens
        self.token_totals["completion"] += usage.completion_tokens
        if usage.prompt_tokens_details and usage.prompt_tokens_details.cached_tokens:
            self.token_totals["cached_prompt"] += usage.prompt_tokens_details.cached_tokens
        if usage.completion_tokens_details and usage.completion_tokens_details.reasoning_tokens:
            self.token_totals["reasoning"] += usage.completion_tokens_details.reasoning_tokens

    def _interruptible_sleep(self, seconds: float):
        # Watch the keyboard during backoff so ESC/Ctrl+C cancels the wait
        # too (raises RequestInterrupted, which call_api lets propagate).
        run_interruptible(lambda: time.sleep(seconds), self.cancellation)

    def _operation(self, label: str):
        if self.cancellation is None:
            return nullcontext()
        return self.cancellation.operation(label)

    def call_api(self):
        """Call the API using current context, with automatic retry on any error.

        All errors are retried with exponential backoff up to MAX_RETRIES times.
        """

        # context management: prune old tool results, then compact if needed
        if self.context_length > 0:
            t_prune = time.monotonic()
            cpt = self.token_estimator.chars_per_token
            self.context.replace_messages(
                prune_context(self.context.messages, self.context_length, cpt)
            )
            prune_elapsed = time.monotonic() - t_prune
            if self.verbose and prune_elapsed > 0.1:
                self.console.debug(f"Context pruning took {prune_elapsed:.2f}s")

            if needs_compaction(self.context.messages, self.context_length, cpt):
                before_chars = estimate_context_chars(self.context.messages)
                before_msgs = len(self.context.messages)
                before_tokens = self.token_estimator.chars_to_tokens(before_chars)
                self.console.system(
                    f"Context window pressure — compacting via LLM summarization "
                    f"({before_msgs} messages, ~{before_tokens:,} tokens)..."
                )
                t_compact = time.monotonic()
                self.context.replace_messages(
                    compact_context(
                        self.context.messages, self.client, self.model,
                        console=self.console,
                        context_length=self.context_length,
                        chars_per_token=cpt,
                    )
                )
                compact_elapsed = time.monotonic() - t_compact
                after_chars = estimate_context_chars(self.context.messages)
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
            ctx_chars = estimate_context_chars(self.context.messages)
            ctx_tokens = self.token_estimator.chars_to_tokens(ctx_chars)
            ctx_pct = ctx_tokens / self.context_length * 100 if self.context_length else 0
            self.console.debug(
                f"Calling API (round: {self.round_id}, "
                f"context: ~{ctx_tokens}tok / {self.context_length}tok [{ctx_pct:.0f}%])"
            )

        messages = self.context.get()

        kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        if self.tools:
            kwargs["tools"] = self.tools
        
        # thinking budget (driven by model profile)
        if self.profile.thinking == "openai":
            kwargs["reasoning_effort"] = self.thinking_budget
        elif self.profile.thinking == "gemini":
            kwargs["extra_body"] = {
                "google": {
                    "thinking_config": {
                        "thinking_budget": "low" if self.thinking_budget == "low" else "high",
                        "include_thoughts": True,
                    },
                },
            }
        elif self.profile.thinking == "deepseek":
            kwargs["reasoning_effort"] = "max"
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

        # ---- retry loop with exponential backoff ----
        t_api = time.monotonic()
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                with self.console.thinking():
                    response = run_interruptible(
                        lambda: self.client.chat.completions.create(**kwargs),
                        self.cancellation,
                    )
                break
            except RequestInterrupted:
                raise  # user cancelled — never retry, let get_response roll back
            except Exception as e:
                if attempt < self.MAX_RETRIES:
                    wait_time = self.INITIAL_BACKOFF * (2 ** attempt)
                    self.console.system(f"[Retry {attempt + 1}/{self.MAX_RETRIES}] {e} — retrying in {wait_time:.1f}s…")
                    self._interruptible_sleep(wait_time)
                else:
                    raise RuntimeError(f"Max retries ({self.MAX_RETRIES}) exceeded. Last error: {e}") from e
        api_elapsed = time.monotonic() - t_api

        message = response.choices[0].message
        usage = response.usage

        self._accumulate_usage(usage)
        self.token_estimator.calibrate(
            estimate_context_chars(messages), usage.prompt_tokens
        )
        self.context.add(message)

        if self.verbose:
            cached = getattr(usage.prompt_tokens_details, "cached_tokens", None) if usage.prompt_tokens_details else None
            reasoning = getattr(usage.completion_tokens_details, "reasoning_tokens", None) if usage.completion_tokens_details else None
            self.console.debug(
                f"API response in {api_elapsed:.1f}s — total_tokens: {usage.total_tokens} = "
                f"output: {usage.completion_tokens} (reasoning: {reasoning or 'N/A'}) "
                f"input: {usage.prompt_tokens} (cached: {cached or 'N/A'})"
            )
            if message.tool_calls:
                self.console.debug(f"Requested tool calls: {len(message.tool_calls)}")

        return message


    def execute_tool_calls(self, tool_calls: list):
        """Execute tool calls via the built-in ToolExecutor."""

        t_all = time.monotonic()
        for i, tool_call in enumerate(tool_calls):
            function_name = tool_call.function.name
            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                function_args = {}
                self.console.error(f"Failed to parse tool args: {e}")
            
            if self.verbose:
                self.console.debug(f"Tool call {i+1}/{len(tool_calls)}: {function_name}({function_args})")

            allowed, denial_reason = self.permissions.check(function_name, function_args)
            t_exec = time.monotonic()
            if not allowed:
                msg = f"Tool call denied: {function_name}"
                if denial_reason:
                    msg += f"\nReason: {denial_reason}"
                result = {"error": msg, "success": False}
            else:
                result = self.tool_executor.execute(function_name, function_args)
            exec_elapsed = time.monotonic() - t_exec
            result_text = format_tool_result(result)

            success = result.get("success", False)
            if result.get("streamed"):
                exit_code = result.get("exit_code", "?")
                self.console.tool_result(f"exit code {exit_code} ({exec_elapsed:.1f}s)", success=success)
            elif function_name in ("edit_file", "write_file") and "diff" in result:
                self.console.diff_edit(**result["diff"], success=success)
            elif function_name == "read_file":
                self._display_read_result(result, success)
            elif function_name in ("glob_files", "grep_files"):
                self._display_search_result(function_name, result, success)
            else:
                self.console.tool_result(format_tool_summary(result_text), success=success)

            # Layer 1: generic truncation relative to context window
            if self.context_length > 0:
                result_text = truncate_tool_result(result_text, self.context_length, self.token_estimator.chars_per_token)

            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_text,
            }
            self.context.add(tool_message)

        total_elapsed = time.monotonic() - t_all
        if self.verbose and len(tool_calls) > 1:
            self.console.debug(f"All {len(tool_calls)} tool calls completed in {total_elapsed:.1f}s")

    # -- per-tool result display helpers ------------------------------------

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
                self.console.system("Request cancelled.")
                return None
            except RuntimeError as e:
                self.context.rollback(snapshot)
                self.console.error(f"API call failed: {e}")
                return None
            except Exception:
                raise

            if message.content:
                self.console.response(message.content)

            if not message.tool_calls:
                if self.verbose:
                    turn_elapsed = time.monotonic() - t_turn_start
                    self.console.debug(f"Turn complete: {iteration} iteration(s) in {turn_elapsed:.1f}s")
                return message.content if message.content else None

            self.execute_tool_calls(message.tool_calls)

            if self.verbose:
                iter_elapsed = time.monotonic() - t_iter
                self.console.debug(f"Iteration {iteration} total: {iter_elapsed:.1f}s")

    # ----- bash command (!) ------------------------------------------------

    def _run_bash_command(self, command: str) -> None:
        """Execute a shell command starting with '!' directly, without sending to the model.

        Output is streamed in real-time and the command can be interrupted via Ctrl+C.
        """
        self.console.tool(f"! {command}")
        with self._operation("shell command"):
            result = self.tool_executor.execute("exec_command", {"command": command})
        result_text = format_tool_result(result)
        success = result.get("success", False)
        if result.get("streamed", True):
            exit_code = result.get("exit_code", "?")
            self.console.tool_result(f"exit code {exit_code}", success=success)
        else:
            self.console.tool_result(format_tool_summary(result_text), success=success)

    # ----- slash commands ---------------------------------------------------

    COMMANDS = {"help", "compact", "usage", "exit", "quit", "clear", "resume", "perm", "model", "context", "rewind", "skills"}

    def _handle_command(self, raw: str) -> bool:
        """Handle a /command.  Returns True if the agent loop should stop."""
        cmd = raw.split()[0][1:].lower()

        if cmd in ("exit", "quit"):
            return True

        if cmd == "help":
            self._cmd_help()
        elif cmd == "compact":
            self._cmd_compact()
        elif cmd == "usage":
            self._cmd_usage()
        elif cmd == "clear":
            self._cmd_clear()
        elif cmd == "resume":
            self._cmd_resume(raw)
        elif cmd == "perm":
            self._cmd_perm(raw)
        elif cmd == "model":
            self._cmd_model(raw)
        elif cmd == "context":
            self._cmd_context()
        elif cmd == "rewind":
            self._cmd_rewind(raw)
        elif cmd == "skills":
            self._cmd_skills()

        return False

    def _cmd_help(self):
        self.console.print(
            "[bold blue]Available commands:[/bold blue]\n"
            "\n"
            "  [cyan]!<cmd>[/cyan]       — Run a shell command directly (e.g. !ls, !git diff)\n"
            "  [cyan]/help[/cyan]        — Show this help message\n"
            "  [cyan]/context[/cyan]     — Show a concise one-line-per-message context log\n"
            "  [cyan]/compact[/cyan]     — Force context compaction via LLM summarization\n"
            "  [cyan]/usage[/cyan]       — Show token usage for this session\n"
            "  [cyan]/model[/cyan]       — Show or switch LLM model (/model <name>)\n"
            "  [cyan]/skills[/cyan]      — List installed skills from .kia/skills/\n"
            "  [cyan]/perm[/cyan]        — Show or change permission mode (/perm auto|default|strict)\n"
            "  [cyan]/rewind[/cyan]      — Roll back conversation and/or code to a previous round\n"
            "  [cyan]/clear[/cyan]       — Clear conversation history (keep system prompt)\n"
            "  [cyan]/resume[/cyan]      — Save current, then resume a previous session (/resume [session_id])\n"
            "  [cyan]/exit[/cyan]        — Exit the agent (also: /quit)\n"
            "\n"
            "  Press [bold]Enter[/bold] to send, [bold]Escape → Enter[/bold] for a newline."
        )

    def _cmd_compact(self):
        if len(self.context.messages) <= 2:
            self.console.system("Not enough messages to compact.")
            return
        before_chars = estimate_context_chars(self.context.messages)
        before_msgs = len(self.context.messages)
        before_tokens = self.token_estimator.chars_to_tokens(before_chars)
        self.console.system(
            f"Compacting conversation "
            f"({before_msgs} messages, ~{before_tokens:,} tokens)..."
        )
        self.context.replace_messages(
            compact_context(
                self.context.messages, self.client, self.model,
                console=self.console,
                context_length=self.context_length,
                chars_per_token=self.token_estimator.chars_per_token,
            )
        )
        after_chars = estimate_context_chars(self.context.messages)
        after_msgs = len(self.context.messages)
        after_tokens = self.token_estimator.chars_to_tokens(after_chars)
        saved_pct = (1 - after_chars / before_chars) * 100 if before_chars else 0
        self.console.system(
            f"Compaction complete: "
            f"{before_msgs} messages → {after_msgs} messages, "
            f"~{before_tokens:,} tokens → ~{after_tokens:,} tokens "
            f"(saved {saved_pct:.0f}%)"
        )

    def _cmd_usage(self):
        ctx_chars = estimate_context_chars(self.context.messages)
        ctx_tokens = self.token_estimator.chars_to_tokens(ctx_chars)
        ctx_pct = ctx_tokens / self.context_length * 100 if self.context_length else 0

        self.console.print(
            f"[bold blue]Session usage (round {self.round_id}):[/bold blue]\n"
            f"  Total tokens   : {self.token_totals['total']}\n"
            f"  Prompt tokens  : {self.token_totals['prompt']}  (cached: {self.token_totals['cached_prompt']})\n"
            f"  Output tokens  : {self.token_totals['completion']}  (reasoning: {self.token_totals['reasoning']})\n"
            f"  Context window : ~{ctx_tokens} / {self.context_length} tokens [{ctx_pct:.0f}%]\n"
            f"  Messages       : {len(self.context.messages)}"
        )

    def _cmd_clear(self):
        # Save the current session before starting fresh
        if self._session_id and self.context.messages:
            try:
                self.save_session(self._session_id)
                self.console.system(f"Session '{self._session_id}' saved.")
            except Exception as e:
                self.console.warn(f"Could not save session before clear: {e}")

        # Start a brand-new session
        self._session_id = time.strftime("%Y%m%d_%H%M%S")
        self._last_save_time = 0.0  # allow immediate save of the new session
        self.context.replace_messages([])
        self.round_id = 0
        # Create a fresh change tracker for the new session
        _wd = self.tool_executor._work_dir or os.getcwd()
        self.changes = ChangeTracker(self._session_id, _wd, self.console)
        self.tool_executor._change_tracker = self.changes
        self.tool_executor._get_round_id = lambda: self.round_id
        self.console.system(f"Started new session '{self._session_id}'.")

    @staticmethod
    def _session_preview(messages: list) -> str:
        """Short preview from the last user message of a saved session."""
        for m in reversed(messages):
            if not (isinstance(m, dict) and m.get("role") == "user"):
                continue
            content = m.get("content", "")
            if isinstance(content, list):
                text = " ".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            else:
                text = str(content)
            text = text.replace("\n", " ").strip()
            return text[:60] + ("..." if len(text) > 60 else "")
        return ""

    def _pick_session(self) -> str | None:
        """List saved sessions and let the user pick one interactively."""
        sessions_dir = self._sessions_dir()
        files = sorted(sessions_dir.glob("*.json"), reverse=True)
        # Don't offer the current (unsaved-in-progress) session as a target.
        files = [f for f in files if f.stem != self._session_id]
        if not files:
            self.console.system(f"No other saved sessions in {sessions_dir}")
            return None

        stems: list[str] = []
        labels: list[str] = []
        for f in files:
            stem = f.stem
            try:
                meta = json.loads(f.read_text(encoding="utf-8"))
                messages = meta.get("messages", [])
                n_msgs = len(messages)
                rnd = meta.get("round_id", "?")
                model = meta.get("model", "?")
                preview = self._session_preview(messages)
            except Exception:
                n_msgs, rnd, model, preview = "?", "?", "?", ""
            label = f"{stem}  │  msgs:{n_msgs}  rounds:{rnd}  model:{model}"
            if preview:
                label += f"  │  {preview}"
            stems.append(stem)
            labels.append(label)

        picked = self.console.select(message="Pick a session to resume", choices=labels)
        if picked is None:
            return None
        for stem, label in zip(stems, labels):
            if label == picked:
                return stem
        return None

    def _cmd_resume(self, raw: str):
        """Handle /resume — save the current session, then load a previous one."""
        parts = raw.split(maxsplit=1)
        target: str | None = parts[1].strip() if len(parts) > 1 else None

        if target:
            if not (self._sessions_dir() / f"{target}.json").exists():
                self.console.error(f"Session not found: {target}")
                return
        else:
            target = self._pick_session()
            if target is None:
                self.console.system("Resume cancelled.")
                return

        if target == self._session_id:
            self.console.system(f"Already in session '{target}'.")
            return

        # Save the current session before switching away.
        if self._session_id and self.context.messages:
            try:
                self.save_session(self._session_id)
                self.console.system(f"Session '{self._session_id}' saved.")
            except Exception as e:
                self.console.warn(f"Could not save current session: {e}")

        # Switch to the target session and reset per-session state.
        old_id = self._session_id
        old_save_time = self._last_save_time
        self._session_id = target
        self._last_save_time = 0.0
        if not self.load_session(target):
            self._session_id = old_id
            self._last_save_time = old_save_time
            return
        _wd = self.tool_executor._work_dir or os.getcwd()
        self.changes = ChangeTracker(self._session_id, _wd, self.console)
        self.tool_executor._change_tracker = self.changes
        self.tool_executor._get_round_id = lambda: self.round_id

    def _cmd_rewind(self, raw: str):
        """Handle /rewind — roll back to a previous round."""
        if not self.changes:
            self.console.warn("Rewind is only available in interactive chat mode with a session.")
            return

        rounds = self.changes.build_round_list(self.context.messages)
        if not rounds:
            self.console.system("No rounds to rewind.")
            return

        parts = raw.split()
        target_round: int | None = None

        if len(parts) >= 2:
            try:
                target_round = int(parts[1])
                if target_round < 1 or target_round > len(rounds):
                    self.console.error(f"Round must be between 1 and {len(rounds)}.")
                    return
            except ValueError:
                self.console.error(f"Invalid round number: {parts[1]}")
                return
        else:
            # Build round choices for questionary picker
            round_choices: list[str] = []
            round_map: dict[str, int] = {}  # choice label → round number
            for r in reversed(rounds):
                n = r["round"]
                preview = r["preview"]
                files = r.get("files", 0)
                added = r.get("added", 0)
                removed = r.get("removed", 0)
                # Build change summary
                parts_line = []
                if files > 0:
                    parts_line.append(f"{files} file{'s' if files > 1 else ''}")
                if added > 0:
                    parts_line.append(f"+{added} line{'s' if added > 1 else ''}")
                if removed > 0:
                    parts_line.append(f"-{removed} line{'s' if removed > 1 else ''}")
                change_str = ""
                if parts_line:
                    change_str = f"  ({', '.join(parts_line)})"
                elif files == 0 and added == 0 and removed == 0:
                    change_str = "  (no file changes)"
                label = f"Round {n:>3} — {preview}{change_str}"
                round_choices.append(label)
                round_map[label] = n

            picked = self.console.select(
                message="Pick a round to revert",
                choices=round_choices,
            )
            if picked is None:
                return
            target_round = round_map.get(picked)
            if target_round is None or target_round < 1 or target_round > len(rounds):
                self.console.error(f"Invalid choice: {picked}")
                return

        # Revert to the state BEFORE target_round
        actual_target = target_round - 1

        # Ask revert mode
        self.console.print(f"\n[bold blue]? Revert round {target_round}?[/bold blue]")
        if actual_target > 0:
            preview = rounds[actual_target - 1]["preview"]
            self.console.print(f"  Will roll back to after round [cyan]{actual_target}[/cyan] — {preview}")
        else:
            self.console.print(f"  Will roll back to [cyan]initial state[/cyan] (before any messages)")

        # Show what will be reverted
        reverted_rounds = list(range(target_round, len(rounds) + 1))
        total_files = 0
        total_added = 0
        total_removed = 0
        for r in rounds:
            if r["round"] >= target_round:
                total_files += r.get("files", 0)
                total_added += r.get("added", 0)
                total_removed += r.get("removed", 0)
        if total_files > 0 or total_added > 0 or total_removed > 0:
            impact = []
            if total_files > 0:
                impact.append(f"{total_files} file{'s' if total_files > 1 else ''}")
            if total_added > 0:
                impact.append(f"+{total_added} lines")
            if total_removed > 0:
                impact.append(f"-{total_removed} lines")
            self.console.print(f"  Code impact: [yellow]{', '.join(impact)}[/yellow] across {len(reverted_rounds)} round{'s' if len(reverted_rounds) > 1 else ''}")

        mode_choice = self.console.select(
            message="Choose revert mode",
            choices=[
                "1. Conversation only (keep code changes)",
                "2. Code + conversation (restore both)",
                "3. Code only (keep conversation)",
            ],
        )
        if mode_choice is None:
            return
        mode = mode_choice[0]  # "1", "2", or "3"

        revert_code = mode in ("2", "3")
        revert_conv = mode in ("1", "2")

        # Execute rollback
        self.console.system(f"Reverting round {target_round} and later...")

        if revert_code:
            restored = self.changes.rollback_code(actual_target)
            self.console.system(f"Code reverted: {restored} files restored.")
        else:
            self.console.system("Code changes preserved.")

        if revert_conv:
            self.context.messages, self.round_id = self.changes.rollback_conversation(
                self.context.messages, actual_target
            )
            self.console.system(f"Conversation rolled back to round {self.round_id} ({len(self.context.messages)} messages).")
        else:
            self.console.system("Conversation preserved.")

        # Persist the rewinded state to disk
        try:
            self.save_session(self.changes.session_id)
        except Exception:
            pass

    def _cmd_context(self):
        msgs = self.context.messages
        if not msgs:
            self.console.system("Context is empty (no messages).")
            return

        total_chars = 0
        lines = [f"[bold blue]Context log ({len(msgs)} messages):[/bold blue]"]

        def _tc_name(tc) -> str:
            if isinstance(tc, dict):
                return tc.get("function", {}).get("name", "?")
            return getattr(getattr(tc, "function", None), "name", "?")

        def _tc_id(tc) -> str | None:
            return tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)

        tc_id_to_name: dict[str, str] = {}
        for m in msgs:
            if get_role(m) == "assistant":
                for tc in get_tool_calls(m):
                    tid = _tc_id(tc)
                    if tid:
                        tc_id_to_name[tid] = _tc_name(tc)

        for idx, m in enumerate(msgs):
            role = get_role(m)
            text = get_text(m)
            chars = msg_chars(m)
            total_chars += chars

            preview = text.replace("\n", " ").strip()
            if len(preview) > 80:
                preview = preview[:77] + "..."

            role_tag = f"[cyan]{role:>9}[/cyan]"
            size_tag = f"[dim]{chars:>6} ch[/dim]"

            tcs = get_tool_calls(m) if role == "assistant" else []
            if tcs:
                n_tc = len(tcs)
                tc_names = ", ".join(_tc_name(tc) for tc in tcs)
                extra = f"[yellow]{n_tc} call{'s' if n_tc > 1 else ''}[/yellow] ({tc_names})"
                if preview:
                    lines.append(f"  [dim]#{idx:<3}[/dim] {role_tag} {size_tag}  {extra}  {preview}")
                else:
                    lines.append(f"  [dim]#{idx:<3}[/dim] {role_tag} {size_tag}  {extra}")
            elif role == "tool":
                tid = get_tool_call_id(m)
                tool_name = tc_id_to_name.get(tid, "?") if tid else "?"
                lines.append(f"  [dim]#{idx:<3}[/dim] {role_tag} {size_tag}  [magenta]({tool_name})[/magenta]  {preview}")
            else:
                lines.append(f"  [dim]#{idx:<3}[/dim] {role_tag} {size_tag}  {preview}")

        est_tokens = self.token_estimator.chars_to_tokens(total_chars)
        ctx_pct = est_tokens / self.context_length * 100 if self.context_length else 0
        lines.append(f"\n  [bold]Total:[/bold] ~{est_tokens:,} tokens / {self.context_length:,} [{ctx_pct:.0f}%]")

        self.console.print("\n".join(lines))

    def _cmd_perm(self, raw: str):
        parts = raw.split()
        if len(parts) < 2:
            mode = self.permissions.mode.value
            allowed = ", ".join(sorted(self.permissions.session_allowed_tools)) or "(none)"
            work_dir = self.permissions.safety.work_dir
            self.console.print(
                f"[bold blue]Permission mode:[/bold blue] [cyan]{mode}[/cyan]\n"
                f"  Session-allowed tools: {allowed}\n"
                f"  Safety guard work dir: {work_dir}\n"
                f"  Usage: [cyan]/perm auto|default|strict[/cyan]"
            )
            return

        target = parts[1].lower()
        valid = {m.value for m in PermissionMode}
        if target not in valid:
            self.console.error(f"Unknown mode '{target}'. Choose from: {', '.join(sorted(valid))}")
            return

        new_mode = PermissionMode(target)
        self.permissions.mode = new_mode
        self.permissions.reset_session()
        self.console.system(f"Permission mode changed to: {new_mode.value}")

    def _cmd_model(self, raw: str):
        from kiui.config import conf

        parts = raw.split(maxsplit=1)
        openai_conf = conf.get("openai", {})

        if len(parts) < 2 or not parts[1].strip():
            lines = [
                f"[bold blue]Current model:[/bold blue] [cyan]{self.model}[/cyan]"
                + (f" (alias: {self.model_alias})" if self.model_alias else ""),
                f"[bold blue]Available models:[/bold blue]",
            ]
            for name, mc in openai_conf.items():
                marker = " [green]◀[/green]" if name == self.model_alias else ""
                lines.append(f"  [cyan]{name}[/cyan] → {mc.get('model', name)}{marker}")
            lines.append(f"\n  Usage: [cyan]/model <name>[/cyan]")
            self.console.print("\n".join(lines))
            return

        target = parts[1].strip()
        if target not in openai_conf:
            self.console.error(f"Model '{target}' not found in config. Use /model to list available models.")
            return

        if target == self.model_alias:
            self.console.system(f"Already using model '{target}'.")
            return

        model_conf = openai_conf[target]
        self.model = model_conf.get("model", target)
        self.model_alias = target
        self._api_key = model_conf.get("api_key", "")
        self._base_url = model_conf.get("base_url", "")
        self.client = OpenAI(api_key=self._api_key, base_url=self._base_url, max_retries=0)
        self.profile = resolve_model_profile(self.model, self.model_alias)
        self.context_length = self.profile.context_length

        if self.subagent_manager:
            self.subagent_manager.model_alias = target

        self.console.system(
            f"Switched to model: {self.model} "
            f"(context: {self.context_length:,} tokens, thinking: {self.profile.thinking or 'none'})"
        )

    def _cmd_skills(self):
        """List installed skills from .kia/skills/."""
        if not self.skills:
            skills_dir = (Path(self.tool_executor._work_dir) if self.tool_executor._work_dir else Path.cwd()) / ".kia" / "skills"
            self.console.print(
                f"[bold blue]No skills installed.[/bold blue]\n"
                f"\n"
                f"  Skills are folders under [cyan]{skills_dir}[/cyan] each containing a [cyan]SKILL.md[/cyan] file.\n"
                f"\n"
                f"  [bold]Example:[/bold]\n"
                f"    [cyan]{skills_dir / 'git-workflow' / 'SKILL.md'}[/cyan]\n"
                f"\n"
                f"  The SKILL.md should describe what the skill does and provide domain-specific instructions.\n"
                f"  When a skill is relevant, the model can invoke [cyan]load_skill[/cyan] to load its full prompt."
            )
            return

        self.console.print(f"[bold blue]Installed skills ({len(self.skills)}):[/bold blue]\n")
        for name, info in sorted(self.skills.items()):
            desc = info.get("description", "")
            path = info.get("path", "")
            loaded = " [green](loaded)[/green]" if name in self.tool_executor._loaded_skills else ""
            self.console.print(f"  [cyan]{name}[/cyan]{loaded}")
            self.console.print(f"    {desc}")
            self.console.print(f"    [dim]{path}[/dim]")
            self.console.print()

    # ----- session save / load ------------------------------------------------

    SESSIONS_DIR_NAME = "sessions"

    def _sessions_dir(self) -> Path:
        d = get_kia_dir() / self.SESSIONS_DIR_NAME
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _serialize_message(msg) -> dict:
        """Convert a message (dict or OpenAI object) to a plain dict for JSON."""
        if isinstance(msg, dict):
            return msg
        if hasattr(msg, "model_dump"):
            return msg.model_dump(exclude_none=True)
        return dict(msg)

    def save_session(self, name: str | None = None) -> Path:
        """Save the current session state to a JSON file. Returns the file path."""
        if not name:
            name = time.strftime("%Y%m%d_%H%M%S")

        data = {
            "model": self.model,
            "round_id": self.round_id,
            "token_totals": self.token_totals,
            "system_prompt": self.context.system_prompt,
            "messages": [self._serialize_message(m) for m in self.context.messages],
        }

        path = self._sessions_dir() / f"{name}.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def load_session(self, name: str) -> bool:
        """Load a previously saved session by name. Returns True on success."""
        path = self._sessions_dir() / f"{name}.json"
        if not path.exists():
            self.console.error(f"Session file not found: {path}")
            return False
        
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            self.console.error(f"Failed to read session file: {e}")
            return False

        if not isinstance(data, dict) or "messages" not in data:
            self.console.error(f"Corrupted session file (missing 'messages' key): {path}")
            return False

        saved_model = data.get("model", "")
        if saved_model != self.model:
            self.console.system(f"Note: session was saved with model '{saved_model}', current model is '{self.model}'")

        self.context.replace_messages(data["messages"])
        self.round_id = data.get("round_id", 0)
        saved_totals = data.get("token_totals")
        if saved_totals and isinstance(saved_totals, dict):
            self.token_totals.update(saved_totals)

        self.console.system(
            f"Loaded session '{name}' ({len(self.context.messages)} messages, round {self.round_id})"
        )

        # Replay the conversation so user can recall the context
        self._replay_context()
        return True

    def _replay_context(self):
        """Replay all messages using the same display methods as live chat."""
        msgs = self.context.messages
        if not msgs:
            return

        self.console.system("── Session context (replay) ──")

        for msg in msgs:
            role = get_role(msg)
            if role == "user":
                text = get_text(msg)
                self.console.user_input(text)

            elif role == "assistant":
                text = get_text(msg)
                if text:
                    self.console.response(text)
                for tc in get_tool_calls(msg):
                    if isinstance(tc, dict):
                        fn = tc.get("function", {})
                        fname = fn.get("name", "?")
                        try:
                            fargs = json.loads(fn.get("arguments", "{}"))
                        except (json.JSONDecodeError, TypeError):
                            fargs = {}
                    else:
                        fname = getattr(getattr(tc, "function", None), "name", "?")
                        try:
                            fargs = json.loads(getattr(getattr(tc, "function", None), "arguments", "{}"))
                        except (json.JSONDecodeError, TypeError):
                            fargs = {}
                    self.console.tool(f"{fname}({json.dumps(fargs, ensure_ascii=False)})")

            elif role == "tool":
                result_text = get_text(msg)
                success = "error" not in result_text.lower()
                summary = format_tool_summary(result_text)
                self.console.tool_result(summary, success=success)

        self.console.system("── End of replay ──")



    # ----- main loops -------------------------------------------------------

    def _next_submission(self, terminal: TerminalInput) -> UserSubmission:
        """Wait for terminal or web input and return whichever arrives first."""
        if self.input_broker is None:
            return UserSubmission(
                text=terminal.prompt(), source="terminal", id="terminal"
            )

        try:
            return self.input_broker.get_nowait()
        except queue.Empty:
            pass

        async def wait_web() -> UserSubmission:
            while True:
                try:
                    return self.input_broker.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(INPUT_POLL_INTERVAL)

        async def race() -> UserSubmission:
            terminal_task = asyncio.create_task(terminal.prompt_async())
            web_task = asyncio.create_task(wait_web())
            done, _ = await asyncio.wait(
                {terminal_task, web_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if web_task in done:
                if not terminal_task.done():
                    terminal_task.cancel()
                try:
                    await terminal_task
                except (asyncio.CancelledError, EOFError, KeyboardInterrupt):
                    pass
                return web_task.result()
            if not web_task.done():
                web_task.cancel()
            try:
                await web_task
            except asyncio.CancelledError:
                pass
            return UserSubmission(
                text=terminal_task.result(), source="terminal", id="terminal"
            )

        return asyncio.run(race())

    def chat_loop(self, resumed_session_id: str | None = None):

        self.console.system("Type `/help` for available commands. Prefix with `!` to run shell commands.")
        self.console.system("`Enter` to send. `Escape` then `Enter` for a newline.")
        self.console.system("Current working directory: " + os.getcwd())

        # Auto-save session id: reuse resumed session or generate a new one
        self._session_id = resumed_session_id or time.strftime("%Y%m%d_%H%M%S")

        # Initialize change tracker for code+conversation rollback
        _wd = self.tool_executor._work_dir or os.getcwd()
        if self.changes is None:
            self.changes = ChangeTracker(self._session_id, _wd, self.console)
        # Wire the tracker into the tool executor for per-operation tracking
        self.tool_executor._change_tracker = self.changes
        self.tool_executor._get_round_id = lambda: self.round_id

        terminal = TerminalInput(history_path=str(get_kia_dir() / "history"), work_dir=_wd)

        try:
            while True:
                try:
                    self.console.rule()
                    submission = self._next_submission(terminal)
                    query = _strip_at_marks(submission.text.strip())
                except KeyboardInterrupt:
                    # Ctrl+C is handled inside the prompt (clear / double-tap
                    # to quit); a stray one here just starts a fresh prompt.
                    continue
                except EOFError:
                    # Ctrl+D, or double Ctrl+C on an empty prompt → quit.
                    break

                if not query:
                    continue

                self.console.user_input(
                    query,
                    source=submission.source,
                    submission_id=submission.id,
                    with_rule=False,
                )

                # exit shortcut
                if query.lower() in ("exit", "quit"):
                    break

                # bash command shortcut: !<command>
                if query.startswith("!"):
                    bash_cmd = query[1:].strip()
                    if bash_cmd:
                        self._run_bash_command(bash_cmd)
                    else:
                        self.console.warn("Usage: !<shell command>")
                    continue

                # slash commands
                if query.startswith("/"):
                    cmd_word = query.split()[0][1:].lower()
                    if cmd_word in self.COMMANDS:
                        if self._handle_command(query):
                            break
                        continue
                    else:
                        self.console.warn(f"Unknown command: /{cmd_word}. Type /help for available commands.")
                        continue

                user_message = {"role": "user", "content": query}
                self.context.add(user_message)

                self.round_id += 1

                self.console.rule()

                with self._operation("agent response"):
                    self.get_response()

                # Auto-save after each round (throttled: ≤ once per 30 s)
                try:
                    now = time.monotonic()
                    if now - self._last_save_time >= 30:
                        self.save_session(self._session_id)
                        self._last_save_time = now
                except Exception:
                    pass  # never let save failure break the loop
        finally:
            self._print_token_summary()
        
    def execute(self, query: str):
        self.console.system(f"Executing query: {query}")
        t0 = time.time()

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

        with self._operation("agent response"):
            response = self.get_response()

        t1 = time.time()
        self.console.system(f"Execution time: {t1 - t0:.2f} seconds")
        self._print_token_summary()
        return response

    def _print_token_summary(self):
        self.console.system(
            f"Total tokens used: {self.token_totals['total']} "
            f"(input: {self.token_totals['prompt']}, cached input: {self.token_totals['cached_prompt']}, "
            f"output: {self.token_totals['completion']}, reasoning: {self.token_totals['reasoning']})"
        )

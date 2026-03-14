import os
import re
import sys
import time
import json
import atexit
from typing import Literal
from openai import OpenAI
from kiui.agent.ui import AgentConsole
from kiui.agent.terminal import TerminalInput
from kiui.agent.utils import get_text_content_dict, get_image_content_dict, get_kia_dir
from kiui.agent.prompts import build_system_prompt
from kiui.agent.tools import get_tool_definitions, ToolExecutor, format_tool_result, format_tool_summary
from kiui.agent.subagent import SubagentManager
from kiui.agent.permissions import PermissionController, PermissionMode
from kiui.agent.context import (
    truncate_tool_result,
    prune_context,
    needs_compaction,
    compact_context,
    estimate_context_chars,
    CHARS_PER_TOKEN,
)
from kiui.agent.interrupt import InterruptHandler
from kiui.agent.models import resolve_model_profile

def parse_custom_query(query: str) -> list:
    """Parse the user query into formatted content, optionally load images or text files (by using @filename).
    Supported file types:
    - @image.png/jpg/jpeg: load an image
    - @file.txt/md/json/yaml: read a local text file
    e.g. "Please describe the following image @path/to/image.png, and tell me the weather." will be parsed into: 
    [
        {"type": "text", "text": "Please describe the following image"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...", "detail": "low"}},
        {"type": "text", "text": "and tell me the weather."}
    ]
    """
    content = []

    pattern = re.compile(r"@([\w.\/\\:-]+)")
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
    text_extensions = {'.txt', '.md', '.json', '.yaml', '.yml'}
    
    last_end = 0
    for match in pattern.finditer(query):
        pre_text = query[last_end:match.start()].strip()
        if pre_text:
            content.append(get_text_content_dict(pre_text))

        file_path = match.group(1)
        _, ext = os.path.splitext(file_path.lower())
        
        if ext in image_extensions:
            content.append(get_image_content_dict(file_path))
        elif ext in text_extensions:
            with open(file_path, "r") as f:
                file_content = f.read()
            content.append(get_text_content_dict(file_content))
        else:
            content.append(get_text_content_dict(f"@{file_path}"))
            
        last_end = match.end()

    tail = query[last_end:].strip()
    if tail:
        content.append(get_text_content_dict(tail))

    return content

class ContextManager:
    """Flat conversation history with context-management hooks."""
    def __init__(self, system_prompt: str):
        self.system_prompt = {
            "role": "system",
            "content": system_prompt,
        }
        self.messages: list = []

    def add(self, content):
        self.messages.append(content)

    def get(self, include_system: bool = True) -> list:
        res = []
        if include_system:
            res.append(self.system_prompt)
        res.extend(self.messages)
        return res

    def replace_messages(self, new_messages: list):
        """Replace all messages (used after compaction)."""
        self.messages = list(new_messages)

    def checkpoint(self) -> list:
        """Return a shallow copy of messages for rollback on interruption."""
        return list(self.messages)

    def rollback(self, snapshot: list):
        """Restore messages to a previous checkpoint."""
        self.messages = snapshot


class LLMAgent:
    def __init__(
        self, 
        model: str,
        api_key: str,
        base_url: str,
        model_key: str = "",
        verbose: bool = True,
        thinking_budget: Literal["low", "medium", "high"] = "low",
        pipe_mode: bool = False,
        context_window: int | None = None,
        permission_mode: PermissionMode = PermissionMode.DEFAULT,
    ):

        self.model = model
        self.model_key = model_key
        self.profile = resolve_model_profile(model, model_key)
        self._api_key = api_key
        self._base_url = base_url
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.verbose = verbose
        self.thinking_budget = thinking_budget
        self.pipe_mode = pipe_mode
        self.context_window = context_window if context_window is not None else self.profile.context_window

        self.console = AgentConsole(pipe_mode=pipe_mode)

        # built-in system prompt and tools
        self.system_prompt = build_system_prompt()
        self.tools = get_tool_definitions()

        # permission controller — auto in pipe mode regardless of explicit setting
        effective_mode = PermissionMode.AUTO if pipe_mode else permission_mode
        self.permissions = PermissionController(mode=effective_mode, console=self.console)

        # subagent manager (only if we have a model_key for spawning children)
        self.subagent_manager = SubagentManager(model_key=model_key) if model_key else None
        self.tool_executor = ToolExecutor(console=self.console, subagent_manager=self.subagent_manager)

        if self.subagent_manager:
            atexit.register(self.subagent_manager.kill_all)

        self.context = ContextManager(self.system_prompt)
        self.interrupt = InterruptHandler()

        self.round_id = 0

        self.token_totals = {
            "total": 0,
            "prompt": 0,
            "cached_prompt": 0,
            "completion": 0,
            "reasoning": 0,
        }

        self.console.system(f"Created Agent with model: {model} (context: {self.context_window:,} tokens)")
        self.console.system(f"Permission mode: {effective_mode.value}")
        if self.verbose:
            self.console.system(f"System prompt: {self.system_prompt[:100]}...")


    def _recreate_client(self):
        """Recreate the OpenAI client after an interrupt-driven close."""
        self.client = OpenAI(api_key=self._api_key, base_url=self._base_url)

    def _accumulate_usage(self, usage):
        """Add token counts from an API response to session totals."""
        self.token_totals["total"] += usage.total_tokens
        self.token_totals["prompt"] += usage.prompt_tokens
        self.token_totals["completion"] += usage.completion_tokens
        if usage.prompt_tokens_details and usage.prompt_tokens_details.cached_tokens:
            self.token_totals["cached_prompt"] += usage.prompt_tokens_details.cached_tokens
        if usage.completion_tokens_details and usage.completion_tokens_details.reasoning_tokens:
            self.token_totals["reasoning"] += usage.completion_tokens_details.reasoning_tokens

    def _inject_pending_subagent_results(self):
        """Inject any completed sub-agent results into the conversation."""
        if not self.subagent_manager:
            return
        pending = self.subagent_manager.get_pending_results()
        for msg in pending:
            self.context.add(msg)
            label = msg.get("content", "")[:80]
            self.console.system(f"Sub-agent completed: {label}")


    def call_api(self):
        """Call the API using current context."""

        # inject any pending sub-agent results before calling API
        self._inject_pending_subagent_results()

        # context management: prune old tool results, then compact if needed
        if self.context_window > 0:
            self.context.replace_messages(
                prune_context(self.context.messages, self.context_window)
            )
            if needs_compaction(self.context.messages, self.context_window):
                self.console.system("Context window pressure — compacting via LLM summarization...")
                self.context.replace_messages(
                    compact_context(self.context.messages, self.client, self.model, console=self.console)
                )
                self.console.system("Compaction complete.")

        if self.verbose:
            ctx_chars = estimate_context_chars(self.context.messages)
            ctx_pct = ctx_chars / (self.context_window * CHARS_PER_TOKEN) * 100 if self.context_window else 0
            self.console.debug(
                f"Calling API (round: {self.round_id}, "
                f"context: ~{ctx_chars // CHARS_PER_TOKEN}tok / {self.context_window}tok [{ctx_pct:.0f}%])"
            )

        messages = self.context.get()

        kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        if self.tools:
            kwargs["tools"] = self.tools
            kwargs["tool_choice"] = "auto"
        
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

        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        usage = response.usage

        self._accumulate_usage(usage)
        self.context.add(message)

        if self.verbose:
            cached = getattr(usage.prompt_tokens_details, "cached_tokens", None) if usage.prompt_tokens_details else None
            reasoning = getattr(usage.completion_tokens_details, "reasoning_tokens", None) if usage.completion_tokens_details else None
            self.console.debug(
                f"API Response total_tokens: {usage.total_tokens} = "
                f"output: {usage.completion_tokens} (reasoning: {reasoning or 'N/A'}) "
                f"input: {usage.prompt_tokens} (cached: {cached or 'N/A'})"
            )
            if message.tool_calls:
                self.console.debug(f"Requested tool calls: {len(message.tool_calls)}")

        return message


    def execute_tool_calls(self, tool_calls: list):
        """Execute tool calls via the built-in ToolExecutor."""

        for i, tool_call in enumerate(tool_calls):
            if self.interrupt.interrupted:
                break
            function_name = tool_call.function.name
            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                function_args = {}
                self.console.error(f"Failed to parse tool args: {e}")
            
            if self.verbose:
                self.console.debug(f"Tool call {i+1}/{len(tool_calls)}: {function_name}({function_args})")

            allowed, denial_reason = self.permissions.check(function_name, function_args)
            if not allowed:
                msg = f"Tool call denied by user: {function_name}"
                if denial_reason:
                    msg += f"\nUser feedback: {denial_reason}"
                result = {"error": msg, "success": False}
            else:
                result = self.tool_executor.execute(function_name, function_args)
            result_text = format_tool_result(result)

            success = result.get("success", False)
            self.console.tool_result(format_tool_summary(result_text), success=success)

            # Layer 1: generic truncation relative to context window
            if self.context_window > 0:
                result_text = truncate_tool_result(result_text, self.context_window)

            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_text,
            }
            self.context.add(tool_message)

    def get_response(self):
        """Process the context and update the current response.

        Supports flag-based cancellation: if interrupted, rolls back messages
        from the current iteration and recreates the HTTP client.
        """

        while True:
            if self.interrupt.interrupted:
                return None

            snapshot = self.context.checkpoint()

            try:
                message = self.call_api()
            except Exception:
                if self.interrupt.interrupted:
                    self.context.rollback(snapshot)
                    self._recreate_client()
                    self.console.system("Interrupted — partial response rolled back.")
                    return None
                raise

            if message.content and not self.pipe_mode:
                self.console.response(message.content)

            if not message.tool_calls:
                return message.content if message.content else None

            self.execute_tool_calls(message.tool_calls)

            if self.interrupt.interrupted:
                self.context.rollback(snapshot)
                self._recreate_client()
                self.console.system("Interrupted — partial response rolled back.")
                return None

    # ----- slash commands ---------------------------------------------------

    COMMANDS = {"help", "compact", "usage", "exit", "quit", "clear", "perm"}

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
        elif cmd == "perm":
            self._cmd_perm(raw)

        return False

    def _cmd_help(self):
        self.console.print(
            "[bold blue]Available commands:[/bold blue]\n"
            "\n"
            "  [cyan]/help[/cyan]     — Show this help message\n"
            "  [cyan]/compact[/cyan]  — Force context compaction via LLM summarization\n"
            "  [cyan]/usage[/cyan]    — Show token usage for this session\n"
            "  [cyan]/perm[/cyan]     — Show or change permission mode (/perm auto|default|strict)\n"
            "  [cyan]/clear[/cyan]    — Clear conversation history (keep system prompt)\n"
            "  [cyan]/exit[/cyan]     — Exit the agent (also: /quit)\n"
            "\n"
            "  Attach files with @filename (images & text files supported).\n"
            "  Press [bold]Enter[/bold] to send, [bold]Escape → Enter[/bold] for a newline."
        )

    def _cmd_compact(self):
        if len(self.context.messages) <= 2:
            self.console.system("Not enough messages to compact.")
            return
        self.console.system("Compacting conversation...")
        self.context.replace_messages(
            compact_context(self.context.messages, self.client, self.model, console=self.console)
        )
        self.console.system("Compaction complete.")

    def _cmd_usage(self):
        ctx_chars = estimate_context_chars(self.context.messages)
        ctx_tokens = ctx_chars // CHARS_PER_TOKEN
        ctx_pct = ctx_tokens / self.context_window * 100 if self.context_window else 0

        self.console.print(
            f"[bold blue]Session usage (round {self.round_id}):[/bold blue]\n"
            f"  Total tokens   : {self.token_totals['total']}\n"
            f"  Prompt tokens  : {self.token_totals['prompt']}  (cached: {self.token_totals['cached_prompt']})\n"
            f"  Output tokens  : {self.token_totals['completion']}  (reasoning: {self.token_totals['reasoning']})\n"
            f"  Context window : ~{ctx_tokens} / {self.context_window} tokens [{ctx_pct:.0f}%]\n"
            f"  Messages       : {len(self.context.messages)}"
        )

    def _cmd_clear(self):
        self.context.replace_messages([])
        self.round_id = 0
        self.console.system("Conversation cleared.")

    def _cmd_perm(self, raw: str):
        parts = raw.split()
        if len(parts) < 2:
            mode = self.permissions.mode.value
            allowed = ", ".join(sorted(self.permissions._session_allowed)) or "(none)"
            self.console.print(
                f"[bold blue]Permission mode:[/bold blue] [cyan]{mode}[/cyan]\n"
                f"  Session-allowed tools: {allowed}\n"
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
        self.permissions._session_allowed.clear()
        self.console.system(f"Permission mode changed to: {new_mode.value}")

    # ----- main loops -------------------------------------------------------

    def chat_loop(self):

        self.console.print("[system][SYSTEM] Type [bold]/help[/bold] for available commands.[/system]")
        self.console.system("`Enter` to send. `Escape` then `Enter` for a newline.")
        self.console.system("Ctrl+C to interrupt. Double Ctrl+C to force quit.")
        self.console.system("Current working directory: " + os.getcwd())

        terminal = TerminalInput(history_path=str(get_kia_dir() / "history"))
        self.interrupt.install(self)

        try:
            while True:
                try:
                    query = terminal.prompt().strip()
                except KeyboardInterrupt:
                    self.console.print("")
                    continue
                except EOFError:
                    break

                if not query:
                    continue

                # exit shortcut
                if query.lower() in ("exit", "quit"):
                    break

                # slash commands
                if query.startswith("/"):
                    cmd_word = query.split()[0][1:].lower()
                    if cmd_word in self.COMMANDS:
                        if self._handle_command(query):
                            break
                        continue

                user_message = {"role": "user", "content": parse_custom_query(query)}
                self.context.add(user_message)

                self.interrupt.reset()
                self.interrupt.set_task_running(True)
                try:
                    self.get_response()
                finally:
                    self.interrupt.set_task_running(False)

                self.round_id += 1
        finally:
            self.interrupt.uninstall()
            self._print_token_summary()
        
    def execute(self, query: str):
        self.console.system(f"Executing query: {query}")
        t0 = time.time()
        self.interrupt.install(self)

        user_message = {"role": "user", "content": parse_custom_query(query)}
        self.context.add(user_message)

        self.interrupt.reset()
        self.interrupt.set_task_running(True)
        try:
            response = self.get_response()
        finally:
            self.interrupt.set_task_running(False)
            self.interrupt.uninstall()

        t1 = time.time()
        self.console.system(f"Execution time: {t1 - t0:.2f} seconds")
        self._print_token_summary()
        return response

    def run_pipe_mode(self):
        """Run in pipe mode: read JSON from stdin, write JSON to stdout.

        Used by persistent sub-agent sessions. The parent process communicates
        via newline-delimited JSON on stdin/stdout.

        Input:  {"type": "message", "content": "do X"}
        Output: {"type": "response", "content": "Done.", "usage": {...}}
        Shutdown: {"type": "shutdown"} or EOF on stdin
        """
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                self._pipe_write({"type": "error", "error": "Invalid JSON"})
                continue

            if msg.get("type") == "shutdown":
                break

            if msg.get("type") != "message":
                self._pipe_write({"type": "error", "error": f"Unknown type: {msg.get('type')}"})
                continue

            user_content = msg.get("content", "")
            if not user_content:
                self._pipe_write({"type": "error", "error": "Empty message"})
                continue

            user_message = {"role": "user", "content": parse_custom_query(user_content)}
            self.context.add(user_message)

            try:
                self.get_response()
            except Exception as e:
                self._pipe_write({"type": "error", "error": str(e)})
                continue

            response_text = self._last_assistant_text()
            self._pipe_write({
                "type": "response",
                "content": response_text,
                "usage": self.token_totals,
            })
            self.round_id += 1

    def _last_assistant_text(self) -> str:
        """Return the text content of the most recent assistant message."""
        for m in reversed(self.context.messages):
            if isinstance(m, dict):
                if m.get("role") == "assistant" and m.get("content"):
                    return m["content"]
            elif getattr(m, "role", None) == "assistant" and getattr(m, "content", None):
                return m.content
        return ""

    def _pipe_write(self, data: dict):
        """Write a JSON line to stdout (pipe protocol)."""
        sys.stdout.write(json.dumps(data) + "\n")
        sys.stdout.flush()

    def _print_token_summary(self):
        self.console.system(
            f"Total tokens used: {self.token_totals['total']} "
            f"(input: {self.token_totals['prompt']}, cached input: {self.token_totals['cached_prompt']}, "
            f"output: {self.token_totals['completion']}, reasoning: {self.token_totals['reasoning']})"
        )

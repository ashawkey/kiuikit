import os
import re
import time
import json
from pathlib import Path
from typing import Any, Literal
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
    get_role,
    get_text,
    get_tool_calls,
    get_tool_call_id,
    msg_chars,
)
from kiui.agent.interrupt import InterruptHandler
from kiui.agent.models import resolve_model_profile

def parse_custom_query(query: str) -> list:
    """Parse the user query into formatted content, optionally load images or text files (by using @filename).
    Supported file types:
    - @image.png/jpg/jpeg: load an image
    - @file.txt/md/json/yaml: read a local text file
    Use @"path with spaces/file.txt" for paths containing spaces or special characters.
    e.g. "Please describe the following image @path/to/image.png, and tell me the weather." will be parsed into: 
    [
        {"type": "text", "text": "Please describe the following image"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...", "detail": "low"}},
        {"type": "text", "text": "and tell me the weather."}
    ]
    """
    content = []

    # Match @path (unquoted) or @"path with spaces" (quoted)
    pattern = re.compile(r'@"([^"]+)"|@([\w.\/\\:-]+)')
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
    text_extensions = {'.txt', '.md', '.json', '.yaml', '.yml'}
    
    last_end = 0
    for match in pattern.finditer(query):
        pre_text = query[last_end:match.start()].strip()
        if pre_text:
            content.append(get_text_content_dict(pre_text))

        file_path = match.group(1) or match.group(2)
        _, ext = os.path.splitext(file_path.lower())
        
        if ext in image_extensions:
            content.append(get_image_content_dict(file_path))
        elif ext in text_extensions:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    file_content = f.read()
                content.append(get_text_content_dict(file_content))
            except (FileNotFoundError, PermissionError, OSError) as e:
                content.append(get_text_content_dict(f"[Error reading @{file_path}: {e}]"))
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
        self.messages = list(new_messages)

    def checkpoint(self) -> list[Any]:
        """Return a shallow copy of messages for rollback on interruption."""
        return list(self.messages)

    def rollback(self, snapshot: list[Any]) -> None:
        """Restore messages to a previous checkpoint."""
        self.messages = snapshot


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
    ):

        self.model = model
        self.model_alias = model_alias
        self.profile = resolve_model_profile(model, model_alias)
        self._api_key = api_key
        self._base_url = base_url
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.verbose = verbose
        self.thinking_budget = thinking_budget
        self.context_length = context_length if context_length is not None else self.profile.context_length

        self.console = AgentConsole()

        # built-in system prompt and tools
        self.system_prompt = build_system_prompt(exec_mode=exec_mode, work_dir=work_dir)
        self.tools = get_tool_definitions()

        self.permissions = PermissionController(mode=permission_mode, console=self.console, work_dir=work_dir)

        # subagent manager (only if we have a model_alias for spawning children)
        self.subagent_manager = SubagentManager(model_alias=model_alias, console=self.console) if model_alias else None
        self.interrupt = InterruptHandler()
        self.tool_executor = ToolExecutor(console=self.console, subagent_manager=self.subagent_manager, interrupt_handler=self.interrupt, work_dir=work_dir)

        self.context = ContextManager(self.system_prompt)

        self.round_id = 0

        self.token_totals = {
            "total": 0,
            "prompt": 0,
            "cached_prompt": 0,
            "completion": 0,
            "reasoning": 0,
        }

        self.console.system(f"Created Agent with model: {model} (context: {self.context_length:,} tokens)")
        self.console.system(f"Permission mode: {permission_mode.value}")
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

    def _interruptible_sleep(self, seconds: float):
        """Sleep in short increments so Ctrl+C / interrupts are responsive."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.interrupt.interrupted:
                return
            time.sleep(min(0.5, max(0, deadline - time.monotonic())))

    def call_api(self):
        """Call the API using current context, with automatic retry on any error.

        All errors are retried with exponential backoff up to MAX_RETRIES times.
        """

        # context management: prune old tool results, then compact if needed
        if self.context_length > 0:
            self.context.replace_messages(
                prune_context(self.context.messages, self.context_length)
            )
            if needs_compaction(self.context.messages, self.context_length):
                self.console.system("Context window pressure — compacting via LLM summarization...")
                self.context.replace_messages(
                    compact_context(self.context.messages, self.client, self.model, console=self.console)
                )
                self.console.system("Compaction complete.")

        if self.verbose:
            ctx_chars = estimate_context_chars(self.context.messages)
            ctx_pct = ctx_chars / (self.context_length * CHARS_PER_TOKEN) * 100 if self.context_length else 0
            self.console.debug(
                f"Calling API (round: {self.round_id}, "
                f"context: ~{ctx_chars // CHARS_PER_TOKEN}tok / {self.context_length}tok [{ctx_pct:.0f}%])"
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

        # ---- retry loop with exponential backoff ----
        for attempt in range(self.MAX_RETRIES + 1):
            if self.interrupt.interrupted:
                raise InterruptedError("Interrupted while waiting to call API")

            try:
                response = self.client.chat.completions.create(**kwargs)
                break  # success
            except Exception as e:
                if attempt < self.MAX_RETRIES:
                    wait_time = self.INITIAL_BACKOFF * (2 ** attempt)
                    self.console.system(
                        f"[Retry {attempt + 1}/{self.MAX_RETRIES}] {e} "
                        f"— retrying in {wait_time:.1f}s…"
                    )
                    self._interruptible_sleep(wait_time)
                else:
                    raise RuntimeError(
                        f"Max retries ({self.MAX_RETRIES}) exceeded. Last error: {e}"
                    ) from e

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
                msg = f"Tool call denied: {function_name}"
                if denial_reason:
                    msg += f"\nReason: {denial_reason}"
                result = {"error": msg, "success": False}
            else:
                result = self.tool_executor.execute(function_name, function_args)
            result_text = format_tool_result(result)

            success = result.get("success", False)
            if result.get("streamed"):
                exit_code = result.get("exit_code", "?")
                self.console.tool_result(f"exit code {exit_code}", success=success)
            else:
                self.console.tool_result(format_tool_summary(result_text), success=success)

            # Layer 1: generic truncation relative to context window
            if self.context_length > 0:
                result_text = truncate_tool_result(result_text, self.context_length)

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

        Fatal API errors (auth, quota, bad request, …) are caught, displayed
        to the user, and cause the turn to end gracefully rather than crash.
        """

        while True:
            if self.interrupt.interrupted:
                return None

            snapshot = self.context.checkpoint()

            try:
                message = self.call_api()
            except RuntimeError as e:
                self.context.rollback(snapshot)
                self.console.error(f"API call failed: {e}")
                return None
            except Exception:
                if self.interrupt.interrupted:
                    self.context.rollback(snapshot)
                    self._recreate_client()
                    self.console.system("Interrupted — partial response rolled back.")
                    return None
                raise

            if message.content:
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

    COMMANDS = {"help", "compact", "usage", "exit", "quit", "clear", "perm", "model", "save", "load", "context"}

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
        elif cmd == "model":
            self._cmd_model(raw)
        elif cmd == "save":
            self._cmd_save(raw)
        elif cmd == "load":
            self._cmd_load(raw)
        elif cmd == "context":
            self._cmd_context()

        return False

    def _cmd_help(self):
        self.console.print(
            "[bold blue]Available commands:[/bold blue]\n"
            "\n"
            "  [cyan]/help[/cyan]        — Show this help message\n"
            "  [cyan]/context[/cyan]     — Show a concise one-line-per-message context log\n"
            "  [cyan]/compact[/cyan]     — Force context compaction via LLM summarization\n"
            "  [cyan]/usage[/cyan]       — Show token usage for this session\n"
            "  [cyan]/model[/cyan]       — Show or switch LLM model (/model <name>)\n"
            "  [cyan]/perm[/cyan]        — Show or change permission mode (/perm auto|default|strict)\n"
            "  [cyan]/save[/cyan] [name] — Save session to .kia/sessions/ (default: timestamp)\n"
            "  [cyan]/load[/cyan] [name] — Load a saved session (no name: list available)\n"
            "  [cyan]/clear[/cyan]       — Clear conversation history (keep system prompt)\n"
            "  [cyan]/exit[/cyan]        — Exit the agent (also: /quit)\n"
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
        self.context.replace_messages([])
        self.round_id = 0
        self.console.system("Conversation cleared.")

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

        est_tokens = total_chars // CHARS_PER_TOKEN
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
        self.client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        self.profile = resolve_model_profile(self.model, self.model_alias)
        self.context_length = self.profile.context_length

        if self.subagent_manager:
            self.subagent_manager.model_alias = target

        self.console.system(
            f"Switched to model: {self.model} "
            f"(context: {self.context_length:,} tokens, thinking: {self.profile.thinking or 'none'})"
        )

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
        return True

    def _cmd_save(self, raw: str):
        parts = raw.split(maxsplit=1)
        name = parts[1].strip() if len(parts) > 1 else None
        if not self.context.messages:
            self.console.system("Nothing to save — conversation is empty.")
            return
        path = self.save_session(name)
        self.console.system(f"Session saved to {path} ({len(self.context.messages)} messages)")

    def _cmd_load(self, raw: str):
        parts = raw.split(maxsplit=1)

        if len(parts) < 2 or not parts[1].strip():
            sessions_dir = self._sessions_dir()
            files = sorted(sessions_dir.glob("*.json"))
            if not files:
                self.console.system(f"No saved sessions in {sessions_dir}")
                return
            lines = [f"[bold blue]Saved sessions ({sessions_dir}):[/bold blue]"]
            for f in files:
                stem = f.stem
                try:
                    meta = json.loads(f.read_text(encoding="utf-8"))
                    n_msgs = len(meta.get("messages", []))
                    rnd = meta.get("round_id", "?")
                    model = meta.get("model", "?")
                    lines.append(f"  [cyan]{stem}[/cyan]  — {n_msgs} msgs, round {rnd}, model: {model}")
                except Exception:
                    lines.append(f"  [cyan]{stem}[/cyan]  — (unreadable)")
            self.console.print("\n".join(lines))
            return

        name = parts[1].strip()
        self.load_session(name)

    # ----- main loops -------------------------------------------------------

    def chat_loop(self):

        self.console.system("Type `/help` for available commands.")
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

    def _print_token_summary(self):
        self.console.system(
            f"Total tokens used: {self.token_totals['total']} "
            f"(input: {self.token_totals['prompt']}, cached input: {self.token_totals['cached_prompt']}, "
            f"output: {self.token_totals['completion']}, reasoning: {self.token_totals['reasoning']})"
        )

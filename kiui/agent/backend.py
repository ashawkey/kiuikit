import os
import re
import time
import json
from pathlib import Path
from typing import Literal
from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    BadRequestError,
    UnprocessableEntityError,
    APIStatusError,
)
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

class FatalAPIError(Exception):
    """Unrecoverable API error (e.g., auth failure, quota exhausted)."""


# Keywords in RateLimitError messages that signal quota/billing issues (fatal).
_QUOTA_KEYWORDS = ("quota", "insufficient", "billing", "exceeded your current", "budget")


def _classify_api_error(error: Exception) -> tuple[bool, str]:
    """Classify an API error as retryable or fatal.

    Returns ``(is_retryable, human_readable_message)``.
    """
    # --- clearly retryable (network / transient) ---
    if isinstance(error, APIConnectionError):
        return True, f"Connection error: {error}"
    if isinstance(error, APITimeoutError):
        return True, f"Request timed out: {error}"
    if isinstance(error, InternalServerError):
        return True, f"Server error ({error.status_code}): {error.message}"

    # --- rate limit: retryable unless it's a quota/billing issue ---
    if isinstance(error, RateLimitError):
        msg_lower = str(error).lower()
        if any(kw in msg_lower for kw in _QUOTA_KEYWORDS):
            return False, f"Quota/billing error: {error.message}"
        return True, f"Rate limited: {error.message}"

    # --- clearly fatal ---
    if isinstance(error, AuthenticationError):
        return False, f"Authentication failed (check your API key): {error.message}"
    if isinstance(error, PermissionDeniedError):
        return False, f"Permission denied: {error.message}"
    if isinstance(error, NotFoundError):
        return False, f"Resource not found (check model name): {error.message}"
    if isinstance(error, BadRequestError):
        return False, f"Bad request: {error.message}"
    if isinstance(error, UnprocessableEntityError):
        return False, f"Unprocessable request: {error.message}"

    # --- other APIStatusError subtypes ---
    if isinstance(error, APIStatusError):
        if error.status_code >= 500:
            return True, f"Server error ({error.status_code}): {error.message}"
        return False, f"API error ({error.status_code}): {error.message}"

    # Unknown — assume retryable so we don't silently swallow transient issues
    return True, f"Unexpected error: {error}"


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
    MAX_RETRIES = 10
    INITIAL_BACKOFF = 1.0   # seconds

    def __init__(
        self, 
        model: str,
        api_key: str,
        base_url: str,
        model_key: str = "",
        verbose: bool = True,
        thinking_budget: Literal["low", "medium", "high"] = "low",
        context_window: int | None = None,
        permission_mode: PermissionMode = PermissionMode.DEFAULT,
        exec_mode: bool = False,
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
        self.context_window = context_window if context_window is not None else self.profile.context_window

        self.console = AgentConsole()

        # built-in system prompt and tools
        self.system_prompt = build_system_prompt(exec_mode=exec_mode)
        self.tools = get_tool_definitions()

        self.permissions = PermissionController(mode=permission_mode, console=self.console)

        # subagent manager (only if we have a model_key for spawning children)
        self.subagent_manager = SubagentManager(model_key=model_key, console=self.console) if model_key else None
        self.interrupt = InterruptHandler()
        self.tool_executor = ToolExecutor(console=self.console, subagent_manager=self.subagent_manager, interrupt_handler=self.interrupt)

        self.context = ContextManager(self.system_prompt)

        self.round_id = 0

        self.token_totals = {
            "total": 0,
            "prompt": 0,
            "cached_prompt": 0,
            "completion": 0,
            "reasoning": 0,
        }

        self.console.system(f"Created Agent with model: {model} (context: {self.context_window:,} tokens)")
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
        """Call the API using current context, with automatic retry for transient errors.

        Retryable errors (network, timeout, rate-limit, 5xx) are retried with
        exponential backoff + jitter.  Fatal errors (auth, quota, bad request)
        raise ``FatalAPIError`` immediately.
        """

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

        # ---- retry loop with exponential backoff ----
        for attempt in range(self.MAX_RETRIES + 1):
            if self.interrupt.interrupted:
                raise InterruptedError("Interrupted while waiting to call API")

            try:
                response = self.client.chat.completions.create(**kwargs)
                break  # success
            except Exception as e:
                retryable, err_msg = _classify_api_error(e)

                if not retryable:
                    raise FatalAPIError(err_msg) from e

                if attempt < self.MAX_RETRIES:
                    wait_time = self.INITIAL_BACKOFF * (2 ** attempt)
                    self.console.system(
                        f"[Retry {attempt + 1}/{self.MAX_RETRIES}] {err_msg} "
                        f"— retrying in {wait_time:.1f}s…"
                    )
                    self._interruptible_sleep(wait_time)
                else:
                    raise FatalAPIError(
                        f"Max retries ({self.MAX_RETRIES}) exceeded. Last error: {err_msg}"
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

        Fatal API errors (auth, quota, bad request, …) are caught, displayed
        to the user, and cause the turn to end gracefully rather than crash.
        """

        while True:
            if self.interrupt.interrupted:
                return None

            snapshot = self.context.checkpoint()

            try:
                message = self.call_api()
            except FatalAPIError as e:
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

    COMMANDS = {"help", "compact", "usage", "exit", "quit", "clear", "perm", "save", "load", "context"}

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

    def _cmd_context(self):
        msgs = self.context.messages
        if not msgs:
            self.console.system("Context is empty (no messages).")
            return

        total_chars = 0
        lines = [f"[bold blue]Context log ({len(msgs)} messages):[/bold blue]"]

        # build a map from tool_call_id -> tool function name for annotation
        tc_id_to_name: dict[str, str] = {}
        for m in msgs:
            role = m.get("role", "") if isinstance(m, dict) else getattr(m, "role", "")
            if role == "assistant":
                tool_calls = (m.get("tool_calls") if isinstance(m, dict) else getattr(m, "tool_calls", None)) or []
                for tc in tool_calls:
                    tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                    name = (tc.get("function", {}).get("name", "?") if isinstance(tc, dict)
                            else getattr(getattr(tc, "function", None), "name", "?"))
                    if tc_id:
                        tc_id_to_name[tc_id] = name

        for idx, m in enumerate(msgs):
            role = m.get("role", "") if isinstance(m, dict) else getattr(m, "role", "")
            content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "") or ""

            # extract text
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = " ".join(
                    item.get("text", "") for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            else:
                text = ""

            chars = len(text)
            tool_calls = []
            if role == "assistant":
                tool_calls = (m.get("tool_calls") if isinstance(m, dict) else getattr(m, "tool_calls", None)) or []
                for tc in tool_calls:
                    args = (tc.get("function", {}).get("arguments", "") if isinstance(tc, dict)
                            else getattr(getattr(tc, "function", None), "arguments", "") or "")
                    chars += len(args) if isinstance(args, str) else 0

            total_chars += chars

            # one-line preview (strip newlines, cap length)
            preview = text.replace("\n", " ").strip()
            if len(preview) > 80:
                preview = preview[:77] + "..."

            # format per role
            role_tag = f"[cyan]{role:>9}[/cyan]"
            size_tag = f"[dim]{chars:>6} ch[/dim]"

            if role == "assistant" and tool_calls:
                n_tc = len(tool_calls)
                tc_names = ", ".join(
                    (tc.get("function", {}).get("name", "?") if isinstance(tc, dict)
                     else getattr(getattr(tc, "function", None), "name", "?"))
                    for tc in tool_calls
                )
                extra = f"[yellow]{n_tc} call{'s' if n_tc > 1 else ''}[/yellow] ({tc_names})"
                if preview:
                    lines.append(f"  [dim]#{idx:<3}[/dim] {role_tag} {size_tag}  {extra}  {preview}")
                else:
                    lines.append(f"  [dim]#{idx:<3}[/dim] {role_tag} {size_tag}  {extra}")
            elif role == "tool":
                tc_id = m.get("tool_call_id") if isinstance(m, dict) else getattr(m, "tool_call_id", None)
                tool_name = tc_id_to_name.get(tc_id, "?") if tc_id else "?"
                lines.append(f"  [dim]#{idx:<3}[/dim] {role_tag} {size_tag}  [magenta]({tool_name})[/magenta]  {preview}")
            else:
                lines.append(f"  [dim]#{idx:<3}[/dim] {role_tag} {size_tag}  {preview}")

        est_tokens = total_chars // CHARS_PER_TOKEN
        ctx_pct = est_tokens / self.context_window * 100 if self.context_window else 0
        lines.append(f"\n  [bold]Total:[/bold] ~{est_tokens:,} tokens / {self.context_window:,} [{ctx_pct:.0f}%]")

        self.console.print("\n".join(lines))

    def _cmd_perm(self, raw: str):
        parts = raw.split()
        if len(parts) < 2:
            mode = self.permissions.mode.value
            allowed = ", ".join(sorted(self.permissions._session_allowed)) or "(none)"
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
        self.permissions._session_allowed.clear()
        self.console.system(f"Permission mode changed to: {new_mode.value}")

    # ----- session save / load ------------------------------------------------

    SESSIONS_DIR_NAME = "sessions"

    def _sessions_dir(self) -> Path:
        d = get_kia_dir() / self.SESSIONS_DIR_NAME
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_session(self, name: str | None = None) -> Path:
        """Save the current session state to a JSON file. Returns the file path."""
        if not name:
            name = time.strftime("%Y%m%d_%H%M%S")

        data = {
            "model": self.model,
            "round_id": self.round_id,
            "token_totals": self.token_totals,
            "system_prompt": self.context.system_prompt,
            "messages": self.context.messages,
        }

        path = self._sessions_dir() / f"{name}.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
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

    def _print_token_summary(self):
        self.console.system(
            f"Total tokens used: {self.token_totals['total']} "
            f"(input: {self.token_totals['prompt']}, cached input: {self.token_totals['cached_prompt']}, "
            f"output: {self.token_totals['completion']}, reasoning: {self.token_totals['reasoning']})"
        )

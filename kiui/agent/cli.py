"""CLI entry point for kia: terminal-based AI agent.

Usage:
    kia [--model MODEL] [--verbose] [--perm auto|default|strict]
        [--resume [SESSION_ID]]
        [--list | --storage | --clean]
"""

import json
import os
import socket
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TYPE_CHECKING

if TYPE_CHECKING:
    from kiui.agent.hubclient import HubClient

import tyro
from rich.table import Table

from kiui.config import conf, LOCAL_CONFIG_PATH
from kiui.agent.ui import AgentConsole
from kiui.agent.backend import LLMAgent
from kiui.agent.permissions import PermissionMode
from kiui.agent.models import ReasoningEffort, resolve_model_profile


# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------

@dataclass
class Args:
    """Terminal-based AI agent with tool-use, web access, and shell execution."""
    model: str = ""
    persona: str = ""  # persona to run as (see /persona; default: coder)
    verbose: bool = False
    stream: bool = True  # stream the response token-by-token as it is generated
    reasoning_effort: ReasoningEffort | None = None  # defaults to model config, then high

    perm: PermissionMode = PermissionMode.AUTO
    resume: str | None = None  # --resume [session_id]
    list: Annotated[bool, tyro.conf.FlagCreatePairsOff] = False  # --list: show available models and exit
    storage: Annotated[bool, tyro.conf.FlagCreatePairsOff] = False  # show project .kia usage and exit
    clean: Annotated[bool, tyro.conf.FlagCreatePairsOff] = False  # remove generated project .kia data and exit
    hub: Annotated[bool, tyro.conf.FlagCreatePairsOff] = False  # --hub: run the shared web hub daemon
    web_port: int = 8765


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_agent(args: Args) -> "tuple[LLMAgent | None, HubClient | None]":
    """Create an LLMAgent (and its optional hub link) from parsed arguments.

    Returns ``(None, None)`` if the model is not found. If a reachable hub is
    running, the agent links to it automatically; otherwise it runs terminal-only.
    """
    console = AgentConsole()
    openai_conf = conf.get("openai", {})

    if not args.model:
        if not openai_conf:
            console.error(f"No models found in config: {LOCAL_CONFIG_PATH}")
            return None, None
        args.model = next(iter(openai_conf))

    if args.model not in openai_conf:
        console.error(f"Model '{args.model}' not found in config: {LOCAL_CONFIG_PATH}")
        console.print("Available models:", list(openai_conf.keys()))
        return None, None

    model_conf = openai_conf[args.model]
    events = inputs = prompts = cancellation = hub_client = None

    from kiui.agent.hub import discover_hub

    info = discover_hub(args.web_port)
    if info:
        from kiui.agent.io import (
            CancellationToken, EventHub, InputBroker, PromptBroker,
        )
        from kiui.agent.hubclient import HubClient

        events = EventHub()
        inputs = InputBroker(events)
        prompts = PromptBroker(events)
        cancellation = CancellationToken(events, prompts)
        console = AgentConsole(events=events)

        cwd = os.getcwd()
        meta = {
            "title": f"{Path(cwd).name} · {args.model}",
            "cwd": cwd,
            "model": args.model,
            "pid": os.getpid(),
            "host": socket.gethostname(),
        }
        hub_client = HubClient(
            events,
            inputs,
            prompts,
            cancellation,
            host=info.get("host", "127.0.0.1"),
            port=int(info.get("port", args.web_port)),
            token=info.get("token", ""),
            session_id=uuid.uuid4().hex,
            meta=meta,
        )

    agent = LLMAgent(
        model=model_conf.get("model", args.model),
        api_key=model_conf.get("api_key", ""),
        base_url=model_conf.get("base_url", ""),
        model_alias=args.model,
        verbose=args.verbose,
        stream=args.stream,
        reasoning_effort=args.reasoning_effort or model_conf.get("reasoning_effort", "high"),

        permission_mode=args.perm,
        persona=args.persona,
        console=console,
        events=events,
        input_broker=inputs,
        prompt_broker=prompts,
        cancellation=cancellation,
    )
    return agent, hub_client


def _last_user_preview(messages: list) -> str:
    """Extract a short preview from the last user message."""
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user":
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


def _pick_session(console: AgentConsole) -> str | None:
    """List saved sessions and let the user pick one interactively."""
    from kiui.agent.utils import get_kia_dir

    sessions_dir = get_kia_dir() / "sessions"
    if not sessions_dir.exists():
        console.error(f"No sessions directory found: {sessions_dir}")
        return None

    files = sorted(sessions_dir.glob("*.json"), reverse=True)
    if not files:
        console.system(f"No saved sessions in {sessions_dir}")
        return None

    entries: list[str] = []
    choice_labels: list[str] = []
    for f in files:
        stem = f.stem
        try:
            meta = json.loads(f.read_text(encoding="utf-8"))
            messages = meta.get("messages", [])
            n_msgs = len(messages)
            rnd = meta.get("round_id", "?")
            model = meta.get("model", "?")
            preview = _last_user_preview(messages)
        except Exception:
            n_msgs, rnd, model, preview = "?", "?", "?", ""
        entries.append(stem)
        label = f"{stem}  │  msgs:{n_msgs}  rounds:{rnd}  model:{model}"
        if preview:
            label += f"  │  {preview}"
        choice_labels.append(label)

    picked = console.select(
        message="Pick a session to resume",
        choices=choice_labels,
    )
    if picked is None:
        return None

    for i, label in enumerate(choice_labels):
        if label == picked:
            return entries[i]

    console.error("Invalid selection.")
    return None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_list():
    console = AgentConsole()
    openai_conf = conf.get("openai", {})

    if not openai_conf:
        exists = LOCAL_CONFIG_PATH.exists()
        console.error(f"No models found in config: {LOCAL_CONFIG_PATH}")
        if not exists:
            console.print(
                f"Config file does not exist. Create it at:\n  {LOCAL_CONFIG_PATH}\n\n"
                "Example:\n"
                "openai:\n"
                "  my-model:\n"
                "    model: gpt-4o\n"
                "    api_key: sk-...\n"
                "    base_url: https://api.openai.com/v1"
            )
        else:
            console.print("Please add model configurations under the 'openai' key.")
        return

    table = Table(title="Available Models", show_header=True, header_style="bold magenta")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Model", style="green")
    table.add_column("Base URL", style="yellow")
    table.add_column("Context", style="blue", justify="right")
    table.add_column("Thinking", style="magenta")

    for name, model_conf in openai_conf.items():
        model_id = model_conf.get("model", name)
        profile = resolve_model_profile(model_id, name)
        ctx = f"{profile.context_length // 1000}K"
        table.add_row(
            name,
            model_id,
            model_conf.get("base_url", "N/A"),
            ctx,
            f"{profile.reasoning or '-'} / {model_conf.get('reasoning_effort', 'high')}",
        )

    console.table(table)


def cmd_storage():
    """Show allocated disk usage for each entry in the project .kia directory."""
    from kiui.agent.storage import format_size, kia_storage_dir, storage_entries

    console = AgentConsole()
    root = kia_storage_dir()
    entries = storage_entries()
    if not entries:
        console.system(f"No storage found in {root}")
        return

    table = Table(title=f"kia storage: {root}")
    table.add_column("Entry", style="cyan")
    table.add_column("Type", style="dim")
    table.add_column("Size", justify="right", style="green")
    for entry in entries:
        table.add_row(entry.name, "directory" if entry.is_dir else "file", format_size(entry.size))
    table.add_section()
    table.add_row("Total", "", format_size(sum(entry.size for entry in entries)), style="bold")
    console.table(table)


def cmd_clean():
    """Remove generated sessions, tool results, and command history."""
    from kiui.agent.storage import clean_storage, cleanable_entries, format_size, kia_storage_dir

    console = AgentConsole()
    entries = cleanable_entries()
    if not entries:
        console.system(f"Nothing to clean in {kia_storage_dir()}")
        return

    removed = clean_storage()
    console.system(f"Cleaned {format_size(removed)} from {kia_storage_dir()}")


def cmd_hub(args: Args):
    """Run the shared web hub daemon (owns the public port)."""
    console = AgentConsole()
    from kiui.agent.hub import Hub

    try:
        hub = Hub(port=args.web_port, token=conf.get("kia_web_token"), console=console)
        hub.start()
    except Exception as exc:
        console.error(f"Could not start hub: {exc}")
        return

    console.system(f"kia hub running at {hub.url}")
    console.local(f"[bold yellow]Web access token:[/bold yellow] {hub.token}")
    console.system("Agents started with `kia` will auto-link while this hub is running.")
    console.system("Press Ctrl+C to stop the hub.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        hub.stop()
        console.system("Hub stopped.")


def cmd_chat(args: Args):
    agent, hub_client = get_agent(args)
    if not agent:
        return

    if hub_client is not None:
        hub_client.start()
        agent.console.system(
            f"Linked to kia hub at {hub_client.host}:{hub_client.port} "
            f"(session {hub_client.session_id[:8]})"
        )

    try:
        # Handle --resume
        session_id: str | None = args.resume
        if session_id == "":  # bare --resume → pick interactively
            session_id = _pick_session(agent.console)

        if session_id:
            agent.load_session(session_id)

        agent.chat_loop(resumed_session_id=session_id)
    finally:
        if hub_client is not None:
            hub_client.stop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    raw_args = sys.argv[1:]

    # Pre-parse --resume with an optional value, so bare
    # `--resume` works as well as `--resume <session_id>`.
    cleaned: list[str] = []
    resume_value: str | None = None
    i = 0
    while i < len(raw_args):
        if raw_args[i] == "--resume":
            if i + 1 < len(raw_args) and not raw_args[i + 1].startswith("-"):
                resume_value = raw_args[i + 1]
                i += 2
            else:
                resume_value = ""  # sentinel for bare --resume
                i += 1
        else:
            cleaned.append(raw_args[i])
            i += 1

    # Replace sys.argv so tyro sees the cleaned version
    sys.argv = [sys.argv[0]] + cleaned

    args = tyro.cli(Args, description="Terminal AI agent with auto-detected synchronized Web UI")

    # Patch in the pre-parsed resume value (tyro can't handle optional-value flags natively)
    if resume_value is not None:
        args.resume = resume_value

    commands = (args.list, args.storage, args.clean, args.hub)
    if sum(commands) > 1:
        raise SystemExit("Choose only one of --list, --storage, --clean, or --hub")

    if args.list:
        cmd_list()
    elif args.storage:
        cmd_storage()
    elif args.clean:
        cmd_clean()
    elif args.hub:
        cmd_hub(args)
    else:
        cmd_chat(args)


if __name__ == "__main__":
    main()

"""CLI entry point for kia: terminal-based AI agent.

Usage:
    kia [--model MODEL] [--verbose] [--perm auto|default|strict]
        [--resume [SESSION_ID]]
        [--list]
"""

import json
import sys
from dataclasses import dataclass
from typing import Annotated

import tyro
from rich.table import Table

from kiui.config import conf, LOCAL_CONFIG_PATH
from kiui.agent.ui import AgentConsole
from kiui.agent.backend import LLMAgent
from kiui.agent.permissions import PermissionMode
from kiui.agent.models import resolve_model_profile


# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------

@dataclass
class Args:
    """Terminal-based AI agent with tool-use, web access, and shell execution."""
    model: str = ""
    verbose: bool = False
    perm: PermissionMode = PermissionMode.DEFAULT
    resume: str | None = None  # --resume [session_id]
    list: Annotated[bool, tyro.conf.FlagCreatePairsOff] = False  # --list: show available models and exit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_agent(args: Args) -> LLMAgent | None:
    """Create an LLMAgent from parsed arguments. Returns None if model not found."""
    console = AgentConsole()
    openai_conf = conf.get("openai", {})

    if not args.model:
        if not openai_conf:
            console.error(f"No models found in config: {LOCAL_CONFIG_PATH}")
            return None
        args.model = next(iter(openai_conf))

    if args.model not in openai_conf:
        console.error(f"Model '{args.model}' not found in config: {LOCAL_CONFIG_PATH}")
        console.print("Available models:", list(openai_conf.keys()))
        return None

    model_conf = openai_conf[args.model]

    return LLMAgent(
        model=model_conf.get("model", args.model),
        api_key=model_conf.get("api_key", ""),
        base_url=model_conf.get("base_url", ""),
        model_alias=args.model,
        verbose=args.verbose,
        permission_mode=args.perm,
    )


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
            profile.thinking or "-",
        )

    console.table(table)


def cmd_chat(args: Args):
    agent = get_agent(args)
    if not agent:
        return

    # Handle --resume
    session_id: str | None = args.resume
    if session_id == "":  # bare --resume → pick interactively
        session_id = _pick_session(AgentConsole())

    if session_id:
        agent.load_session(session_id)

    agent.chat_loop(resumed_session_id=session_id)


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

    args = tyro.cli(Args, description="Terminal-based AI agent")

    # Patch in the pre-parsed resume value (tyro can't handle optional-value flags natively)
    if resume_value is not None:
        args.resume = resume_value

    if args.list:
        cmd_list()
    else:
        cmd_chat(args)


if __name__ == "__main__":
    main()

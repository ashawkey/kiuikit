import json
import sys
from dataclasses import dataclass
from typing import Annotated, Union

import tyro

from kiui.config import conf, LOCAL_CONFIG_PATH
from kiui.agent.ui import AgentConsole
from kiui.agent.backend import LLMAgent
from kiui.agent.permissions import PermissionMode


# ---------------------------------------------------------------------------
# Subcommand configs
# ---------------------------------------------------------------------------

@dataclass
class ListCmd:
    """List all available models from the config file (~/.kiui.yaml)."""
    pass


@dataclass
class ChatCmd:
    """Start interactive chat mode with the specified model."""
    model: str = ""
    verbose: bool = False
    context_length: int | None = None
    permission_mode: PermissionMode = PermissionMode.DEFAULT
    resume: str | None = None  # --resume <session_id>


@dataclass
class ExecCmd:
    """Execute a single query and return the result."""
    prompt: str
    model: str = ""
    verbose: bool = False
    context_length: int | None = None
    permission_mode: PermissionMode = PermissionMode.AUTO


Command = Union[
    Annotated[ListCmd, tyro.conf.subcommand("list")],
    Annotated[ChatCmd, tyro.conf.subcommand("chat")],
    Annotated[ExecCmd, tyro.conf.subcommand("exec")],
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_agent(cfg: ChatCmd | ExecCmd, exec_mode: bool = False) -> LLMAgent | None:
    """Create an LLMAgent from a config dataclass. Returns None if model not found."""
    console = AgentConsole()
    openai_conf = conf.get("openai", {})

    if not cfg.model:
        if not openai_conf:
            console.error(f"No models found in config: {LOCAL_CONFIG_PATH}")
            return None
        cfg.model = next(iter(openai_conf))

    if cfg.model not in openai_conf:
        console.error(f"Model '{cfg.model}' not found in config: {LOCAL_CONFIG_PATH}")
        console.print("Available models:", list(openai_conf.keys()))
        return None

    model_conf = openai_conf[cfg.model]

    agent = LLMAgent(
        model=model_conf.get("model", cfg.model),
        api_key=model_conf.get("api_key", ""),
        base_url=model_conf.get("base_url", ""),
        model_alias=cfg.model,
        verbose=cfg.verbose,
        context_length=cfg.context_length,
        permission_mode=cfg.permission_mode,
        exec_mode=exec_mode,
    )

    return agent


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
    """List saved sessions and let the user pick one interactively using questionary."""
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
    for i, f in enumerate(files, 1):
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
        # Build a compact label for the questionary picker
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

    # Find which entry was chosen by matching the label
    for i, label in enumerate(choice_labels):
        if label == picked:
            return entries[i]

    console.error("Invalid selection.")
    return None


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_list(cfg: ListCmd):
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


def cmd_chat(cfg: ChatCmd):
    agent = get_agent(cfg)
    if not agent:
        return

    # Handle --resume
    session_id: str | None = cfg.resume
    if session_id == "":  # bare --resume with no value → pick interactively
        session_id = _pick_session(AgentConsole())

    if session_id:
        agent.load_session(session_id)

    agent.chat_loop(resumed_session_id=session_id)


def cmd_exec(cfg: ExecCmd):
    if agent := get_agent(cfg, exec_mode=True):
        agent.execute(cfg.prompt)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]

    # Pre-parse --resume with an optional value before tyro, so bare
    # `--resume` works as well as `--resume <session_id>`.
    cleaned: list[str] = []
    resume_value: str | None = None
    i = 0
    while i < len(args):
        if args[i] == "--resume":
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                resume_value = args[i + 1]
                i += 2
            else:
                resume_value = ""  # sentinel for bare --resume
                i += 1
        else:
            cleaned.append(args[i])
            i += 1

    # Replace sys.argv so tyro sees the cleaned version
    sys.argv = [sys.argv[0]] + cleaned

    if not cleaned:
        # No subcommand given — default to chat
        cmd = ChatCmd()
        if resume_value is not None:
            cmd.resume = resume_value
        cmd_chat(cmd)
    else:
        cmd = tyro.cli(Command, description="LLM Agent CLI")
        if isinstance(cmd, ChatCmd) and resume_value is not None:
            cmd.resume = resume_value

        if isinstance(cmd, ListCmd):
            cmd_list(cmd)
        elif isinstance(cmd, ChatCmd):
            cmd_chat(cmd)
        elif isinstance(cmd, ExecCmd):
            cmd_exec(cmd)


if __name__ == "__main__":
    main()

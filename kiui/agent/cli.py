import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Union

import tyro
from rich.table import Table

from kiui.agent.ui import AgentConsole
from kiui.config import conf, CONFIG_PATH
from kiui.agent.backend import LLMAgent
from kiui.agent.models import resolve_model_profile
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
    model: str
    verbose: bool = False
    context_window: int | None = None
    permission_mode: PermissionMode = PermissionMode.DEFAULT


@dataclass
class ExecCmd:
    """Execute a single query and return the result."""
    model: str
    prompt: str
    verbose: bool = False
    context_window: int | None = None
    permission_mode: PermissionMode = PermissionMode.AUTO
    result_file: str | None = None


@dataclass
class PipeCmd:
    """Run in pipe mode for sub-agent communication."""
    model: str
    verbose: bool = False
    context_window: int | None = None
    permission_mode: PermissionMode = PermissionMode.AUTO


Command = Union[
    Annotated[ListCmd, tyro.conf.subcommand("list")],
    Annotated[ChatCmd, tyro.conf.subcommand("chat")],
    Annotated[ExecCmd, tyro.conf.subcommand("exec")],
    Annotated[PipeCmd, tyro.conf.subcommand("pipe")],
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_agent(cfg: ChatCmd | ExecCmd | PipeCmd, pipe_mode: bool = False) -> LLMAgent | None:
    """Create an LLMAgent from a config dataclass. Returns None if model not found."""
    console = AgentConsole()
    openai_conf = conf.get("openai", {})

    if cfg.model not in openai_conf:
        console.error(f"Model '{cfg.model}' not found in config: {CONFIG_PATH}")
        console.print("Available models:", list(openai_conf.keys()))
        return None

    model_conf = openai_conf[cfg.model]

    agent = LLMAgent(
        model=model_conf.get("model", cfg.model),
        api_key=model_conf.get("api_key", ""),
        base_url=model_conf.get("base_url", ""),
        model_key=cfg.model,
        verbose=cfg.verbose,
        pipe_mode=pipe_mode,
        context_window=cfg.context_window,
        permission_mode=cfg.permission_mode,
    )

    return agent


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_list(cfg: ListCmd):
    console = AgentConsole()
    openai_conf = conf.get("openai", {})

    if not openai_conf:
        exists = CONFIG_PATH.exists()
        console.error(f"No models found in config: {CONFIG_PATH}")
        if not exists:
            console.print(
                f"Config file does not exist. Create it at:\n  {CONFIG_PATH}\n\n"
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
        ctx = f"{profile.context_window // 1000}K"
        table.add_row(
            name,
            model_id,
            model_conf.get("base_url", "N/A"),
            ctx,
            profile.thinking or "-",
        )

    console.table(table)


def cmd_chat(cfg: ChatCmd):
    if agent := get_agent(cfg):
        agent.chat_loop()


def cmd_exec(cfg: ExecCmd):
    if agent := get_agent(cfg):
        response = agent.execute(cfg.prompt)

        if cfg.result_file and response:
            result = {
                "summary": response[:2000],
                "response": response,
            }
            result_path = Path(cfg.result_file)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(result, indent=2))


def cmd_pipe(cfg: PipeCmd):
    if agent := get_agent(cfg, pipe_mode=True):
        agent.run_pipe_mode()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cmd = tyro.cli(Command, description="LLM Agent CLI")
    if isinstance(cmd, ListCmd):
        cmd_list(cmd)
    elif isinstance(cmd, ChatCmd):
        cmd_chat(cmd)
    elif isinstance(cmd, ExecCmd):
        cmd_exec(cmd)
    elif isinstance(cmd, PipeCmd):
        cmd_pipe(cmd)


if __name__ == "__main__":
    main()

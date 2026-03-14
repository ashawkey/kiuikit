"""Centralized system prompt builder for kiui agent."""

import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

from kiui.agent.tools import get_tool_definitions

_TOOL_TIPS = {
    "read_file": "Output truncated to 2000 lines / 50KB. Use offset/limit for large files.",
    "write_file": "Creates parent directories automatically.",
    "edit_file": "old_text must match exactly including whitespace.",
    "exec_command": "Output capped at 50KB. Use timeout for long-running commands.",
    "glob_files": "Faster and safer than exec_command with find. Max 500 results.",
    "grep_files": "Returns up to 200 matches with file path and line number.",
    "web_fetch": "Content capped at 20K chars.",
    "web_search": "Search the web for real-time information.",
    "remove_file": "Remove a file or directory.",
    "spawn_subagent": "mode='run' for background tasks, mode='session' for persistent sessions.",
    "list_subagents": "Shows status of all active and completed sub-agents.",
    "kill_subagent": "Terminate a running sub-agent by run ID.",
    "send_to_subagent": "Send a message to a persistent session and get a response.",
}


def build_system_prompt() -> str:
    """Build the complete system prompt from ordered sections."""
    sections = []

    # 1. Core identity
    sections.append(
        "You are a terminal-based AI coding agent. "
        "Be helpful, accurate, and concise. "
        "Prioritize correctness, then clarity, then brevity."
    )

    # 2. Tool call style
    sections.append("""## Tool Call Style
- Do not narrate routine, low-risk tool calls — just call the tool.
- Narrate only for multi-step work, complex problems, or sensitive actions (e.g., deletions).
- Keep narration brief and value-dense.""")

    # 3. Safety
    sections.append("""## Safety
- Prioritize safety and human oversight over task completion.
- Do not run destructive commands without asking first.
- Confirm before: deleting files, sending emails, anything irreversible.
- When in doubt, ask.""")

    # 4. Available tools
    sections.append(_build_tools_section())

    # 5. Task execution
    sections.append("""## Task Execution
- Keep going until the task is completely resolved before yielding back to the user.
- Fix problems at the root cause rather than applying surface-level patches.
- Avoid unneeded complexity. Keep changes minimal and consistent with existing style.
- Do not attempt to fix unrelated bugs or broken tests.
- Use `exec_command` with `git log` / `git blame` to search history if additional context is needed.
- Do not add inline comments unless explicitly requested.""")

    # 6. Sub-agents
    sections.append("""## Sub-Agents
You can spawn sub-agents to handle tasks in parallel or delegate work:
- **spawn_subagent** with mode='run': fires a one-shot background task. The result is delivered automatically when it completes.
- **spawn_subagent** with mode='session': starts a persistent session. Use **send_to_subagent** to send follow-up messages and get responses.
- **list_subagents**: check status of all sub-agents.
- **kill_subagent**: terminate a running sub-agent.
Use sub-agents for independent, parallelizable work (e.g., researching one topic while editing another).""")

    # 7. Project instructions (optional AGENTS.md)
    project = _build_project_section()
    if project:
        sections.append(project)

    # 8. Context
    sections.append(_build_context_section())

    return "\n\n".join(sections)


def _build_tools_section() -> str:
    """Build tools section listing all available tools with tips."""
    tools = get_tool_definitions()
    tool_lines = []
    for tool in tools:
        func = tool["function"]
        name = func["name"]
        description = func["description"]
        tip = _TOOL_TIPS.get(name, "")
        if tip:
            tool_lines.append(f"- **{name}**: {description} — {tip}")
        else:
            tool_lines.append(f"- **{name}**: {description}")

    return "## Available Tools\n" + "\n".join(tool_lines) + \
        "\n\nUse tools by making function calls. Always check tool results before proceeding."


def _build_project_section() -> str:
    """Include AGENTS.md project instructions if present."""
    agents_file = Path.cwd() / "AGENTS.md"
    if not agents_file.exists():
        return ""
    try:
        content = agents_file.read_text(encoding="utf-8")
        return f"## Project Instructions\n{content}"
    except Exception:
        return ""


def _build_context_section() -> str:
    """Build context section with current environment information."""
    return f"""## Current Context
- Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Working Directory: {Path.cwd()}
- Operating System: {platform.system()} {platform.release()}
- Python: {platform.python_version()}
- User: {os.getenv("USER") or os.getenv("USERNAME", "unknown")}"""

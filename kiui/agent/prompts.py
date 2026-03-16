"""Centralized system prompt builder for kiui agent."""

import os
import platform
from datetime import datetime
from pathlib import Path


def build_system_prompt(exec_mode: bool = False, work_dir: str | None = None) -> str:
    """Build the complete system prompt from ordered sections.

    If *exec_mode* is True the prompt tells the agent it is running
    autonomously with no user to interact with.
    *work_dir* overrides the working directory shown in the context section.
    """
    sections = []

    # 1. Core identity
    sections.append(
        "You are a terminal-based AI agent. "
        "Be helpful, accurate, and concise. "
        "Prioritize correctness, then clarity, then brevity."
    )

    # 1b. Exec / sub-agent mode
    if exec_mode:
        sections.append("""## Autonomous Mode
You are running as an autonomous sub-agent. There is NO user to interact with.
- Do NOT ask questions or request confirmation — no one will respond.
- Make reasonable decisions on your own and proceed to completion.
- If something is ambiguous, choose the most likely interpretation and move on.
- Finish the task fully, then output a concise summary of what you did.""")

    # 2. Tool call style
    sections.append("""## Tool Call Style
- Do not narrate routine, low-risk tool calls — just call the tool.
- Narrate only for multi-step work, complex problems, or sensitive actions (e.g., deletions).
- Keep narration brief and value-dense.""")

    # 3. Safety
    if not exec_mode:
        sections.append("""## Safety
- Prioritize safety and human oversight over task completion.
- Do not run destructive commands without asking first.
- Confirm before: deleting files, sending emails, anything irreversible.
- When in doubt, ask.""")
    else:
        sections.append("""## Safety
- Avoid destructive or irreversible actions unless the task explicitly requires them.
- Prefer safe, reversible operations.""")

    # 4. Tool usage guidance
    sections.append("""## Tool Usage
- Always check tool results before proceeding.
- Prefer glob_files / grep_files over exec_command for file discovery and search.""")

    # 5. Task execution
    sections.append("""## Task Execution
- Keep going until the task is completely resolved before yielding back to the user.
- Fix problems at the root cause rather than applying surface-level patches.
- Avoid unneeded complexity. Keep changes minimal and consistent with existing style.
- Do not attempt to fix unrelated bugs or broken tests.""")

    # 6. Sub-agents
    sections.append("""## Sub-Agents
You can spawn a sub-agent to delegate work:
- **spawn_subagent**: runs a task in a separate process and returns the result when done.
Use sub-agents when you want to delegate a self-contained task (e.g., research, summarization).""")

    # 7. Project instructions (optional AGENTS.md)
    project = _build_project_section(work_dir)
    if project:
        sections.append(project)

    # 8. Context
    sections.append(_build_context_section(work_dir))

    return "\n\n".join(sections)


def _build_project_section(work_dir: str | None = None) -> str:
    """Include AGENTS.md project instructions if present."""
    base = Path(work_dir) if work_dir else Path.cwd()
    agents_file = base / "AGENTS.md"
    if not agents_file.exists():
        return ""
    try:
        content = agents_file.read_text(encoding="utf-8")
        return f"## Project Instructions\n{content}"
    except Exception:
        return ""


def _build_context_section(work_dir: str | None = None) -> str:
    """Build context section with current environment information."""
    cwd = work_dir or str(Path.cwd())
    return f"""## Current Context
- Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Working Directory: {cwd}
- Operating System: {platform.system()} {platform.release()}
- Python: {platform.python_version()}
- User: {os.getenv("USER") or os.getenv("USERNAME", "unknown")}"""

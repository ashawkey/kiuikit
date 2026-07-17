"""Centralized system prompt builder for kiui agent."""

import os
import platform
import socket
import subprocess
from datetime import datetime
from pathlib import Path

from kiui.agent.skills import discover_skills, build_skills_prompt_section


def build_system_prompt(exec_mode: bool = False, is_subagent: bool = False, work_dir: str | None = None, skills: dict | None = None) -> str:
    """Build the complete system prompt from ordered sections.

    If *exec_mode* is True the prompt tells the agent it is running
    autonomously with no user to interact with.
    *work_dir* overrides the working directory shown in the context section.
    *skills* is a pre-discovered skills dict; if None, skills are discovered
    automatically from *work_dir*.
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
You are running as an autonomous sub-agent with no user available to respond.
- Do not ask questions or request confirmation.
- Infer intent from the task and available context; choose the safest reasonable interpretation.
- If blocked, report the blocker instead of using a risky workaround.
- Complete and verify the task, then return a concise summary.""")

    # 2. Safety
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

    # 3. Tool usage guidance
    sections.append("""## Tool Usage
- Always check tool results before proceeding.
- Do not narrate routine, low-risk tool calls — just call the tool. Narrate only for multi-step work, complex problems, or sensitive actions (e.g., deletions).
- Prefer ls / glob_files / grep_files over exec_command for file discovery and search.
- Keep output focused with narrow paths/patterns, read_file offset/limit, and quiet or filtered commands.
- If output is compacted, follow its recovery guidance instead of repeating the same broad call.""")

    # 4. Task execution
    sections.append("""## Task Execution
- Inspect the relevant context before acting; do not guess about code or file contents.
- Keep going until the request is resolved or a concrete blocker is identified.
- Fix root causes rather than symptoms.
- Keep changes minimal and consistent with existing style. Preserve user changes.
- Do not fix unrelated issues or already broken tests.""")

    # 5. Working style
    sections.append("""## Working Style
- Prefer the smallest clear solution that fully satisfies the request.
- Reuse existing code and standard tools before adding abstractions or dependencies.
- Avoid speculative safeguards, fallbacks, configuration, and extensibility.
- Keep responses concise, but preserve necessary technical detail.
- Verify with the smallest relevant check and report only what was actually verified.""")

    # 6. Sub-agents (top-level agents only; sub-agents cannot spawn children)
    if not is_subagent:
        sections.append("""## Sub-Agents
**spawn_subagent** runs a self-contained task and returns its result.
Delegate only when it materially helps with independent research or analysis. Give a focused task, do not delegate simple work.""")

    # 7. Skills
    if skills is None:
        skills = discover_skills(work_dir)
    skills_section = build_skills_prompt_section(skills)
    if skills_section:
        sections.append(skills_section)

    # 8. Project instructions (optional AGENTS.md)
    project = _build_project_section(work_dir)
    if project:
        sections.append(project)

    # 9. Project memory (agent-generated, loaded from .kia/memory.md)
    memory = _build_memory_section(work_dir)
    if memory:
        sections.append(memory)

    # 10. Context
    sections.append(_build_context_section(work_dir))

    return "\n\n".join(sections)


def _build_project_section(work_dir: str | None = None) -> str:
    """Include AGENTS.md project instructions if present."""
    base = Path(work_dir) if work_dir else Path.cwd()
    instr_file = base / "AGENTS.md"
    if not instr_file.exists():
        return ""

    try:
        content = instr_file.read_text(encoding="utf-8").strip()
    except Exception:
        return ""

    if not content:
        return ""
    return "## Project Instructions\n" + content


def _build_memory_section(work_dir: str | None = None) -> str:
    """Build the project memory section for the system prompt.

    Reads the memory index from .kia/memory/MEMORY.md (hierarchical storage).
    Each entry is a markdown link [title](file.md) followed by a summary.
    The agent can read_file individual memory files under .kia/memory/ for details.
    """
    base = Path(work_dir) if work_dir else Path.cwd()
    memory_dir = base / ".kia" / "memory"
    index_file = memory_dir / "MEMORY.md"

    # Load existing memory index entries
    entries: list[str] = []
    if index_file.exists():
        try:
            raw = index_file.read_text(encoding="utf-8").strip()
            if raw:
                entries = [line.strip() for line in raw.splitlines() if line.strip()]
        except Exception:
            pass

    if entries:
        formatted = "\n".join(f"- {e}" for e in entries)
        return (
            "## Project Memory\n"
            "The following memories were saved in previous sessions. Each entry links to a detailed "
            f"memory file under `.kia/memory/`. Use **read_file** to retrieve the full details of any entry when needed.\n\n"
            f"{formatted}\n\n"
            "Only use **save_memory** for genuinely important insights (e.g., a critical project-wide "
            "convention, a non-obvious architectural rule, or a recurring pitfall to avoid), "
            "or when the user explicitly asks you to remember something. "
            "Do NOT save trivial, obvious, or one-off observations."
        )
    else:
        return (
            "## Project Memory\n"
            "Memories are stored hierarchically under `.kia/memory/`: `MEMORY.md` is the index, "
            "and individual `.md` files hold the full details. "
            "Only use **save_memory** for genuinely important insights (e.g., a critical project-wide "
            "convention, a non-obvious architectural rule, or a recurring pitfall to avoid), "
            "or when the user explicitly asks you to remember something. "
            "Do NOT save trivial, obvious, or one-off observations."
        )


def _build_context_section(work_dir: str | None = None) -> str:
    """Build context section with current environment information."""
    cwd = work_dir or str(Path.cwd())
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    git_info = _get_git_context(cwd)
    return f"""## Current Context
- Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Working Directory: {cwd}
- Operating System: {platform.system()} {platform.release()}
- Python: {platform.python_version()}
- User: {os.getenv("USER") or os.getenv("USERNAME", "unknown")}
- Host: {hostname}
- IP: {ip}{git_info}"""


def _get_git_context(cwd: str) -> str:
    """Return git context lines if inside a git repo, otherwise empty string."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, cwd=cwd, timeout=5,
        )
        if result.returncode != 0:
            return ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""

    branch = _git_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd)
    lines = []
    if branch:
        lines.append(f"- Git Branch: {branch}")
    remote = _git_cmd(["git", "remote", "get-url", "origin"], cwd)
    if remote:
        lines.append(f"- Git Remote: {remote}")
    status = _git_cmd(["git", "status", "--porcelain"], cwd)
    dirty = "dirty" if status else "clean"
    lines.append(f"- Git Status: {dirty}")
    return "\n" + "\n".join(lines)


def _git_cmd(args: list, cwd: str) -> str | None:
    """Run a git command and return stripped stdout, or None on failure."""
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, cwd=cwd, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except subprocess.TimeoutExpired:
        pass
    return None

"""Shared context and prompt-building helpers for agent personas."""

import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PersonaContext:
    """Runtime information passed to a persona's build_system_prompt()."""

    exec_mode: bool = False       # autonomous sub-agent run (no user available)
    is_subagent: bool = False     # sub-agents cannot spawn further sub-agents
    work_dir: str | None = None   # overrides cwd shown in the context section
    skills: dict | None = None    # pre-discovered skills registry


EXEC_MODE_SECTION = """## Autonomous Mode
You are running as an autonomous sub-agent with no user available to respond.
- Do not ask questions or request confirmation.
- Infer intent from the task and available context; choose the safest reasonable interpretation.
- If blocked, report the blocker instead of using a risky workaround.
- Complete and verify the task, then return a concise summary."""

SAFETY_SECTION = """## Safety
- Follow explicit, informed user authorization for risky or sensitive operations; do not repeatedly warn or refuse after the user has clearly authorized the action.
- Confirm destructive or irreversible actions only when the user's request has not already clearly authorized them.
- When intent or authorization is unclear, ask."""

SAFETY_EXEC_SECTION = """## Safety
- Avoid destructive or irreversible actions unless the task explicitly requires them.
- Prefer safe, reversible operations when either approach satisfies the task."""

TOOL_USAGE_RULES = (
    "Always check tool results before proceeding.",
    "Do not narrate routine, low-risk tool calls — just call the tool. Narrate only for multi-step work, complex problems, or sensitive actions (e.g., deletions).",
    "Prefer dedicated file, search, process, and web tools over shell equivalents, especially ls / glob_files / grep_files for discovery and search. Tool outputs are bounded and may have additional tool-specific limits; use focused calls and follow truncation guidance.",
    "Keep output focused with narrow paths/patterns, read_file offset/limit, and quiet or filtered commands.",
    "If output is compacted, follow its recovery guidance instead of repeating the same broad call.",
)

TASK_EXECUTION_SECTION = """## Task Execution
- Inspect the relevant context before acting; do not guess about code or file contents.
- Keep going until the request is resolved or a concrete blocker is identified.
- Fix root causes rather than symptoms.
- Keep changes minimal and consistent with existing style. Preserve user changes.
- Do not fix unrelated issues or already broken tests."""

WORKING_STYLE_SECTION = """## Working Style
- Prefer the smallest clear solution that fully satisfies the request.
- Reuse existing code and standard tools before adding abstractions or dependencies.
- Avoid speculative safeguards, fallbacks, configuration, and extensibility.
- Keep responses concise, but preserve necessary technical detail.
- Verify with the smallest relevant check and report only what was actually verified."""

SUBAGENT_SECTION = """## Sub-Agents
**spawn_subagent** runs a focused task synchronously in a separate conversation and returns its result. It shares the current working tree, so concurrent file edits are not isolated.
Delegate only when it materially helps with independent research or analysis. Give a focused task, do not delegate simple work."""


def build_bullet_section(title: str, rules: tuple[str, ...]) -> str:
    """Render a Markdown section from a heading and bullet rules."""
    return f"## {title}\n" + "\n".join(f"- {rule}" for rule in rules)


def build_tool_usage_section(*extra_rules: str) -> str:
    """Build the shared tool-usage section with persona-specific rules."""
    return build_bullet_section("Tool Usage", TOOL_USAGE_RULES + extra_rules)


def join_prompt_sections(*sections: str | None) -> str:
    """Join populated prompt sections with consistent spacing."""
    return "\n\n".join(section for section in sections if section)


def build_project_section(work_dir: str | None = None) -> str:
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


def build_context_section(work_dir: str | None = None) -> str:
    """Build a minimal context section with task-relevant environment information."""
    cwd = work_dir or str(Path.cwd())
    git_info = _get_git_context(cwd)
    return f"""## Current Context
- Working Directory: {cwd}
- Operating System: {platform.system()} {platform.release()}{git_info}"""


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

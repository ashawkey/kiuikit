"""The default coding agent persona — full tool access, project-aware."""

from .common import (
    EXEC_MODE_SECTION,
    SAFETY_EXEC_SECTION,
    SAFETY_SECTION,
    SUBAGENT_SECTION,
    TASK_EXECUTION_SECTION,
    WORKING_STYLE_SECTION,
    build_context_section,
    build_project_section,
    build_tool_usage_section,
    join_prompt_sections,
)
from kiui.agent.skills import build_skills_prompt_section

NAME = "coder"
DESCRIPTION = "Full coding agent — all tools, project-aware (default)."
TOOLS = None  # all tools


def build_system_prompt(ctx) -> str:
    sections = [
        "You are a terminal-based AI agent. "
        "Be helpful, accurate, and concise. "
        "Prioritize correctness, then clarity, then brevity."
    ]

    if ctx.exec_mode:
        sections.append(EXEC_MODE_SECTION)

    sections.append(SAFETY_EXEC_SECTION if ctx.exec_mode else SAFETY_SECTION)
    sections.append(build_tool_usage_section(
        "Use exec_command for foreground commands expected to finish reliably; when the command exits, the agent automatically continues from its result.",
        "Use start_process for servers and long-running or potentially stuck commands, then inspect_processes(wait=N) to wait; request log_tail_chars when recent output is needed.",
    ))
    sections.append(TASK_EXECUTION_SECTION)
    sections.append(WORKING_STYLE_SECTION)

    # sub-agents cannot spawn children, so advertising it would be misleading
    if not ctx.is_subagent:
        sections.append(SUBAGENT_SECTION)

    skills_section = build_skills_prompt_section(ctx.skills or {})
    if skills_section:
        sections.append(skills_section)

    project = build_project_section(ctx.work_dir)
    if project:
        sections.append(project)

    sections.append(build_context_section(ctx.work_dir))

    return join_prompt_sections(*sections)

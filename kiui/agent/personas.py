"""Declarative persona parsing, discovery, and prompt rendering."""

from __future__ import annotations

import hashlib
import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from kiui.agent.utils.frontmatter import split_frontmatter
from kiui.agent.skills import build_skills_prompt_section
from kiui.agent.tools.registry import BUILTIN_TOOL_NAMES

DEFAULT_PERSONA = "coder"
BUNDLED_PERSONAS_DIR = Path(__file__).parent / "bundled_personas"
_PERSONA_FILE = "PERSONA.md"
_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_RESERVED_NAMES = frozenset({"reload"})
_MARKER_RE = re.compile(r"\{\{kia:([^{}]+)\}\}")
_MARKERS = frozenset({
    "autonomous-mode",
    "sub-agents",
    "skills",
    "project-instructions",
    "current-context",
})

_AUTONOMOUS_MODE = """## Autonomous Mode
You are running as an autonomous sub-agent with no user available to respond.
- Do not ask questions or request confirmation.
- Infer intent from the task and available context; choose the safest reasonable interpretation.
- If blocked, report the blocker instead of using a risky workaround.
- Complete and verify the task, then return a concise summary."""

_SUBAGENTS = """## Sub-Agents
**spawn_subagent** runs a focused task synchronously in a separate conversation and returns its result.
Use a sub-agent only when the user explicitly requests delegation or when a task is genuinely independent and context isolation is clearly necessary, such as work unrelated to the current codebase. When delegation is justified, give one focused, self-contained task."""


@dataclass(frozen=True)
class PersonaContext:
    """Runtime information available to persona markers."""

    exec_mode: bool = False
    is_subagent: bool = False
    work_dir: str | None = None
    skills: dict | None = None


@dataclass(frozen=True)
class PersonaInfo:
    name: str
    description: str
    tools: frozenset[str] | None
    template: str
    path: str
    source: str
    digest: str

    def build(self, ctx: PersonaContext) -> str:
        return render_persona(self, ctx)


def valid_persona_name(name: str) -> bool:
    return (
        len(name) <= 64
        and _NAME_RE.fullmatch(name) is not None
        and name not in _RESERVED_NAMES
    )


def read_persona(persona_dir: str | Path, source: str = "local") -> PersonaInfo:
    """Parse and validate one directory containing ``PERSONA.md``."""
    item = Path(persona_dir)
    persona_md = item / _PERSONA_FILE
    raw = persona_md.read_text(encoding="utf-8")
    frontmatter, body = split_frontmatter(raw)
    if frontmatter is None:
        raise ValueError("invalid or missing YAML frontmatter")

    name = frontmatter.get("name")
    if not isinstance(name, str) or not valid_persona_name(name):
        raise ValueError("'name' must be 1-64 lowercase alphanumeric/hyphen chars and not 'reload'")
    if name != item.name:
        raise ValueError(f"name '{name}' does not match directory '{item.name}'")

    description = frontmatter.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("'description' must be a non-empty string")
    if len(description) > 1024:
        raise ValueError("description exceeds 1024 characters")

    tools_value = frontmatter.get("tools")
    if tools_value == "all":
        tools = None
    elif isinstance(tools_value, list) and all(isinstance(tool, str) for tool in tools_value):
        unknown = set(tools_value) - set(BUILTIN_TOOL_NAMES)
        if unknown:
            raise ValueError(
                f"unknown tool(s): {sorted(unknown)}; valid tools: {sorted(BUILTIN_TOOL_NAMES)}"
            )
        tools = frozenset(tools_value)
    else:
        raise ValueError("'tools' must be 'all' or a list of built-in tool names")

    template = body.strip()
    if not template:
        raise ValueError("persona prompt body is empty")

    seen: set[str] = set()
    for line in template.splitlines():
        matches = list(_MARKER_RE.finditer(line))
        if not matches:
            continue
        match = matches[0]
        marker = match.group(1)
        if marker not in _MARKERS:
            raise ValueError(f"unknown marker '{{{{kia:{marker}}}}}'")
        if len(matches) != 1 or line.strip() != match.group(0):
            raise ValueError(f"marker '{{{{kia:{marker}}}}}' must occupy its own line")
        if marker in seen:
            raise ValueError(f"duplicate marker '{{{{kia:{marker}}}}}'")
        seen.add(marker)

    return PersonaInfo(
        name=name,
        description=description.strip(),
        tools=tools,
        template=template,
        path=str(persona_md),
        source=source,
        digest=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    )


def discover_personas(
    work_dir: str | Path | None = None,
    issues: dict | None = None,
) -> dict[str, PersonaInfo]:
    """Discover bundled, project, and personal personas in precedence order."""
    base = Path(work_dir) if work_dir else Path.cwd()
    home = Path.home()
    locations = [
        (BUNDLED_PERSONAS_DIR, "bundled"),
        (base / ".kia" / "personas", "project"),
    ]
    if base.resolve() != home.resolve():
        locations.append((home / ".kia" / "personas", "personal"))

    personas: dict[str, PersonaInfo] = {}
    shadowed: list[dict] = []
    errors: list[dict] = []
    for root, source in locations:
        if not root.is_dir():
            continue
        for item in sorted(root.iterdir()):
            if not item.is_dir() or not (item / _PERSONA_FILE).is_file():
                continue
            if item.name in personas:
                shadowed.append({
                    "name": item.name,
                    "path": str(item / _PERSONA_FILE),
                    "shadowed_by": personas[item.name].path,
                })
                continue
            try:
                personas[item.name] = read_persona(item, source=source)
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                errors.append({
                    "name": item.name,
                    "path": str(item / _PERSONA_FILE),
                    "reason": str(exc),
                })

    if issues is not None:
        issues["shadowed"] = shadowed
        issues["errors"] = errors
    return personas


def list_personas(work_dir: str | Path | None = None) -> dict[str, PersonaInfo]:
    return discover_personas(work_dir)


def get_persona(
    name: str,
    work_dir: str | Path | None = None,
    personas: dict[str, PersonaInfo] | None = None,
) -> PersonaInfo:
    available = personas if personas is not None else discover_personas(work_dir)
    if name not in available:
        raise ValueError(f"Unknown persona '{name}'. Available: {', '.join(available)}")
    return available[name]


def render_persona(persona: PersonaInfo, ctx: PersonaContext) -> str:
    """Expand a persona's recognized markers exactly once."""
    expansions = {
        "autonomous-mode": _AUTONOMOUS_MODE if ctx.exec_mode else "",
        "sub-agents": _SUBAGENTS if not ctx.is_subagent and _allows(persona, "spawn_subagent") else "",
        "skills": build_skills_prompt_section(ctx.skills or {}) if _allows(persona, "load_skill") else "",
        "project-instructions": _build_project_section(ctx.work_dir),
        "current-context": _build_context_section(ctx.work_dir),
    }
    rendered = []
    for line in persona.template.splitlines():
        match = _MARKER_RE.fullmatch(line.strip())
        rendered.append(expansions[match.group(1)] if match else line)
    return "\n".join(rendered).strip()


def _allows(persona: PersonaInfo, tool: str) -> bool:
    return persona.tools is None or tool in persona.tools


def _build_project_section(work_dir: str | None) -> str:
    instr_file = Path(work_dir) / "AGENTS.md" if work_dir else Path.cwd() / "AGENTS.md"
    try:
        content = instr_file.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return ""
    return f"## Project Instructions\n{content}" if content else ""


def _build_context_section(work_dir: str | None) -> str:
    cwd = work_dir or str(Path.cwd())
    git_info = _get_git_context(cwd)
    os_line = f"- Operating System: {platform.system()} {platform.release()}"
    if platform.system() == "Windows":
        os_line += "\n- Shell: PowerShell"
    return f"## Current Context\n- Working Directory: {cwd}\n{os_line}{git_info}"


def _get_git_context(cwd: str) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
        )
        if result.returncode != 0:
            return ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""

    branch = _git_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd)
    lines = [f"- Git Branch: {branch}"] if branch else []
    status = _git_cmd(["git", "status", "--porcelain"], cwd)
    lines.append(f"- Git Status: {'dirty' if status else 'clean'}")
    return "\n" + "\n".join(lines)


def _git_cmd(args: list[str], cwd: str) -> str | None:
    try:
        result = subprocess.run(args, capture_output=True, text=True, cwd=cwd, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None

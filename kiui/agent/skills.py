"""Skill discovery, parsing, and registry formatting for kiui agent.

Implements the open Agent Skills format (https://agentskills.io). A skill is a
folder under ``<agent-dir>/skills/<name>/`` containing a ``SKILL.md`` file with
YAML frontmatter (``name`` + ``description`` required, plus optional ``license``,
``compatibility``, ``metadata``, ``allowed-tools``) followed by markdown
instructions. Skills may bundle ``scripts/``, ``references/``, and ``assets/``
directories referenced by relative paths from the skill root.

All frontmatter fields are parsed for compatibility, but ``allowed-tools`` is
not enforced: kia uses its own permission model, so a skill cannot narrow or
widen tool access. The field is accepted (so cross-agent skills load cleanly)
but has no effect.

Skills load via progressive disclosure: only name+description is advertised in
the system prompt; the full body is pulled into context on demand via the
load_skill tool (or the manual ``/skills <name>`` command). Bundled resource
files are read only when the instructions call for them (using the ordinary
read_file / exec_command tools).

Skills are discovered from kiui's bundled skill directory and ``.kia/skills``
under the working directory (project skills) and the user's home (personal
skills shared across projects). Bundled skills take precedence so they always
match the installed kiui version; project skills take precedence over personal
skills. Skills for other agents can still be used by giving kia their paths
explicitly.
"""

from __future__ import annotations

import importlib.util
import re
import uuid
from pathlib import Path
from typing import Any

from kiui.agent.utils.frontmatter import split_frontmatter

SKILL_DIRS = (".kia",)

# Bundled skills are loaded directly from the installed package so their
# instructions and resources always match the installed kiui version.
BUNDLED_SKILLS_DIR = Path(__file__).parent / "bundled_skills"

# name: 1-64 chars, lowercase alphanumeric + single hyphens, no leading/trailing/
# consecutive hyphens (per the Agent Skills spec).
_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def valid_skill_name(name: str) -> bool:
    """Return whether *name* is a valid Agent Skills identifier."""
    return len(name) <= 64 and _NAME_RE.fullmatch(name) is not None


def read_skill(skill_dir: str | Path, strict: bool = False) -> dict:
    """Read one skill directory and return its parsed metadata.

    When *strict* is true, specification warnings are rejected. This is used at
    library boundaries while local discovery remains permissive for compatible
    third-party skills.
    """
    item = Path(skill_dir)
    skill_md = item / "SKILL.md"
    raw = skill_md.read_text(encoding="utf-8")
    parsed = _parse_skill(raw)
    if parsed is None:
        raise ValueError("invalid or missing YAML frontmatter")

    frontmatter, body = parsed
    description = frontmatter.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("missing or non-string 'description'")

    warnings = validate_skill(item.name, frontmatter)
    if strict and warnings:
        raise ValueError("; ".join(warnings))

    return {
        "path": str(skill_md),
        "dir": str(item),
        "description": description.strip(),
        "body": body,
        "frontmatter": frontmatter,
        "warnings": warnings,
        "active": True,
    }


def discover_skills(
    work_dir: str | Path | None = None,
    issues: dict | None = None,
) -> dict[str, dict]:
    """Scan bundled, project, and personal skill folders for SKILL.md files.

    Bundled skills are searched first so stale project copies left by older kia
    versions cannot shadow them. The project directory is then searched before
    the user's home directory. Returns ``{skill_name: {"path", "dir",
    "description", "body", "frontmatter", "active"}}`` where ``dir`` is the
    skill root (for resolving bundled resources), ``body`` is the markdown
    instructions with frontmatter stripped, and ``active`` is always true.

    When *issues* is a dict, it is populated (in place) with non-fatal discovery
    problems so callers can surface them:

    - ``issues["shadowed"]``: list of ``{"name", "path", "shadowed_by"}`` for
      skills dropped because an earlier scope/dir already defined that name.
    - ``issues["errors"]``: list of ``{"name", "path", "reason"}`` for SKILL.md
      files that could not be read or parsed (unreadable, invalid YAML, or
      missing required frontmatter/description).
    """
    base = Path(work_dir) if work_dir else Path.cwd()
    home = Path.home()

    shadowed: list[dict] = []
    errors: list[dict] = []

    # Bundled skills are authoritative. Project scope then wins over personal
    # scope. Skip home when the project already is home to avoid a duplicate scan.
    skill_dirs = [BUNDLED_SKILLS_DIR, base / SKILL_DIRS[0] / "skills"]
    if base.resolve() != home.resolve():
        skill_dirs.append(home / SKILL_DIRS[0] / "skills")

    skills: dict[str, dict] = {}
    for skills_dir in skill_dirs:
        if not skills_dir.is_dir():
            continue

        for item in sorted(skills_dir.iterdir()):
            if not item.is_dir():
                continue  # skip loose files

            skill_md = item / "SKILL.md"
            if not skill_md.is_file():
                continue  # folder without SKILL.md is not a skill

            skill_name = item.name
            if skill_name in skills:
                # A higher-precedence scope/dir already defined this name.
                shadowed.append({
                    "name": skill_name,
                    "path": str(skill_md),
                    "shadowed_by": skills[skill_name]["path"],
                })
                continue

            try:
                skills[skill_name] = read_skill(item)
            except (OSError, UnicodeDecodeError, ValueError) as e:
                errors.append({
                    "name": skill_name,
                    "path": str(skill_md),
                    "reason": str(e),
                })
                continue

    if issues is not None:
        issues["shadowed"] = shadowed
        issues["errors"] = errors

    return skills


def _parse_skill(raw: str) -> tuple[dict, str] | None:
    """Parse a SKILL.md into (frontmatter dict, body markdown).

    Requires a leading YAML frontmatter block delimited by ``---`` lines.
    Field-level specification validation is performed by :func:`validate_skill`.
    """
    frontmatter, body = split_frontmatter(raw)
    if frontmatter is None:
        return None

    return frontmatter, body.strip()


def validate_skill(name: str, frontmatter: dict) -> list[str]:
    """Return non-fatal Agent Skills specification warnings (empty if valid).

    Discovery intentionally accepts usable third-party skills that deviate from
    the standard. The directory name is always the runtime identifier.
    """
    errors: list[str] = []

    fm_name = frontmatter.get("name")
    if not isinstance(fm_name, str):
        errors.append("'name' must be a string")
    else:
        if not valid_skill_name(fm_name):
            errors.append("name must be 1-64 lowercase alphanumeric/hyphen chars")
        if fm_name != name:
            errors.append(f"name '{fm_name}' does not match directory '{name}'")

    desc = frontmatter.get("description")
    if not isinstance(desc, str) or not desc.strip():
        errors.append("'description' must be a non-empty string")
    elif len(desc) > 1024:
        errors.append("description exceeds 1024 characters")

    compatibility = frontmatter.get("compatibility")
    if compatibility is not None and (
        not isinstance(compatibility, str)
        or not compatibility
        or len(compatibility) > 500
    ):
        errors.append("compatibility must be a non-empty string of at most 500 characters")

    metadata = frontmatter.get("metadata")
    if metadata is not None and (
        not isinstance(metadata, dict)
        or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items())
    ):
        errors.append("metadata must map strings to strings")

    # unsupported, just parse it.
    allowed_tools = frontmatter.get("allowed-tools")
    if allowed_tools is not None and not isinstance(allowed_tools, str):
        errors.append("allowed-tools must be a space-separated string")

    license_value = frontmatter.get("license")
    if license_value is not None and not isinstance(license_value, str):
        errors.append("license must be a string")

    return errors


def build_skills_prompt_section(skills: dict[str, dict]) -> str:
    """Build a concise skills registry section for the system prompt.

    Advertises each skill's name + description (progressive disclosure stage 1)
    and how to activate it via the load_skill tool.
    """
    active_skills = {
        name: info for name, info in skills.items() if info.get("active", True)
    }
    if not active_skills:
        return ""

    lines = [
        "## Skills",
        "",
        "The following specialized skills are available. Before acting, check whether",
        "the request matches a skill below; if it does, use **load_skill** before doing",
        "the task and follow the loaded instructions. If the user asks to create or",
        "modify a skill, load a skill-creation skill when one is available. A loaded skill",
        "may reference bundled",
        "files (e.g. references/… or scripts/…) relative to its directory; read or",
        "run those with the ordinary file/exec tools when the instructions call for it.",
        "",
    ]

    for name, info in active_skills.items():
        desc = info.get("description", "")
        lines.append(f"- **{name}**: {desc}")

    return "\n".join(lines)


def get_skill_body(skills: dict[str, dict], name: str) -> str | None:
    """Return the full SKILL.md body for a given skill name, or None if not found."""
    skill = skills.get(name)
    return skill["body"] if skill else None


def load_skill_tools(skill_dir: str | Path) -> list[dict[str, Any]]:
    """Load the tool definitions a skill contributes, or ``[]`` if it has none.

    A skill may ship a ``tools.py`` at its root exposing a module-level
    ``TOOLS`` list. Each entry must provide an OpenAI function ``schema``, a
    ``run`` callable (invoked as ``run(executor, **arguments)``), and an
    optional ``permission`` class (``"safe"`` or ``"risky"``, default risky).

    Since these tools become callable in-process, this is only appropriate for
    trusted (user-authored and bundled) skills. The module is imported under a
    unique name so re-loading a skill picks up edits and never collides.
    """
    tools_py = Path(skill_dir) / "tools.py"
    if not tools_py.is_file():
        return []

    module_name = f"kiui_skill_tools_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, tools_py)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not load tools.py for skill at {skill_dir}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    entries = getattr(module, "TOOLS", None)
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise ValueError(f"skill tools.py at {skill_dir} must define TOOLS as a list")
    return entries

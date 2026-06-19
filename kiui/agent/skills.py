"""Skill discovery, parsing, and registry formatting for kiui agent.

A skill is a folder under .kia/skills/<name>/ containing a SKILL.md file.
The SKILL.md describes what the skill does and provides domain-specific
instructions the model can load on demand via the load_skill tool.
"""

from __future__ import annotations

from pathlib import Path


def discover_skills(work_dir: str | Path | None = None) -> dict[str, dict]:
    """Scan .kia/skills/ for folders containing SKILL.md.

    Returns ``{skill_name: {"path": str, "description": str, "body": str}}``.
    The description is taken from a ``## Description`` section, or the first
    non-blank, non-heading line of the file.
    """
    base = Path(work_dir) if work_dir else Path.cwd()
    skills_dir = base / ".kia" / "skills"

    if not skills_dir.is_dir():
        return {}

    skills: dict[str, dict] = {}
    for item in sorted(skills_dir.iterdir()):
        if not item.is_dir():
            continue  # skip loose files

        skill_md = item / "SKILL.md"
        if not skill_md.is_file():
            continue  # folder without SKILL.md is not a skill

        try:
            body = skill_md.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            continue

        if not body:
            continue

        skill_name = item.name
        description = _extract_description(body)

        if skill_name in skills:
            # Duplicate name — last one wins (unlikely for folder names, but safe)
            pass

        skills[skill_name] = {
            "path": str(skill_md),
            "description": description,
            "body": body,
        }

    return skills


def _extract_description(body: str) -> str:
    """Extract a short description from the SKILL.md body.

    Looks for ``## Description`` first; falls back to the first non-blank,
    non-ATX-heading line (trimmed to 200 chars).
    """
    lines = body.splitlines()

    # Look for ## Description section
    in_description = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("## description"):
            in_description = True
            continue
        if in_description:
            if stripped.startswith("#"):
                break  # next heading ends the description
            if stripped:
                return stripped[:200]

    # Fallback: first non-blank, non-heading line
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped[:200]

    return "(no description)"


def build_skills_prompt_section(skills: dict[str, dict]) -> str:
    """Build a concise skills registry section for the system prompt.

    Lists each skill name, description, and the load_skill invocation pattern
    so the model knows what is available and how to activate each skill.
    """
    if not skills:
        return ""

    lines = [
        "## Skills",
        "",
        "The following specialized skills are available. When a user request matches",
        "a skill's domain, use the **load_skill** tool with the skill name to load",
        "its full instructions into context.",
        "",
    ]

    for name, info in skills.items():
        desc = info.get("description", "")
        lines.append(f"- **{name}**: {desc}")

    return "\n".join(lines)


def get_skill_body(skills: dict[str, dict], name: str) -> str | None:
    """Return the full SKILL.md body for a given skill name, or None if not found."""
    skill = skills.get(name)
    return skill["body"] if skill else None

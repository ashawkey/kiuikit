#!/usr/bin/env python3
"""Validate an Agent Skill package."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from kiui.agent.skills import load_skill_tools, read_skill
from kiui.agent.tools.registry import ToolRegistry

_RESOURCE_RE = re.compile(
    r"`((?:scripts|references|assets)/[^`\r\n]+)`"
    r"|(?<![\w./-])((?:scripts|references|assets)/[A-Za-z0-9_./-]+)"
)
_TRAILING_PUNCTUATION = ".,;:)"


def validate(skill_dir: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    skill_dir = skill_dir.expanduser().resolve()

    if not skill_dir.is_dir():
        return [f"not a directory: {skill_dir}"], warnings
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        return [f"missing file: {skill_md}"], warnings

    try:
        info = read_skill(skill_dir, strict=True)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return [str(exc)], warnings

    if len(info["body"].splitlines()) > 500:
        warnings.append("SKILL.md body exceeds the recommended 500 lines")

    raw = skill_md.read_text(encoding="utf-8")
    referenced = {
        (match.group(1) or match.group(2)).rstrip(_TRAILING_PUNCTUATION)
        for match in _RESOURCE_RE.finditer(raw)
    }
    for relative in sorted(referenced):
        if not (skill_dir / relative).exists():
            errors.append(f"referenced resource does not exist: {relative}")

    for source in sorted(skill_dir.rglob("*.py")):
        try:
            compile(source.read_text(encoding="utf-8"), str(source), "exec")
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            errors.append(
                f"Python syntax error in {source.relative_to(skill_dir)}: {exc}"
            )

    tools_py = skill_dir / "tools.py"
    if tools_py.is_file() and not any(
        error.startswith("Python syntax error in tools.py") for error in errors
    ):
        try:
            entries = load_skill_tools(skill_dir)
            ToolRegistry().register_skill(info["frontmatter"]["name"], entries)
        except Exception as exc:
            errors.append(f"invalid tools.py: {exc}")

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an Agent Skill directory.")
    parser.add_argument("skill_dir", type=Path, help="path to the skill directory")
    args = parser.parse_args()

    errors, warnings = validate(args.skill_dir)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if errors:
        print(f"FAILED ({len(errors)} error(s), {len(warnings)} warning(s))", file=sys.stderr)
        return 1
    print(f"OK ({len(warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""YAML-frontmatter parsing shared by declarative agent resources."""

from __future__ import annotations

import yaml


def split_frontmatter(raw: str) -> tuple[dict | None, str]:
    """Split text into a YAML mapping and body.

    The first non-empty line must be ``---``. Invalid or unterminated
    frontmatter returns ``(None, raw)``.
    """
    text = raw.lstrip("\ufeff")
    lines = text.splitlines()

    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1
    if start >= len(lines) or lines[start].strip() != "---":
        return None, raw

    for end in range(start + 1, len(lines)):
        if lines[end].strip() != "---":
            continue
        try:
            data = yaml.safe_load("\n".join(lines[start + 1 : end]))
        except yaml.YAMLError:
            return None, raw
        if not isinstance(data, dict):
            return None, raw
        return data, "\n".join(lines[end + 1 :])

    return None, raw

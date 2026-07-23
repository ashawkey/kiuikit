"""Unified tool registry: one descriptor and one resolver for every tool.

Every tool the model can call — built-in or skill-provided — is described by a
single :class:`ToolSpec` (OpenAI ``schema`` + ``handler`` + ``permission`` +
optional ``gate``). :class:`ToolRegistry` holds them all and is the single
source of truth for three questions that used to be answered in three places:

- *Which tools are advertised to the API?* — :meth:`ToolRegistry.advertised`
  applies gates (image/sub-agent/goal), the persona whitelist, and the
  per-persona skill-tool policy in one pass.
- *How does a call dispatch?* — :meth:`ToolRegistry.get` returns the spec whose
  ``handler`` is invoked as ``handler(executor, **arguments)``.
- *What is a tool's permission class?* — :attr:`ToolSpec.permission`
  (``"safe"`` / ``"risky"``), consulted by the permission controller.

Registering a skill's tools is atomic and rejects any name that collides with a
built-in or with another loaded skill's tool, so conflicts fail loudly at load
time instead of silently shadowing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .schemas import BUILTIN_TOOL_SCHEMAS

# Tool source marker for built-in (executor-owned) tools; skill tools use their
# skill name as the source.
BUILTIN_SOURCE = "builtin"

# Advertising gates. A gated tool is advertised only when its condition holds:
#   "image"         -> the model supports image input
#   "subagent_root" -> the agent is not itself a sub-agent
#   "goal"          -> a standing goal is currently active
GATE_IMAGE = "image"
GATE_SUBAGENT_ROOT = "subagent_root"
GATE_GOAL = "goal"


@dataclass(frozen=True)
class ToolSpec:
    """A single callable tool exposed to the model.

    ``handler`` is invoked as ``handler(executor, **arguments)`` and returns a
    result dict. ``permission`` is ``"safe"`` (never prompts) or ``"risky"``
    (prompts in default mode). ``source`` is :data:`BUILTIN_SOURCE` for built-in
    tools or the owning skill name. ``gate`` is one of the ``GATE_*`` constants
    or ``None``.
    """

    name: str
    schema: dict[str, Any]
    handler: Callable[..., dict[str, Any]]
    permission: str = "risky"
    source: str = BUILTIN_SOURCE
    gate: str | None = None


# ---------------------------------------------------------------------------
# Built-in tool table
# ---------------------------------------------------------------------------

# name -> (executor method, permission, gate). The schema is pulled from
# BUILTIN_TOOL_SCHEMAS so the wire format stays defined once in schemas.py.
_BUILTIN_TABLE: dict[str, tuple[str, str, str | None]] = {
    "read_file": ("_read_file", "safe", None),
    "read_image": ("_read_image", "safe", GATE_IMAGE),
    "write_file": ("_write_file", "risky", None),
    "edit_file": ("_edit_file", "risky", None),
    "multi_edit": ("_multi_edit", "risky", None),
    "ls": ("_ls", "safe", None),
    "exec_command": ("_exec_command", "risky", None),
    "glob_files": ("_glob_files", "safe", None),
    "grep_files": ("_grep_files", "safe", None),
    "web_search": ("_web_search", "safe", None),
    "web_fetch": ("_web_fetch", "safe", None),
    "remove_file": ("_remove_file", "risky", None),
    "spawn_subagent": ("_spawn_subagent", "risky", GATE_SUBAGENT_ROOT),
    "load_skill": ("_load_skill", "safe", None),
    "report_goal": ("_report_goal", "safe", GATE_GOAL),
}

BUILTIN_TOOL_NAMES = frozenset(_BUILTIN_TABLE)


def _builtin_handler(method_name: str) -> Callable[..., dict[str, Any]]:
    """Wrap an executor method so all specs share one ``handler(executor, ...)``
    calling convention."""

    def handler(executor, **kwargs):
        return getattr(executor, method_name)(**kwargs)

    return handler


def build_builtin_specs() -> dict[str, ToolSpec]:
    """Build the built-in ToolSpec table (name -> spec)."""
    specs: dict[str, ToolSpec] = {}
    for name, (method_name, permission, gate) in _BUILTIN_TABLE.items():
        specs[name] = ToolSpec(
            name=name,
            schema=BUILTIN_TOOL_SCHEMAS[name],
            handler=_builtin_handler(method_name),
            permission=permission,
            source=BUILTIN_SOURCE,
            gate=gate,
        )
    return specs


class ToolRegistry:
    """Holds all tool specs and resolves the advertised set for the API."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = build_builtin_specs()

    def get(self, name: str) -> ToolSpec | None:
        """Return the spec for *name*, or ``None`` if no such tool exists."""
        return self._specs.get(name)

    def permission(self, name: str) -> str | None:
        """Return a tool's permission class, or ``None`` if it is unknown."""
        spec = self._specs.get(name)
        return spec.permission if spec is not None else None

    # -- skill tools --------------------------------------------------------

    def register_skill(self, skill: str, entries: list[dict[str, Any]]) -> None:
        """Register a skill's tools atomically.

        Each entry provides an OpenAI function ``schema``, a ``run`` callable
        (used as the handler), and an optional ``permission`` class (default
        ``"risky"``). A tool name that collides with a built-in or with another
        already-registered skill's tool aborts the whole registration before any
        entry is committed, so a skill is never left partially registered.
        """
        prepared: dict[str, ToolSpec] = {}
        for entry in entries:
            name = entry["schema"]["function"]["name"]
            existing = self._specs.get(name)
            if existing is not None and existing.source != skill:
                where = (
                    "a built-in tool"
                    if existing.source == BUILTIN_SOURCE
                    else f"skill '{existing.source}'"
                )
                raise ValueError(
                    f"Skill '{skill}' tool '{name}' collides with {where}."
                )
            prepared[name] = ToolSpec(
                name=name,
                schema=entry["schema"],
                handler=entry["run"],
                permission=entry.get("permission", "risky"),
                source=skill,
            )
        self._specs.update(prepared)

    def unregister_skill(self, skill: str) -> None:
        """Drop all tools contributed by *skill*."""
        self._specs = {
            name: spec for name, spec in self._specs.items() if spec.source != skill
        }

    def clear_skill_tools(self) -> None:
        """Drop every skill-contributed tool, keeping only built-ins."""
        self._specs = {
            name: spec
            for name, spec in self._specs.items()
            if spec.source == BUILTIN_SOURCE
        }

    def skill_tool_schemas(self) -> list[dict[str, Any]]:
        """Return schemas for every currently-registered skill tool (any source
        that is not built-in)."""
        return [
            spec.schema
            for spec in self._specs.values()
            if spec.source != BUILTIN_SOURCE
        ]

    # -- advertising --------------------------------------------------------

    def advertised(
        self,
        *,
        persona_tools: frozenset[str] | None,
        include_subagent: bool,
        supports_image: bool,
        goal_active: bool,
    ) -> list[dict[str, Any]]:
        """Return the OpenAI tool schemas to advertise for the current turn.

        Applies, in one pass: advertising gates and the persona's built-in
        whitelist (``None`` = all built-ins). Skill tools cannot be named in a
        whitelist (their names are not known up front), so they are advertised
        whenever the persona can call ``load_skill`` — a persona that cannot
        load skills cannot obtain their tools anyway.
        """
        gate_ok = {
            GATE_IMAGE: supports_image,
            GATE_SUBAGENT_ROOT: include_subagent,
            GATE_GOAL: goal_active,
        }
        allow_skill_tools = persona_tools is None or "load_skill" in persona_tools
        advertised: list[dict[str, Any]] = []
        for spec in self._specs.values():
            if spec.gate is not None and not gate_ok[spec.gate]:
                continue
            if spec.source == BUILTIN_SOURCE:
                if persona_tools is not None and spec.name not in persona_tools:
                    continue
            elif not allow_skill_tools:
                continue
            advertised.append(spec.schema)
        return advertised

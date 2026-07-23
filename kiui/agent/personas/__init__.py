"""Persona registry for kiui agent.

A persona is a Python module in this package that owns the agent's identity,
system prompt, and tool surface:

- ``build_system_prompt(ctx) -> str`` (required): builds the complete system
  prompt from a :class:`~kiui.agent.personas.common.PersonaContext`.
- ``TOOLS`` (required): whitelist of built-in tool names; ``None`` = all
  built-ins, ``[]`` = no built-ins. Enforced in the advertised tool set. Tools
  contributed by loaded skills are advertised whenever the persona can call
  ``load_skill`` (i.e. it is in ``TOOLS``, or ``TOOLS is None``), since a
  persona that cannot load skills cannot obtain their tools anyway.
- ``NAME`` / ``DESCRIPTION`` (required): shown by ``/persona``.

The bundled ``coder`` persona is the default. Select another at startup with
``kia --persona <name>`` or mid-session with ``/persona <name>`` (which
restarts the conversation).
"""

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .common import PersonaContext

PERSONAS_DIR = Path(__file__).parent
DEFAULT_PERSONA = "coder"


@dataclass
class PersonaInfo:
    name: str
    description: str
    tools: frozenset | None  # None = all built-in tools
    build: Callable[[PersonaContext], str]
    path: str


def _load_persona(path: Path) -> PersonaInfo:
    """Import one persona module and validate its contract."""
    module = importlib.import_module(f"{__name__}.{path.stem}")

    tools = module.TOOLS
    if tools is not None:
        from kiui.agent.tools import BUILTIN_TOOL_NAMES

        unknown = set(tools) - set(BUILTIN_TOOL_NAMES)
        if unknown:
            raise ValueError(
                f"Persona '{path.stem}' whitelists unknown tool(s): {sorted(unknown)}. "
                f"Valid tools: {sorted(BUILTIN_TOOL_NAMES)}"
            )
        tools = frozenset(tools)

    return PersonaInfo(
        name=module.NAME,
        description=module.DESCRIPTION,
        tools=tools,
        build=module.build_system_prompt,
        path=str(path),
    )


def list_personas() -> dict[str, PersonaInfo]:
    """Import every bundled persona module, keyed by persona name."""
    personas: dict[str, PersonaInfo] = {}
    for path in sorted(PERSONAS_DIR.glob("*.py")):
        if path.stem in {"__init__", "common"}:
            continue
        info = _load_persona(path)
        personas[info.name] = info
    return personas


def get_persona(name: str) -> PersonaInfo:
    """Resolve a persona by name. Raises ValueError listing available names."""
    personas = list_personas()
    if name not in personas:
        raise ValueError(f"Unknown persona '{name}'. Available: {', '.join(personas)}")
    return personas[name]

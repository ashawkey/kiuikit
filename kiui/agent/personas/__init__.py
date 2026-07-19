"""Persona registry for kiui agent.

A persona is a Python module in this package that owns the agent's identity,
system prompt, and tool surface:

- ``build_system_prompt(ctx) -> str`` (required): builds the complete system
  prompt from a :class:`~kiui.agent.prompts.PersonaContext`.
- ``TOOLS`` (required): whitelist of tool names; ``None`` = all tools,
  ``[]`` = no tools. Enforced in the advertised tool definitions.
- ``NAME`` / ``DESCRIPTION`` (required): shown by ``/persona``.

The bundled ``coder`` persona is the default. Select another at startup with
``kia --persona <name>`` or mid-session with ``/persona <name>`` (which
restarts the conversation).
"""

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from kiui.agent.prompts import PersonaContext

PERSONAS_DIR = Path(__file__).parent
DEFAULT_PERSONA = "coder"


@dataclass
class PersonaInfo:
    name: str
    description: str
    tools: frozenset | None  # None = all tools
    build: Callable[[PersonaContext], str]
    path: str


def _load_persona(path: Path) -> PersonaInfo:
    """Import one persona module and validate its contract."""
    module = importlib.import_module(f"{__name__}.{path.stem}")

    tools = module.TOOLS
    if tools is not None:
        from kiui.agent.tools import ToolExecutor

        unknown = set(tools) - set(ToolExecutor._DISPATCH_MAP)
        if unknown:
            raise ValueError(
                f"Persona '{path.stem}' whitelists unknown tool(s): {sorted(unknown)}. "
                f"Valid tools: {sorted(ToolExecutor._DISPATCH_MAP)}"
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
        if path.stem == "__init__":
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

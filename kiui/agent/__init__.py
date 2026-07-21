"""kiui agent — terminal-based AI agent with tool-use, web access, and shell execution."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kiui.agent.backend import LLMAgent
    from kiui.agent.permissions import PermissionMode

__all__ = ["LLMAgent", "PermissionMode"]


def __getattr__(name: str):
    if name == "LLMAgent":
        from kiui.agent.backend import LLMAgent
        return LLMAgent
    if name == "PermissionMode":
        from kiui.agent.permissions import PermissionMode
        return PermissionMode
    raise AttributeError(name)

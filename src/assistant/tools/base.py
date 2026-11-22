"""Shared tool definition: one format for the MCP server and the agent."""
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class Risk(str, Enum):
    READ_ONLY = "read_only"    # Safe: runs automatically
    WRITE = "write"            # Changes state: needs user confirmation
    DANGEROUS = "dangerous"    # Destructive or external side effects: confirmation required


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    fn: Callable
    risk: Risk
    external: bool = False     # reaches outside this machine (web, email, third-party APIs)


# Global list that @tool appends to
_TOOLS: list[ToolSpec] = []


def tool(name: str, description: str, risk: Risk = Risk.READ_ONLY, external: bool = False):
    """Decorator that registers a function as a tool.

    The function's type hints become the input schema, so keep them accurate.
    """
    def decorator(fn: Callable) -> Callable:
        if any(t.name == name for t in _TOOLS):
            raise ValueError(f"Duplicate tool name: {name}")
        _TOOLS.append(ToolSpec(name=name, description=description, fn=fn, risk=risk, external=external))
        return fn
    return decorator


def all_tools() -> list[ToolSpec]:
    return list(_TOOLS)

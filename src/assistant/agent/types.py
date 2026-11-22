"""Small shared types for the agent package."""
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class ToolInfo:
    name: str
    description: str
    schema: dict                  # JSON schema of the tool's arguments
    read_only_hint: bool | None   # from the MCP tool annotations
    open_world_hint: bool | None = False  # reaches outside the machine; None means undeclared


class ToolProvider(Protocol):
    """What the agent graph needs from its tools. MCPToolClient implements it."""

    async def list_tools(self) -> list[ToolInfo]: ...

    async def call_tool(self, name: str, args: dict) -> str: ...


@dataclass(frozen=True)
class AgentEvent:
    """What the caller gets back: either the final answer, or a request for approval."""

    kind: Literal["final", "confirm"]
    run_id: str
    answer: str | None = None   # set when kind == "final"
    tool: str | None = None     # set when kind == "confirm"
    args: dict | None = None    # set when kind == "confirm"

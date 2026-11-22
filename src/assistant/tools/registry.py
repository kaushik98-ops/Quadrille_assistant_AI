"""Imports every built-in tool module so their @tool decorators run."""
from assistant.tools.base import ToolSpec, all_tools


def load_tools() -> list[ToolSpec]:
    # Importing registers the tools as a side effect.
    from assistant.tools.builtin import documents, files, notes, web  # noqa: F401
    return all_tools()

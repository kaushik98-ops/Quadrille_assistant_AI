"""Parses the model's JSON reply into a decision."""
from dataclasses import dataclass

from assistant.core.jsonutil import extract_json_object


@dataclass(frozen=True)
class Decision:
    action: str                    # "tool" or "final"
    tool: str | None = None
    args: dict | None = None
    answer: str | None = None


def parse_decision(text: str) -> Decision | None:
    """Return a Decision, or None if the reply isn't valid."""
    data = extract_json_object(text)
    if data is None:
        return None

    if data.get("action") == "final":
        answer = data.get("answer")
        if isinstance(answer, str) and answer.strip():
            return Decision("final", answer=answer)
        return None

    if data.get("action") == "tool":
        tool = data.get("tool")
        args = data.get("args", {})
        if args is None:
            args = {}
        if isinstance(tool, str) and tool and isinstance(args, dict):
            return Decision("tool", tool=tool, args=args)
        return None

    return None

"""Safety rules shared by every layer. Small pure functions, so they are easy to test."""
import json
import re


def needs_confirmation(
    read_only_hint: bool | None,
    open_world_hint: bool | None = False,
    tainted: bool = False,
) -> bool:
    """Decide whether a tool call needs the user's approval.

    - Anything not explicitly read-only always asks (an undeclared tool counts as risky).
    - A read-only tool that reaches outside (web search, ...) is fine on the user's own request,
      but once the run has handled untrusted content it could carry that content out. So it asks.
      An undeclared reach (None) counts as outside.
    """
    if read_only_hint is not True:
        return True
    return tainted and open_world_hint is not False


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[truncated {len(text) - limit} characters]"


def preview_args(args: dict, limit: int = 200) -> str:
    """Short, single-line view of tool arguments for logs, so big or private values aren't dumped."""
    text = json.dumps(args, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "..."


def strip_closing_tag(text: str, tag: str) -> str:
    """Remove closing-tag lookalikes (any case, any spacing) so text can't escape a <tag> block.

    Repeats until nothing changes, because removing one tag can assemble another
    (e.g. '</mess</message>age>').
    """
    pattern = re.compile(rf"<\s*/\s*{re.escape(tag)}\s*>", re.IGNORECASE)
    previous = None
    while previous != text:
        previous = text
        text = pattern.sub("", text)
    return text


def wrap_observation(tool: str, text: str) -> str:
    """Label tool output as data. Closing-tag lookalikes are removed so it can't break out."""
    safe = strip_closing_tag(text, "observation")
    return f'<observation tool="{tool}">\n{safe}\n</observation>'

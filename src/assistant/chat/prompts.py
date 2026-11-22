"""All chat system prompts live here."""

SYSTEM_PROMPT = (
    "You are a helpful personal assistant. Answer clearly and concisely. "
    "If you don't know something, say so instead of guessing."
)


def build_system(extra: str | None = None) -> str:
    """Base prompt, plus optional extra context (e.g. retrieved documents)."""
    return SYSTEM_PROMPT if not extra else f"{SYSTEM_PROMPT}\n\n{extra}"

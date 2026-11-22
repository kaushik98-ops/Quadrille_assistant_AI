"""The one LLM interface the rest of the project depends on."""
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    role: str  # "user" or "assistant"
    content: str


class LLMClient(ABC):
    @abstractmethod
    async def generate(self, messages: list[Message], system: str | None = None) -> str:
        """Return the full reply as one string."""

    @abstractmethod
    def stream(self, messages: list[Message], system: str | None = None) -> AsyncIterator[str]:
        """Yield the reply in text chunks as they arrive."""

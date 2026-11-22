"""Wraps any LLMClient and logs one line per call: timing and sizes. Never logs content."""
import logging
import time
from collections.abc import AsyncIterator

from assistant.core.llm.base import LLMClient, Message
from assistant.core.logging import get_logger

log = get_logger(__name__)


def _size(messages: list[Message], system: str | None) -> int:
    return sum(len(m.content) for m in messages) + len(system or "")


class LoggedLLM(LLMClient):
    def __init__(self, inner: LLMClient, label: str):
        self._inner = inner
        self._label = label

    def _log(self, kind: str, status: str, started: float, in_chars: int, out_chars: int,
             first_token_ms: int | None = None) -> None:
        fields = {
            "llm": self._label, "kind": kind, "status": status,
            "ms": round((time.monotonic() - started) * 1000),
            "in_chars": in_chars, "out_chars": out_chars,
        }
        if first_token_ms is not None:
            fields["first_token_ms"] = first_token_ms
        level = logging.INFO if status == "ok" else logging.WARNING
        log.log(level, "llm_call", extra={"fields": fields})

    async def generate(self, messages: list[Message], system: str | None = None) -> str:
        started, size = time.monotonic(), _size(messages, system)
        try:
            out = await self._inner.generate(messages, system=system)
        except Exception:
            self._log("generate", "error", started, size, 0)
            raise
        self._log("generate", "ok", started, size, len(out))
        return out

    async def stream(self, messages: list[Message], system: str | None = None) -> AsyncIterator[str]:
        started, size = time.monotonic(), _size(messages, system)
        out_chars, first_ms, status = 0, None, "cancelled"  # stays "cancelled" if the client leaves early
        try:
            async for chunk in self._inner.stream(messages, system=system):
                if first_ms is None:
                    first_ms = round((time.monotonic() - started) * 1000)
                out_chars += len(chunk)
                yield chunk
            status = "ok"
        except Exception:
            status = "error"
            raise
        finally:
            self._log("stream", status, started, size, out_chars, first_ms)

"""Small async retry helper with exponential backoff."""
import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from assistant.core.logging import get_logger

log = get_logger(__name__)
T = TypeVar("T")


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> T:
    """Call `fn()` up to `attempts` times, waiting base_delay, 2x, 4x... between tries.

    Raises the last error if every attempt fails. Cancellation is never retried.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except retry_on as exc:
            if attempt == attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            log.warning(
                "retry",
                extra={"fields": {"attempt": attempt, "delay_s": delay, "error": type(exc).__name__}},
            )
            await (sleep or asyncio.sleep)(delay)
    raise AssertionError("unreachable")

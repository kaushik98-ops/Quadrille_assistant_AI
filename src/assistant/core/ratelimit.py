"""Sliding-window rate limiter, in memory."""
import time
from collections import deque
from collections.abc import Callable


class RateLimiter:
    def __init__(self, limit: int, window: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self._limit = limit
        self._window = window
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}

    def check(self, key: str) -> float:
        """Record a request. Returns 0.0 if allowed, otherwise seconds until one is allowed.

        A limit of 0 or less disables limiting. No awaits here, so it is safe under asyncio.
        """
        if self._limit <= 0:
            return 0.0
        now = self._clock()
        hits = self._hits.setdefault(key, deque())
        while hits and now - hits[0] >= self._window:
            hits.popleft()
        if len(hits) >= self._limit:
            return self._window - (now - hits[0])
        hits.append(now)
        return 0.0

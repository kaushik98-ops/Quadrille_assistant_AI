"""Anthropic adapter (default provider)."""
from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic

from assistant.core.llm.base import LLMClient, Message


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, model: str, max_tokens: int = 1024):
        # max_retries gives automatic retry with backoff on transient errors.
        self._client = AsyncAnthropic(api_key=api_key, max_retries=3)
        self._model = model
        self._max_tokens = max_tokens

    def _kwargs(self, messages: list[Message], system: str | None) -> dict:
        kwargs = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if system:
            kwargs["system"] = system
        return kwargs

    async def generate(self, messages: list[Message], system: str | None = None) -> str:
        resp = await self._client.messages.create(**self._kwargs(messages, system))
        return "".join(block.text for block in resp.content if block.type == "text")

    async def stream(self, messages: list[Message], system: str | None = None) -> AsyncIterator[str]:
        async with self._client.messages.stream(**self._kwargs(messages, system)) as s:
            async for text in s.text_stream:
                yield text

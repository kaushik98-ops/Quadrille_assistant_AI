"""OpenAI adapter."""
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from assistant.core.llm.base import LLMClient, Message


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str):
        self._client = AsyncOpenAI(api_key=api_key, max_retries=3)
        self._model = model

    @staticmethod
    def _payload(messages: list[Message], system: str | None) -> list[dict]:
        # OpenAI takes the system prompt as the first message.
        out = [{"role": "system", "content": system}] if system else []
        out += [{"role": m.role, "content": m.content} for m in messages]
        return out

    async def generate(self, messages: list[Message], system: str | None = None) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model, messages=self._payload(messages, system)
        )
        return resp.choices[0].message.content or ""

    async def stream(self, messages: list[Message], system: str | None = None) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self._model, messages=self._payload(messages, system), stream=True
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

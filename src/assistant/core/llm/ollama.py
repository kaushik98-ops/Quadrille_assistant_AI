"""Local Ollama adapter, talking to its HTTP API directly."""
import json
from collections.abc import AsyncIterator

import httpx

from assistant.core.llm.base import LLMClient, Message


def _http() -> httpx.AsyncClient:
    # retries=2 re-tries failed connections only (for example while Ollama is starting up).
    return httpx.AsyncClient(timeout=120, transport=httpx.AsyncHTTPTransport(retries=2))


class OllamaClient(LLMClient):
    def __init__(self, base_url: str, model: str):
        self._base_url = base_url.rstrip("/")
        self._model = model

    def _body(self, messages: list[Message], system: str | None, stream: bool) -> dict:
        msgs = [{"role": "system", "content": system}] if system else []
        msgs += [{"role": m.role, "content": m.content} for m in messages]
        return {"model": self._model, "messages": msgs, "stream": stream}

    async def generate(self, messages: list[Message], system: str | None = None) -> str:
        async with _http() as http:
            resp = await http.post(f"{self._base_url}/api/chat", json=self._body(messages, system, False))
            resp.raise_for_status()
            return resp.json()["message"]["content"]

    async def stream(self, messages: list[Message], system: str | None = None) -> AsyncIterator[str]:
        async with _http() as http, http.stream(
            "POST", f"{self._base_url}/api/chat", json=self._body(messages, system, True)
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if data.get("message", {}).get("content"):
                    yield data["message"]["content"]

import asyncio
import json
import logging

import pytest

from assistant.core.config import Settings
from assistant.core.errors import ConfigError
from assistant.core.llm.base import LLMClient, Message
from assistant.core.llm.factory import get_llm_client
from assistant.core.logging import JsonFormatter, new_request_id
from assistant.core.memory import ConversationMemory


def test_factory_requires_api_key():
    cfg = Settings(_env_file=None, llm_provider="anthropic", anthropic_api_key="")
    with pytest.raises(ConfigError):
        get_llm_client(cfg)


def test_factory_returns_swappable_clients():
    a = get_llm_client(Settings(_env_file=None, llm_provider="anthropic", anthropic_api_key="x"))
    o = get_llm_client(Settings(_env_file=None, llm_provider="ollama"))
    assert isinstance(a, LLMClient) and isinstance(o, LLMClient)
    assert type(a) is not type(o)


def test_memory_roundtrip_and_limit(tmp_path):
    mem = ConversationMemory(str(tmp_path / "t.db"))
    for i in range(5):
        mem.add("c1", "user", f"msg{i}")
    mem.add("c2", "user", "other")

    last3 = mem.history("c1", limit=3)
    assert [m.content for m in last3] == ["msg2", "msg3", "msg4"]  # oldest first
    assert [m.content for m in mem.history("c2")] == ["other"]  # isolated per conversation


def test_log_line_has_request_id():
    rid = new_request_id()
    record = logging.LogRecord("t", logging.INFO, "", 0, "hello", None, None)
    line = json.loads(JsonFormatter().format(record))
    assert line["request_id"] == rid and line["msg"] == "hello"


def test_fake_client_satisfies_interface():
    class Fake(LLMClient):
        async def generate(self, messages, system=None):
            return "ok"

        async def stream(self, messages, system=None):
            yield "o"
            yield "k"

    async def run():
        f = Fake()
        chunks = [c async for c in f.stream([Message("user", "hi")])]
        return await f.generate([]), "".join(chunks)

    assert asyncio.run(run()) == ("ok", "ok")

import asyncio

import pytest

from assistant.chat.service import ChatService
from assistant.core.llm.base import LLMClient, Message
from assistant.core.memory import ConversationMemory


class FakeLLM(LLMClient):
    """Records what it was sent and streams a fixed reply."""

    def __init__(self, reply="hello there", fail_after=None):
        self.reply = reply
        self.fail_after = fail_after
        self.last_messages: list[Message] = []
        self.last_system: str | None = None

    async def generate(self, messages, system=None):
        return self.reply

    async def stream(self, messages, system=None):
        self.last_messages, self.last_system = messages, system
        for i, word in enumerate(self.reply.split(" ")):
            if self.fail_after is not None and i >= self.fail_after:
                raise RuntimeError("provider exploded")
            yield word + " "


def make(tmp_path, llm=None, **kw):
    mem = ConversationMemory(str(tmp_path / "t.db"))
    return ChatService(llm or FakeLLM(), mem, **kw), mem


def test_reply_is_saved_and_returned(tmp_path):
    svc, mem = make(tmp_path)
    text = asyncio.run(svc.reply("c1", "hi"))
    assert text.strip() == "hello there"
    saved = mem.history("c1")
    assert [(m.role, m.content.strip()) for m in saved] == [
        ("user", "hi"),
        ("assistant", "hello there"),
    ]


def test_history_is_sent_on_next_turn(tmp_path):
    llm = FakeLLM()
    svc, _ = make(tmp_path, llm)
    asyncio.run(svc.reply("c1", "first"))
    asyncio.run(svc.reply("c1", "second"))
    roles = [m.role for m in llm.last_messages]
    assert roles == ["user", "assistant", "user"]
    assert llm.last_messages[-1].content == "second"


def test_failed_stream_saves_nothing(tmp_path):
    svc, mem = make(tmp_path, FakeLLM(reply="a b c d", fail_after=2))
    with pytest.raises(RuntimeError):
        asyncio.run(svc.reply("c1", "hi"))
    assert mem.history("c1") == []


def test_window_trims_and_starts_with_user(tmp_path):
    llm = FakeLLM()
    svc, mem = make(tmp_path, llm, max_history_messages=3)
    # 4 saved messages; the last 3 are assistant, user, assistant.
    for role, text in [("user", "u1"), ("assistant", "a1"), ("user", "u2"), ("assistant", "a2")]:
        mem.add("c1", role, text)
    asyncio.run(svc.reply("c1", "u3"))
    # Window starts on a user message, so the leading assistant message was dropped.
    assert [m.content for m in llm.last_messages] == ["u2", "a2", "u3"]


def test_char_budget_drops_oldest(tmp_path):
    llm = FakeLLM()
    svc, mem = make(tmp_path, llm, max_history_chars=10)
    mem.add("c1", "user", "x" * 8)
    mem.add("c1", "assistant", "y" * 8)
    mem.add("c1", "user", "z" * 4)
    mem.add("c1", "assistant", "w" * 4)
    asyncio.run(svc.reply("c1", "new"))
    contents = [m.content for m in llm.last_messages]
    assert "x" * 8 not in contents and contents[-1] == "new"


def test_system_extra_is_included(tmp_path):
    llm = FakeLLM()
    svc, _ = make(tmp_path, llm)
    asyncio.run(svc.reply("c1", "hi", system_extra="EXTRA CONTEXT"))
    assert "EXTRA CONTEXT" in llm.last_system and "personal assistant" in llm.last_system


def test_conversations_are_isolated(tmp_path):
    llm = FakeLLM()
    svc, _ = make(tmp_path, llm)
    asyncio.run(svc.reply("a", "from a"))
    asyncio.run(svc.reply("b", "from b"))
    assert [m.content for m in llm.last_messages] == ["from b"]

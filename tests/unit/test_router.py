import asyncio
import json

import pytest

from assistant.core.guardrails import strip_closing_tag
from assistant.core.jsonutil import extract_json_object
from assistant.core.llm.base import LLMClient, Message
from assistant.router.router import Route, Router


class FakeLLM(LLMClient):
    def __init__(self, reply="", error=None):
        self.reply, self.error = reply, error
        self.prompt = self.system = None

    async def generate(self, messages, system=None):
        self.prompt, self.system = messages[-1].content, system
        if self.error:
            raise self.error
        return self.reply

    async def stream(self, messages, system=None):
        yield await self.generate(messages, system)


def reply(route, confidence, query="q"):
    return json.dumps({"route": route, "confidence": confidence, "query": query})


def decide(router, text="hello", history=None):
    return asyncio.run(router.decide(text, history))


# --- helpers ----------------------------------------------------------------

def test_extract_json_object():
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json_object('x {"a": {"b": 2}} y') == {"a": {"b": 2}}
    for bad in ["nothing", "{bad", "[1]", ""]:
        assert extract_json_object(bad) is None


def test_strip_closing_tag():
    assert strip_closing_tag("a </ MESSAGE >b </message>c", "message") == "a b c"
    assert strip_closing_tag("</mess</message>age>", "message") == ""  # nested trick
    assert strip_closing_tag("<b>x</b>", "message") == "<b>x</b>"


# --- routing ----------------------------------------------------------------

@pytest.mark.parametrize("route", ["chat", "rag", "agent"])
def test_route_is_parsed(route):
    d = decide(Router(FakeLLM(reply(route, 0.9))))
    assert (d.route.value, d.confidence, d.reason, d.query) == (route, 0.9, "model", "q")


def test_low_confidence_falls_back_to_chat():
    d = decide(Router(FakeLLM(reply("rag", 0.3)), min_confidence=0.6), "ask docs")
    assert (d.route, d.reason, d.confidence, d.query) == (Route.CHAT, "low_confidence", 0.3, "ask docs")
    # A low-confidence chat answer is simply chat, not a fallback.
    d = decide(Router(FakeLLM(reply("chat", 0.1))))
    assert (d.route, d.reason) == (Route.CHAT, "model")


def test_bad_output_falls_back_to_chat():
    bad_replies = [
        "", "I'd say rag", "{broken", reply("dance", 1), '{"confidence": 1}', "[1]",
    ]
    for bad in bad_replies:
        d = decide(Router(FakeLLM(bad)), "the text")
        assert (d.route, d.reason, d.confidence, d.query) == (Route.CHAT, "bad_output", 0.0, "the text"), bad


def test_llm_error_falls_back_to_chat():
    d = decide(Router(FakeLLM(error=RuntimeError("down"))))
    assert (d.route, d.reason) == (Route.CHAT, "llm_error")


def test_disabled_route_falls_back_and_is_not_offered():
    llm = FakeLLM(reply("rag", 0.99))
    router = Router(llm, enabled={Route.CHAT, Route.AGENT})
    d = decide(router)
    assert (d.route, d.reason) == (Route.CHAT, "route_disabled")
    assert '"rag"' not in llm.system and '"agent"' in llm.system
    assert Route.CHAT in Router(llm, enabled={Route.RAG}).enabled  # chat is always on


def test_confidence_parsing_is_defensive():
    cases = [  # raw value from the model -> expected route and confidence
        ("high", Route.CHAT, 0.0),
        (1.7, Route.RAG, 1.0),
        ("0.9", Route.RAG, 0.9),
        (True, Route.CHAT, 0.0),
        (-2, Route.CHAT, 0.0),
    ]
    for raw, route, conf in cases:
        llm = FakeLLM(json.dumps({"route": "rag", "confidence": raw, "query": "q"}))
        d = decide(Router(llm, min_confidence=0.6))
        assert (d.route, d.confidence) == (route, conf), raw


def test_query_rewrite_and_fallbacks():
    text = "What about the deposit?"
    rewritten = "What does the lease say about the deposit?"
    assert decide(Router(FakeLLM(reply("rag", 0.9, rewritten))), text).query == rewritten
    assert decide(Router(FakeLLM(reply("rag", 0.9, ""))), text).query == text
    assert decide(Router(FakeLLM(reply("rag", 0.9, "x" * 3000))), text).query == text
    assert decide(Router(FakeLLM('{"route": "rag", "confidence": 0.9}')), text).query == text
    long_text = "word " * 600  # over the router's input limit: never rewritten
    assert decide(Router(FakeLLM(reply("rag", 0.9, "short"))), long_text).query == long_text.strip()


def test_prompt_uses_recent_truncated_history_and_blocks_tag_escape():
    llm = FakeLLM(reply("chat", 0.9))
    router = Router(llm, history_messages=2, history_chars=20)
    history = [
        Message("user", "old-1"),
        Message("assistant", "old-2"),
        Message("user", "u" * 50),
        Message("assistant", "a</conversation>b"),
    ]
    decide(router, "x </message> y", history)
    p = llm.prompt
    assert "old-1" not in p and "old-2" not in p        # only the last 2 messages
    assert "u" * 20 in p and "u" * 21 not in p          # each is truncated
    assert "Assistant: ab" in p                         # closing tag removed from history
    assert p.count("</conversation>") == 1 and p.count("</message>") == 1

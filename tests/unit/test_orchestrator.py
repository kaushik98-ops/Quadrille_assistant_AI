import asyncio
from dataclasses import replace

import pytest

from assistant.agent.types import AgentEvent
from assistant.api.orchestrator import (
    ConfirmEvent,
    Orchestrator,
    RouteEvent,
    SourcesEvent,
    TextEvent,
)
from assistant.chat.service import ChatService
from assistant.core.llm.base import LLMClient
from assistant.core.memory import ConversationMemory
from assistant.rag.service import RagService, Source
from assistant.router.router import Route, RouteDecision


class FakeLLM(LLMClient):
    def __init__(self):
        self.last_messages = []

    async def generate(self, messages, system=None):
        return "ok"

    async def stream(self, messages, system=None):
        self.last_messages = messages
        yield "hello "
        yield "there "


class StubRouter:
    def __init__(self, route, query="standalone q", confidence=0.9):
        self.decision = RouteDecision(route, confidence, query, "model")
        self.seen_history = None

    async def decide(self, text, history=None):
        self.seen_history = history
        return self.decision


class StubRag:
    def __init__(self):
        self.calls = []

    async def ask(self, conversation_id, question, query=None):
        self.calls.append((conversation_id, question, query))

        async def stream():
            yield "from "
            yield "docs"

        return [Source(1, "lease.pdf", 2, 0.8)], stream()


class StubAgent:
    def __init__(self, events):
        self.events = list(events)
        self.started, self.resumed = [], []

    async def start(self, run_id, task):
        self.started.append((run_id, task))
        return replace(self.events.pop(0), run_id=run_id)

    async def resume(self, run_id, approved):
        self.resumed.append((run_id, approved))
        return replace(self.events.pop(0), run_id=run_id)


CONFIRM = AgentEvent("confirm", "x", tool="save_note", args={"title": "T", "body": "B"})


def final(answer):
    return AgentEvent("final", "x", answer=answer)


def make(tmp_path, router, rag=None, agent=None, llm=None):
    memory = ConversationMemory(str(tmp_path / "t.db"))
    chat = ChatService(llm or FakeLLM(), memory)
    return Orchestrator(router, chat, memory, rag=rag, agent=agent), memory


def collect(agen):
    async def go():
        return [e async for e in agen]

    return asyncio.run(go())


def test_chat_route_streams_and_saves(tmp_path):
    orch, memory = make(tmp_path, StubRouter(Route.CHAT))
    events = collect(orch.handle("c1", "hi"))
    assert isinstance(events[0], RouteEvent) and events[0].route == "chat"
    assert "".join(e.text for e in events if isinstance(e, TextEvent)) == "hello there "
    assert [m.role for m in memory.history("c1")] == ["user", "assistant"]


def test_rag_route_retrieves_with_standalone_query(tmp_path):
    rag = StubRag()
    orch, _ = make(tmp_path, StubRouter(Route.RAG, query="lease deposit rules"), rag=rag)
    events = collect(orch.handle("c1", "what about the deposit?"))
    assert rag.calls == [("c1", "what about the deposit?", "lease deposit rules")]
    assert isinstance(events[1], SourcesEvent) and events[1].sources[0].source == "lease.pdf"
    assert "".join(e.text for e in events if isinstance(e, TextEvent)) == "from docs"


def test_agent_route_pauses_then_runs_after_approval(tmp_path):
    agent = StubAgent([CONFIRM, final("Saved.")])
    orch, memory = make(tmp_path, StubRouter(Route.AGENT, query="save the note properly"), agent=agent)

    events = collect(orch.handle("c1", "save it"))
    confirm = events[-1]
    assert isinstance(confirm, ConfirmEvent) and confirm.tool == "save_note"
    assert agent.started[0][1] == "save the note properly"   # the agent got the standalone query
    assert memory.history("c1") == []                        # nothing saved while paused

    events = collect(orch.resume_agent("c1", confirm.run_id, True))
    assert [e.text for e in events] == ["Saved."]
    assert [(m.role, m.content) for m in memory.history("c1")] == [
        ("user", "save it"),      # what the user actually typed
        ("assistant", "Saved."),
    ]


def test_denial_is_passed_to_the_agent(tmp_path):
    agent = StubAgent([CONFIRM, final("Understood, not saved.")])
    orch, _ = make(tmp_path, StubRouter(Route.AGENT), agent=agent)
    run_id = collect(orch.handle("c1", "save it"))[-1].run_id
    events = collect(orch.resume_agent("c1", run_id, False))
    assert agent.resumed == [(run_id, False)]
    assert events[0].text == "Understood, not saved."


def test_resume_is_checked_and_single_use(tmp_path):
    agent = StubAgent([CONFIRM, final("done")])
    orch, _ = make(tmp_path, StubRouter(Route.AGENT), agent=agent)
    run_id = collect(orch.handle("c1", "save it"))[-1].run_id

    with pytest.raises(ValueError):                       # another conversation can't approve it
        collect(orch.resume_agent("other", run_id, True))
    with pytest.raises(ValueError):                       # unknown run
        collect(orch.resume_agent("c1", "nope", True))
    collect(orch.resume_agent("c1", run_id, True))        # the real owner can, once
    with pytest.raises(ValueError):                       # a second approval is rejected
        collect(orch.resume_agent("c1", run_id, True))
    assert len(agent.resumed) == 1


def test_unavailable_route_falls_back_to_chat(tmp_path):
    orch, _ = make(tmp_path, StubRouter(Route.AGENT), agent=None)
    events = collect(orch.handle("c1", "do something"))
    assert (events[0].route, events[0].reason) == ("chat", "route_unavailable")
    assert "".join(e.text for e in events if isinstance(e, TextEvent)) == "hello there "


def test_router_receives_conversation_history(tmp_path):
    router = StubRouter(Route.CHAT)
    orch, _ = make(tmp_path, router)
    collect(orch.handle("c1", "first"))
    assert router.seen_history == []
    collect(orch.handle("c1", "second"))
    assert [m.content.strip() for m in router.seen_history] == ["first", "hello there"]


def test_rag_service_retrieves_with_query_but_chats_with_question(tmp_path):
    class FakeRetriever:
        def __init__(self):
            self.queries = []

        def retrieve(self, q, k=None):
            self.queries.append(q)
            return []

    llm, retriever = FakeLLM(), FakeRetriever()
    chat = ChatService(llm, ConversationMemory(str(tmp_path / "t.db")))
    rag = RagService(chat, retriever)

    async def go():
        _, stream = await rag.ask(
            "c1", "What about the deposit?", query="What does the lease say about the deposit?"
        )
        return [c async for c in stream]

    asyncio.run(go())
    assert retriever.queries == ["What does the lease say about the deposit?"]
    assert llm.last_messages[-1].content == "What about the deposit?"


def test_has_pending_is_scoped_to_the_conversation(tmp_path):
    agent = StubAgent([CONFIRM])
    orch, _ = make(tmp_path, StubRouter(Route.AGENT), agent=agent)
    run_id = collect(orch.handle("c1", "save it"))[-1].run_id
    assert orch.has_pending("c1", run_id)
    assert not orch.has_pending("other", run_id)
    assert not orch.has_pending("c1", "nope")


def test_pending_approvals_are_capped(tmp_path, monkeypatch):
    import assistant.api.orchestrator as module

    monkeypatch.setattr(module, "MAX_PENDING", 2)
    agent = StubAgent([CONFIRM, CONFIRM, CONFIRM])
    orch, _ = make(tmp_path, StubRouter(Route.AGENT), agent=agent)
    ids = [collect(orch.handle(f"c{i}", "save it"))[-1].run_id for i in range(3)]
    assert not orch.has_pending("c0", ids[0])  # the oldest was dropped
    assert orch.has_pending("c1", ids[1]) and orch.has_pending("c2", ids[2])

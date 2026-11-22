"""Orchestrator: routes each message to chat, RAG, or the agent and streams back events.

This is the only place that knows about all the capabilities. The terminal script and the
API endpoints both just consume its events.
"""
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace

from assistant.agent.runtime import AgentRuntime
from assistant.agent.types import AgentEvent
from assistant.chat.service import ChatService
from assistant.core.memory import ConversationMemory
from assistant.rag.service import RagService, Source
from assistant.router.router import Route, RouteDecision, Router

ROUTER_HISTORY = 10  # messages fetched for the router (it uses the most recent few)
MAX_PENDING = 100    # unanswered approvals kept in memory; the oldest is dropped beyond this


@dataclass(frozen=True)
class RouteEvent:
    route: str
    confidence: float
    query: str
    reason: str


@dataclass(frozen=True)
class SourcesEvent:
    sources: list[Source]


@dataclass(frozen=True)
class TextEvent:
    text: str


@dataclass(frozen=True)
class ConfirmEvent:
    run_id: str
    tool: str
    args: dict


Event = RouteEvent | SourcesEvent | TextEvent | ConfirmEvent


class Orchestrator:
    def __init__(
        self,
        router: Router,
        chat: ChatService,
        memory: ConversationMemory,
        rag: RagService | None = None,
        agent: AgentRuntime | None = None,
    ):
        self._router = router
        self._chat = chat
        self._memory = memory
        self._rag = rag
        self._agent = agent
        # run_id -> (conversation_id, original user text). In memory only, like the agent's checkpoints.
        self._pending: dict[str, tuple[str, str]] = {}

    def has_pending(self, conversation_id: str, run_id: str) -> bool:
        owner = self._pending.get(run_id)
        return owner is not None and owner[0] == conversation_id

    def _check_available(self, decision: RouteDecision) -> RouteDecision:
        missing = (decision.route is Route.RAG and self._rag is None) or (
            decision.route is Route.AGENT and self._agent is None
        )
        return replace(decision, route=Route.CHAT, reason="route_unavailable") if missing else decision

    async def handle(self, conversation_id: str, text: str) -> AsyncIterator[Event]:
        history = self._memory.history(conversation_id, limit=ROUTER_HISTORY)
        decision = self._check_available(await self._router.decide(text, history))
        yield RouteEvent(decision.route.value, decision.confidence, decision.query, decision.reason)

        if decision.route is Route.RAG:
            sources, stream = await self._rag.ask(conversation_id, text, query=decision.query)
            yield SourcesEvent(sources)
            async for chunk in stream:
                yield TextEvent(chunk)
        elif decision.route is Route.AGENT:
            run_id = uuid.uuid4().hex[:12]
            event = await self._agent.start(run_id, decision.query)
            async for out in self._after_agent(conversation_id, text, event):
                yield out
        else:
            async for chunk in self._chat.stream_reply(conversation_id, text):
                yield TextEvent(chunk)

    async def resume_agent(
        self, conversation_id: str, run_id: str, approved: bool
    ) -> AsyncIterator[Event]:
        """Answer a pending approval. Only the conversation that started the run can answer it."""
        if self._agent is None:
            raise ValueError("The agent is not available.")
        owner = self._pending.get(run_id)
        if owner is None or owner[0] != conversation_id:
            raise ValueError("No pending action for this conversation and run.")
        # Remove it before resuming, so a double click can't approve (and run) the action twice.
        _, user_text = self._pending.pop(run_id)
        event = await self._agent.resume(run_id, approved)
        async for out in self._after_agent(conversation_id, user_text, event):
            yield out

    async def _after_agent(
        self, conversation_id: str, user_text: str, event: AgentEvent
    ) -> AsyncIterator[Event]:
        if event.kind == "confirm":
            while len(self._pending) >= MAX_PENDING:
                self._pending.pop(next(iter(self._pending)))  # drop the oldest
            self._pending[event.run_id] = (conversation_id, user_text)
            yield ConfirmEvent(event.run_id, event.tool, event.args)
            return
        # Finished: record the exchange so chat and the router can see it later.
        self._memory.add(conversation_id, "user", user_text)
        self._memory.add(conversation_id, "assistant", event.answer)
        yield TextEvent(event.answer)

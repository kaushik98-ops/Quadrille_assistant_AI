"""Builds the real services at startup. Tests pass their own factory instead."""
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

from assistant.agent.mcp_client import MCPToolClient
from assistant.agent.runtime import AgentRuntime
from assistant.api.documents import Documents, RagDocuments
from assistant.api.orchestrator import Orchestrator
from assistant.chat.service import ChatService
from assistant.core.config import Settings
from assistant.core.llm.factory import get_llm_client
from assistant.core.llm.logged import LoggedLLM
from assistant.core.logging import get_logger
from assistant.core.memory import ConversationMemory
from assistant.rag.embeddings import get_embedder
from assistant.rag.retriever import Retriever
from assistant.rag.service import RagService
from assistant.rag.store import build_store
from assistant.router.router import Route, Router

log = get_logger(__name__)


@dataclass(frozen=True)
class Services:
    orchestrator: Orchestrator
    documents: Documents


ServicesFactory = Callable[[Settings], AbstractAsyncContextManager[Services]]


@asynccontextmanager
async def build_services(cfg: Settings) -> AsyncIterator[Services]:
    # A missing provider key fails startup loudly, on purpose.
    llm = LoggedLLM(get_llm_client(cfg), f"{cfg.llm_provider}:{cfg.llm_model or 'default'}")
    memory = ConversationMemory(cfg.db_path)
    chat = ChatService(llm, memory)
    embedder, store = get_embedder(cfg), build_store(cfg)
    rag = RagService(chat, Retriever(embedder, store, cfg.rag_top_k, cfg.rag_min_score))

    async with AsyncExitStack() as stack:
        agent = None
        try:
            tools = await stack.enter_async_context(MCPToolClient.from_settings(cfg))
            agent = await AgentRuntime.create(
                llm, tools, cfg.agent_max_steps, cfg.agent_max_observation_chars
            )
        except Exception:
            # The app still works without the agent: those requests are answered by chat.
            log.exception("agent_unavailable")

        enabled = {Route.CHAT, Route.RAG} | ({Route.AGENT} if agent else set())
        router = Router(
            llm,
            enabled=enabled,
            min_confidence=cfg.router_min_confidence,
            history_messages=cfg.router_history_messages,
        )
        orchestrator = Orchestrator(router, chat, memory, rag=rag, agent=agent)
        yield Services(orchestrator, RagDocuments(embedder, store, cfg.pdf_dir))

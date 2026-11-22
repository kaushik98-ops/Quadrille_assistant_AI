"""The whole product in a terminal: router -> chat / RAG / agent.

  python scripts/assistant_cli.py [conversation_id]
"""
import asyncio
import json
import sys

from assistant.agent.mcp_client import MCPToolClient
from assistant.agent.runtime import AgentRuntime
from assistant.api.orchestrator import (
    ConfirmEvent,
    Orchestrator,
    RouteEvent,
    SourcesEvent,
    TextEvent,
)
from assistant.chat.service import ChatService
from assistant.core.config import settings
from assistant.core.llm.factory import get_llm_client
from assistant.core.logging import new_request_id, setup_logging
from assistant.core.memory import ConversationMemory
from assistant.rag.retriever import build_retriever
from assistant.rag.service import RagService
from assistant.router.router import Route, Router


async def run_turn(orch: Orchestrator, conversation_id: str, text: str) -> None:
    events = orch.handle(conversation_id, text)
    while True:
        pending, sources = None, None
        async for e in events:
            if isinstance(e, RouteEvent):
                print(f"[route: {e.route} | confidence {e.confidence:.2f} | {e.reason}]")
                if e.query != text:
                    print(f"[standalone query: {e.query}]")
                print("bot> ", end="", flush=True)
            elif isinstance(e, SourcesEvent):
                sources = e.sources
            elif isinstance(e, TextEvent):
                print(e.text, end="", flush=True)
            elif isinstance(e, ConfirmEvent):
                pending = e
        print()
        if sources is not None:
            print("Sources:" if sources else "Sources: none above the score threshold")
            for s in sources:
                print(f"  [{s.n}] {s.source}, page {s.page} (score {s.score:.2f})")
        if pending is None:
            return
        print(f"\nThe agent wants to run: {pending.tool}")
        print(json.dumps(pending.args, indent=2, ensure_ascii=False))
        approved = input("Approve? [y/N] ").strip().lower() in ("y", "yes")
        events = orch.resume_agent(conversation_id, pending.run_id, approved)
        print("bot> ", end="", flush=True)


async def main(conversation_id: str) -> None:
    setup_logging("WARNING")
    llm = get_llm_client()
    memory = ConversationMemory(settings.db_path)
    chat = ChatService(llm, memory)
    rag = RagService(chat, build_retriever())

    async with MCPToolClient.from_settings() as tools:
        agent = await AgentRuntime.create(
            llm, tools, settings.agent_max_steps, settings.agent_max_observation_chars
        )
        router = Router(
            llm,
            enabled={Route.CHAT, Route.RAG, Route.AGENT},
            min_confidence=settings.router_min_confidence,
            history_messages=settings.router_history_messages,
        )
        orch = Orchestrator(router, chat, memory, rag=rag, agent=agent)

        print(f"Conversation '{conversation_id}'. Ctrl+C to quit.\n")
        while True:
            try:
                text = input("you> ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                return
            if text:
                new_request_id()
                await run_turn(orch, conversation_id, text)
                print()


asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "assistant-cli"))

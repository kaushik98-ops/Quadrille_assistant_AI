"""Standalone RAG tool. No API server needed.

  python scripts/rag_cli.py ingest             # ingest every PDF in PDF_DIR
  python scripts/rag_cli.py sources            # list what is stored
  python scripts/rag_cli.py ask "question"     # answer with citations
"""
import asyncio
import sys

from assistant.chat.service import ChatService
from assistant.core.config import settings
from assistant.core.llm.factory import get_llm_client
from assistant.core.logging import new_request_id, setup_logging
from assistant.core.memory import ConversationMemory
from assistant.rag.embeddings import get_embedder
from assistant.rag.ingest import ingest_directory
from assistant.rag.retriever import Retriever
from assistant.rag.service import RagService
from assistant.rag.store import build_store


async def ask(question: str) -> None:
    store = build_store()
    retriever = Retriever(get_embedder(), store, settings.rag_top_k, settings.rag_min_score)
    chat = ChatService(get_llm_client(), ConversationMemory(settings.db_path))
    new_request_id()
    sources, stream = await RagService(chat, retriever).ask("rag-cli", question)
    async for chunk in stream:
        print(chunk, end="", flush=True)
    print("\n\nSources:")
    for s in sources:
        print(f"  [{s.n}] {s.source}, page {s.page} (score {s.score:.2f})")
    if not sources:
        print("  none above the score threshold")


def main() -> None:
    setup_logging("WARNING")
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "ingest":
        counts = ingest_directory(settings.pdf_dir, get_embedder(), build_store())
        for name, n in counts.items():
            print(f"{name}: {n} chunks")
        if not counts:
            print(f"No PDFs found in {settings.pdf_dir}")
    elif cmd == "sources":
        for name, n in build_store().sources().items():
            print(f"{name}: {n} chunks")
    elif cmd == "ask" and len(sys.argv) > 2:
        asyncio.run(ask(" ".join(sys.argv[2:])))
    else:
        print(__doc__)


main()

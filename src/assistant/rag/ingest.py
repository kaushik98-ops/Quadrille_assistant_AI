"""Ingestion pipeline: load, chunk, embed, store."""
from pathlib import Path

from assistant.core.logging import get_logger
from assistant.rag.chunking import chunk_pages
from assistant.rag.embeddings import Embedder
from assistant.rag.loader import load_pdf
from assistant.rag.store import VectorStore

log = get_logger(__name__)


def ingest_pages(
    source: str, pages: list[tuple[int, str]], embedder: Embedder, store: VectorStore
) -> int:
    """Replace everything stored for `source` with these pages. Returns chunk count."""
    chunks = chunk_pages(source, pages)
    # Embed BEFORE deleting, so a failed embedding call can't wipe the previous version.
    vectors = embedder.embed([c.text for c in chunks]) if chunks else []
    store.delete_source(source)  # drop stale chunks from an earlier version
    if chunks:
        store.add(chunks, vectors)
    log.info("ingested", extra={"fields": {"source": source, "chunks": len(chunks)}})
    return len(chunks)


def ingest_pdf(path: str | Path, embedder: Embedder, store: VectorStore) -> int:
    path = Path(path)
    return ingest_pages(path.name, load_pdf(path), embedder, store)


def ingest_directory(directory: str | Path, embedder: Embedder, store: VectorStore) -> dict[str, int]:
    return {p.name: ingest_pdf(p, embedder, store) for p in sorted(Path(directory).glob("*.pdf"))}

"""Document library: save an uploaded PDF and index it, safely."""
import asyncio
import os
from pathlib import Path
from typing import Protocol

from assistant.core.logging import get_logger
from assistant.rag.embeddings import Embedder
from assistant.rag.ingest import ingest_pages
from assistant.rag.loader import load_pdf
from assistant.rag.store import VectorStore

log = get_logger(__name__)


class DocumentError(ValueError):
    """The upload can't be used. The message is safe to show to the user."""


class Documents(Protocol):
    async def add_pdf(self, name: str, data: bytes) -> int: ...

    async def sources(self) -> dict[str, int]: ...


class RagDocuments:
    def __init__(self, embedder: Embedder, store: VectorStore, pdf_dir: str | Path):
        self._embedder = embedder
        self._store = store
        self._dir = Path(pdf_dir)
        self._lock = asyncio.Lock()  # one ingest at a time, so re-uploads of a file can't interleave

    async def add_pdf(self, name: str, data: bytes) -> int:
        """Index the PDF, then keep it. Returns the number of chunks stored."""
        async with self._lock:
            return await asyncio.to_thread(self._add_sync, name, data)

    async def sources(self) -> dict[str, int]:
        return await asyncio.to_thread(self._store.sources)

    def _add_sync(self, name: str, data: bytes) -> int:
        self._dir.mkdir(parents=True, exist_ok=True)
        final = self._dir / name
        tmp = self._dir / f"{name}.part"
        tmp.write_bytes(data)
        try:
            try:
                pages = load_pdf(tmp)
            except Exception as exc:  # pypdf raises a variety of errors on broken files
                log.warning("pdf_unreadable", extra={"fields": {"file": name, "error": str(exc)}})
                raise DocumentError("Could not read this file as a PDF.") from exc
            if not pages:
                raise DocumentError(
                    "No text could be extracted. Scanned PDFs need OCR, which is not supported yet."
                )
            chunks = ingest_pages(name, pages, self._embedder, self._store)
            os.replace(tmp, final)  # keep the file only after it was indexed successfully
            return chunks
        finally:
            tmp.unlink(missing_ok=True)

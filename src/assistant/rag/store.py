"""Chroma wrapper. The rest of the project only sees this small interface."""
from collections import Counter
from dataclasses import dataclass

from assistant.core.config import Settings, settings
from assistant.rag.chunking import Chunk


@dataclass(frozen=True)
class Hit:
    text: str
    source: str
    page: int
    score: float  # cosine similarity: higher is more similar


class VectorStore:
    def __init__(self, path: str, collection_name: str, host: str = "", port: int = 8000) -> None:
        import chromadb

        # With a host, talk to a Chroma server (used in Docker). Otherwise keep files on disk.
        if host:
            self._client = chromadb.HttpClient(host=host, port=port)
        else:
            self._client = chromadb.PersistentClient(path=path)
        # embedding_function=None: we always pass our own vectors.
        self._col = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        for i in range(0, len(chunks), 500):  # keep each upsert a safe size
            batch = chunks[i : i + 500]
            self._col.upsert(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                embeddings=embeddings[i : i + 500],
                metadatas=[{"source": c.source, "page": c.page} for c in batch],
            )

    def delete_source(self, source: str) -> None:
        self._col.delete(where={"source": source})

    def query(self, embedding: list[float], k: int) -> list[Hit]:
        if self._col.count() == 0:
            return []
        res = self._col.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append(Hit(text=text, source=meta["source"], page=int(meta["page"]), score=1.0 - dist))
        return hits

    def sources(self) -> dict[str, int]:
        """Chunk count per source file."""
        metas = self._col.get(include=["metadatas"])["metadatas"] or []
        return dict(Counter(m["source"] for m in metas))


def build_store(cfg: Settings = settings) -> VectorStore:
    return VectorStore(
        cfg.chroma_path, f"docs_{cfg.embedding_provider}", cfg.chroma_host, cfg.chroma_port
    )

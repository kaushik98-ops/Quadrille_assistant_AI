"""Query to top-k chunks. This is the function the MCP search_documents tool wraps."""
from assistant.core.config import Settings, settings
from assistant.rag.embeddings import Embedder, get_embedder
from assistant.rag.store import Hit, VectorStore, build_store


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, top_k: int = 4, min_score: float = 0.25):
        self._embedder = embedder
        self._store = store
        self._top_k = top_k
        self._min_score = min_score

    def retrieve(self, query: str, k: int | None = None) -> list[Hit]:
        """Best matches first, with weak matches removed."""
        vector = self._embedder.embed([query])[0]
        hits = self._store.query(vector, k or self._top_k)
        return [h for h in hits if h.score >= self._min_score]


def build_retriever(cfg: Settings = settings) -> Retriever:
    return Retriever(get_embedder(cfg), build_store(cfg), cfg.rag_top_k, cfg.rag_min_score)

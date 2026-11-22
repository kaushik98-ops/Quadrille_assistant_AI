"""Embedding providers behind one small interface."""
from abc import ABC, abstractmethod

from assistant.core.config import Settings, settings
from assistant.core.errors import ConfigError


class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text."""


class LocalEmbedder(Embedder):
    """Chroma's bundled local model. No API key; downloads the model on first use."""

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._ef = DefaultEmbeddingFunction()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vec] for vec in self._ef(texts)]


class OpenAIEmbedder(Embedder):
    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, max_retries=3)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 64):  # batch to stay under request limits
            resp = self._client.embeddings.create(model=self._model, input=texts[i : i + 64])
            out.extend(d.embedding for d in resp.data)
        return out


def get_embedder(cfg: Settings = settings) -> Embedder:
    if cfg.embedding_provider == "openai":
        if not cfg.openai_api_key:
            raise ConfigError("OPENAI_API_KEY is not set (needed for OpenAI embeddings)")
        # Verify the default model name against OpenAI's current docs.
        return OpenAIEmbedder(cfg.openai_api_key, cfg.embedding_model or "text-embedding-3-small")
    return LocalEmbedder()

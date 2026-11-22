"""One typed settings object for the whole project, read from .env or the environment."""
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Docker (and Kubernetes) mount secrets as files in this folder. Each file is named after a
# field below (anthropic_api_key, openai_api_key, api_keys, agent_mcp_api_key).
_SECRETS_DIR = "/run/secrets"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        secrets_dir=_SECRETS_DIR if Path(_SECRETS_DIR).is_dir() else None,
    )

    llm_provider: Literal["anthropic", "openai", "ollama"] = "anthropic"
    llm_model: str | None = None  # None means use the provider default

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    db_path: str = "data/app.db"
    log_level: str = "INFO"

    embedding_provider: Literal["local", "openai"] = "local"
    embedding_model: str | None = None  # None means use the provider default
    chroma_path: str = "data/chroma"
    chroma_host: str = ""     # empty means embedded Chroma on disk; set to use a Chroma server
    chroma_port: int = 8000
    pdf_dir: str = "data/pdfs"
    rag_top_k: int = 4
    rag_min_score: float = 0.25

    sandbox_dir: str = "data/sandbox"

    agent_mcp_url: str = ""   # empty means spawn the MCP server as a subprocess (stdio)
    agent_mcp_api_key: str = ""
    agent_max_steps: int = 6
    agent_max_observation_chars: int = 4000

    router_min_confidence: float = 0.6
    router_history_messages: int = 6

    api_keys: str = ""        # comma-separated; the API refuses to start without one
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    rate_limit_per_minute: int = 30   # per API key; 0 turns it off
    max_upload_mb: int = 20

    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    # Never let a stray log line or traceback print the keys.
    def __repr__(self) -> str:
        return "Settings(<values hidden>)"

    __str__ = __repr__


settings = Settings()

"""Picks the LLM adapter from config. The rest of the app only calls get_llm_client()."""
from assistant.core.config import Settings, settings
from assistant.core.errors import ConfigError
from assistant.core.llm.base import LLMClient

# Verify these model names against each provider's current docs before relying on them.
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.1",
}


def get_llm_client(cfg: Settings = settings) -> LLMClient:
    provider = cfg.llm_provider
    model = cfg.llm_model or DEFAULT_MODELS[provider]

    if provider == "anthropic":
        if not cfg.anthropic_api_key:
            raise ConfigError("ANTHROPIC_API_KEY is not set")
        from assistant.core.llm.anthropic import AnthropicClient
        return AnthropicClient(cfg.anthropic_api_key, model)

    if provider == "openai":
        if not cfg.openai_api_key:
            raise ConfigError("OPENAI_API_KEY is not set")
        from assistant.core.llm.openai import OpenAIClient
        return OpenAIClient(cfg.openai_api_key, model)

    from assistant.core.llm.ollama import OllamaClient
    return OllamaClient(cfg.ollama_base_url, model)

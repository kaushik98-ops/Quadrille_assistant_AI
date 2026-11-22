"""Connects the agent to the MCP server. This is how the agent gets every tool."""
import asyncio
import os
import sys
from collections.abc import Mapping
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from assistant.agent.types import ToolInfo
from assistant.core.config import Settings, settings
from assistant.core.errors import AssistantError

# What the spawned MCP server may inherit. Everything else, including provider keys and
# API_KEYS, is left behind so a compromised tool can't read it from its environment.
SYSTEM_VARS = (
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR", "TEMP", "TMP", "SYSTEMROOT",
    "VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "HF_HOME", "XDG_CACHE_HOME",
)
APP_VARS = (
    "DB_PATH", "SANDBOX_DIR", "PDF_DIR", "CHROMA_PATH", "CHROMA_HOST", "CHROMA_PORT",
    "EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "RAG_TOP_K", "RAG_MIN_SCORE", "LOG_LEVEL",
)


def child_env(parent: Mapping[str, str], cfg: Settings = settings) -> dict[str, str]:
    """The environment for the MCP server subprocess: an allowlist, not a copy."""
    allowed = set(SYSTEM_VARS) | set(APP_VARS)
    if cfg.embedding_provider == "openai":
        allowed.add("OPENAI_API_KEY")  # only needed for OpenAI embeddings in search_documents
    return {k: v for k, v in parent.items() if k in allowed}


class MCPToolClient:
    """Async context manager. Use it as: `async with MCPToolClient(...) as tools:`.

    With no URL it spawns `python -m assistant.mcp_server` and talks over stdio.
    With a URL it connects to a running streamable-HTTP server.
    """

    def __init__(
        self,
        url: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 60.0,
        headers: dict[str, str] | None = None,
    ):
        self._url = url
        self._env = env
        self._timeout = timeout
        self._headers = headers
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    @classmethod
    def from_settings(cls, cfg: Settings = settings) -> "MCPToolClient":
        headers = {"Authorization": f"Bearer {cfg.agent_mcp_api_key}"} if cfg.agent_mcp_api_key else None
        return cls(url=cfg.agent_mcp_url or None, env=child_env(os.environ, cfg), headers=headers)

    async def __aenter__(self) -> "MCPToolClient":
        stack = AsyncExitStack()
        try:
            if self._url:
                read, write, _ = await stack.enter_async_context(
                    streamablehttp_client(self._url, headers=self._headers)
                )
            else:
                params = StdioServerParameters(
                    command=sys.executable,
                    args=["-m", "assistant.mcp_server"],
                    env=self._env if self._env is not None else child_env(os.environ),
                )
                read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except BaseException:
            await stack.aclose()
            raise
        self._stack, self._session = stack, session
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._stack:
            await self._stack.aclose()
        self._stack = self._session = None

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise AssistantError("MCPToolClient must be used inside 'async with'.")
        return self._session

    async def list_tools(self) -> list[ToolInfo]:
        resp = await self._require_session().list_tools()
        return [
            ToolInfo(
                name=t.name,
                description=t.description or "",
                schema=t.inputSchema or {},
                read_only_hint=t.annotations.readOnlyHint if t.annotations else None,
                open_world_hint=t.annotations.openWorldHint if t.annotations else None,
            )
            for t in resp.tools
        ]

    async def call_tool(self, name: str, args: dict) -> str:
        result = await asyncio.wait_for(
            self._require_session().call_tool(name, args), timeout=self._timeout
        )
        text = "\n".join(
            b.text for b in result.content if isinstance(getattr(b, "text", None), str)
        )
        return f"ERROR: {text}" if result.isError else text

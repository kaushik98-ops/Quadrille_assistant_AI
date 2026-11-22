"""Builds the MCP server from the tool registry. Tool logic lives in assistant.tools only."""
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from assistant.tools.base import Risk
from assistant.tools.builtin.documents import sources_json
from assistant.tools.registry import load_tools


def build_mcp_server(host: str = "127.0.0.1", port: int = 8000, *, mounted: bool = False) -> FastMCP:
    kwargs = {}
    if mounted:
        # Mounted inside our API, where the gateway already requires an API key on every request.
        # The SDK's Host-header check is meant for unauthenticated localhost servers and would
        # reject legitimate hostnames (Docker, cloud), so it is turned off here.
        try:
            from mcp.server.transport_security import TransportSecuritySettings

            kwargs["transport_security"] = TransportSecuritySettings(enable_dns_rebinding_protection=False)
        except ImportError:  # older SDKs have no such check to turn off
            pass
    mcp = FastMCP("assistant-platform", host=host, port=port, **kwargs)

    # Tools: registered from the shared registry.
    for spec in load_tools():
        mcp.add_tool(
            spec.fn,
            name=spec.name,
            description=spec.description,
            annotations=ToolAnnotations(
                readOnlyHint=spec.risk == Risk.READ_ONLY,
                destructiveHint=spec.risk == Risk.DANGEROUS,
                openWorldHint=spec.external,
            ),
        )

    # Resource: read-only data a client can load as context.
    @mcp.resource(
        "documents://sources",
        name="documents_sources",
        description="Ingested document files and how many chunks each has.",
        mime_type="application/json",
    )
    def documents_sources() -> str:
        return sources_json()

    # Prompt: a reusable template the client can offer to its user.
    @mcp.prompt()
    def summarize_document(filename: str) -> str:
        """Ask the model to summarize one ingested document."""
        return (
            f"Use the search_documents tool to find the main content of '{filename}', "
            "then write a short summary with the key points. Cite file and page for each point."
        )

    return mcp

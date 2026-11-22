"""Run the MCP server on its own:
  python -m assistant.mcp_server                       # stdio (for Claude Desktop etc.)
  python -m assistant.mcp_server --transport streamable-http --port 8000
"""
import argparse

from assistant.core.config import settings
from assistant.core.logging import setup_logging
from assistant.mcp_server.server import build_mcp_server


def main() -> None:
    parser = argparse.ArgumentParser(description="Assistant platform MCP server")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    setup_logging(settings.log_level)  # goes to stderr, so stdio stays clean
    build_mcp_server(args.host, args.port).run(transport=args.transport)


if __name__ == "__main__":
    main()

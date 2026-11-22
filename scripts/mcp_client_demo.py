"""Connects to the MCP server like any external client would.

  python scripts/mcp_client_demo.py                              # spawns the server over stdio
  python scripts/mcp_client_demo.py http://127.0.0.1:8000/mcp    # connects to a running HTTP server
                                                                 # (set MCP_API_KEY if it needs a key)
"""
import asyncio
import os
import sys
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client


@asynccontextmanager
async def connect(url: str | None):
    if url:
        key = os.environ.get("MCP_API_KEY")
        headers = {"Authorization": f"Bearer {key}"} if key else None
        async with streamablehttp_client(url, headers=headers) as (read, write, _):
            yield read, write
    else:
        params = StdioServerParameters(command=sys.executable, args=["-m", "assistant.mcp_server"])
        async with stdio_client(params) as (read, write):
            yield read, write


async def main(url: str | None) -> None:
    async with connect(url) as (read, write), ClientSession(read, write) as session:
        await session.initialize()

        print("TOOLS")
        for t in (await session.list_tools()).tools:
            read_only = t.annotations.readOnlyHint if t.annotations else None
            print(f"  {t.name:18} read_only={read_only}")

        print("\nPROMPTS:", [p.name for p in (await session.list_prompts()).prompts])
        print("RESOURCES:", [str(r.uri) for r in (await session.list_resources()).resources])

        print("\nCALL save_note")
        r = await session.call_tool("save_note", {"title": "MCP demo", "body": "Saved through MCP."})
        print(" ", r.content[0].text)

        print("CALL search_notes")
        r = await session.call_tool("search_notes", {"query": "MCP"})
        print(" ", r.content[0].text.replace("\n", "\n  "))

        print("CALL read_file (should fail: outside sandbox)")
        r = await session.call_tool("read_file", {"path": "../../etc/passwd"})
        print("  isError =", r.isError, "|", r.content[0].text)


asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else None))

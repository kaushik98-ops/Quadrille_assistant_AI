import asyncio

from assistant.core.config import settings
from assistant.mcp_server.server import build_mcp_server
from assistant.tools.builtin import notes

EXPECTED = {
    "search_documents", "read_file", "list_files", "web_search",
    "save_note", "list_notes", "search_notes",
}


def test_server_lists_all_tools_with_annotations():
    tools = asyncio.run(build_mcp_server().list_tools())
    by_name = {t.name: t for t in tools}
    assert set(by_name) == EXPECTED
    assert by_name["save_note"].annotations.readOnlyHint is False
    assert by_name["read_file"].annotations.readOnlyHint is True


def test_server_lists_prompt_and_resource():
    mcp = build_mcp_server()
    prompts = asyncio.run(mcp.list_prompts())
    resources = asyncio.run(mcp.list_resources())
    assert [p.name for p in prompts] == ["summarize_document"]
    assert [str(r.uri) for r in resources] == ["documents://sources"]


def test_tool_call_through_server_runs_the_tool(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "t.db"))
    asyncio.run(build_mcp_server().call_tool("save_note", {"title": "t", "body": "b"}))
    assert notes.list_notes() == "t"  # verified through the tool module, not the SDK's return shape

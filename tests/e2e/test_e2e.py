"""End-to-end: HTTP in, SSE out, with only the LLM and the embedding model replaced."""
import json
import math
import os
import re
import zlib
from contextlib import asynccontextmanager

from fastapi.testclient import TestClient

from assistant.agent.mcp_client import MCPToolClient, child_env
from assistant.agent.runtime import AgentRuntime
from assistant.api.documents import RagDocuments
from assistant.api.main import create_app
from assistant.api.orchestrator import Orchestrator
from assistant.api.services import Services
from assistant.chat.service import ChatService
from assistant.core.auth import key_id
from assistant.core.config import Settings, settings
from assistant.core.llm.base import LLMClient
from assistant.core.llm.logged import LoggedLLM
from assistant.core.memory import ConversationMemory
from assistant.rag.embeddings import Embedder
from assistant.rag.retriever import Retriever
from assistant.rag.service import RagService
from assistant.rag.store import VectorStore
from assistant.router.router import Route, Router
from assistant.tools.builtin import notes

KEY = "e2e-test-key-0123456789"
HEADERS = {"Authorization": f"Bearer {KEY}"}


class HashEmbedder(Embedder):
    """Hashes words into a unit vector: shared words mean similar vectors. Fully offline."""

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 128
            for w in re.findall(r"[a-z]+", t.lower()):
                v[zlib.crc32(w.encode()) % 128] += 1
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


class ScriptedLLM(LLMClient):
    """Plays the router, the agent and the chat model, telling them apart by their system prompts."""

    async def generate(self, messages, system=None):
        text = messages[-1].content
        if system and system.startswith("You route messages"):
            msg = re.search(r"<message>\n(.*?)\n</message>", text, re.S).group(1)
            route = "rag" if "lease" in msg else "agent" if "note" in msg else "chat"
            return json.dumps({"route": route, "confidence": 0.95, "query": msg})
        if system and system.startswith("You are an agent"):
            if "<observation" not in text:
                args = {"title": "Plan", "body": "written during the e2e test"}
                return json.dumps({"action": "tool", "tool": "save_note", "args": args})
            return json.dumps({"action": "final", "answer": "Note saved."})
        return "ok"

    async def stream(self, messages, system=None):
        yield "Notice is sixty days [1]." if system and "<documents>" in system else "Hello!"


def make_pdf(text: str) -> bytes:
    """A minimal one-page PDF with a text line and a correct cross-reference table."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>"
        ),
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return out


def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        event = data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        events.append((event, data))
    return events


def test_full_flow_through_the_api(tmp_path, monkeypatch):
    db = str(tmp_path / "app.db")
    monkeypatch.setattr(settings, "db_path", db)  # so this process reads what the MCP subprocess writes
    cfg = Settings(
        _env_file=None, api_keys=KEY, rate_limit_per_minute=0, max_upload_mb=1,
        db_path=db, pdf_dir=str(tmp_path / "pdfs"),
    )

    @asynccontextmanager
    async def factory(c):
        llm = LoggedLLM(ScriptedLLM(), "scripted")
        memory = ConversationMemory(c.db_path)
        chat = ChatService(llm, memory)
        embedder = HashEmbedder()
        store = VectorStore(str(tmp_path / "chroma"), "e2e")
        rag = RagService(chat, Retriever(embedder, store, top_k=3, min_score=0.2))
        env = child_env({**os.environ, "DB_PATH": c.db_path}, c)  # the same allowlist production uses
        async with MCPToolClient(env=env) as tools:
            agent = await AgentRuntime.create(llm, tools, max_steps=4)
            router = Router(llm, enabled={Route.CHAT, Route.RAG, Route.AGENT}, min_confidence=0.6)
            orchestrator = Orchestrator(router, chat, memory, rag=rag, agent=agent)
            yield Services(orchestrator, RagDocuments(embedder, store, c.pdf_dir))

    with TestClient(create_app(cfg, factory)) as client:
        def chat(message):
            r = client.post("/chat", json={"conversation_id": "e2e", "message": message}, headers=HEADERS)
            assert r.status_code == 200
            return parse_sse(r.text)

        def text_of(events):
            return "".join(d["text"] for name, d in events if name == "text")

        # No key, no access.
        assert client.post("/chat", json={"conversation_id": "e2e", "message": "hi"}).status_code == 401

        # Upload a real PDF: it is parsed, chunked, embedded and stored.
        pdf = make_pdf("This lease says early termination requires sixty days written notice to the landlord.")
        r = client.post("/documents", files={"file": ("lease.pdf", pdf, "application/pdf")}, headers=HEADERS)
        assert r.status_code == 200 and r.json()["chunks"] >= 1, r.text
        listing = client.get("/documents", headers=HEADERS).json()
        assert [d["name"] for d in listing["documents"]] == ["lease.pdf"]

        # 1. Plain chat.
        events = chat("hello")
        assert [n for n, _ in events] == ["route", "text", "done"]
        assert events[0][1]["route"] == "chat" and text_of(events) == "Hello!"

        # 2. A document question: routed to RAG, answered with a citation and its source.
        events = chat("what does the lease say about early termination notice")
        names = [n for n, _ in events]
        assert names[0] == "route" and events[0][1]["route"] == "rag"
        sources = dict(events)["sources"]["sources"]
        assert (sources[0]["source"], sources[0]["page"]) == ("lease.pdf", 1)
        assert "[1]" in text_of(events)

        # 3. An agent task: it pauses for approval, and nothing is written until approved.
        events = chat("save a note titled Plan")
        assert [n for n, _ in events] == ["route", "confirm", "done"]
        assert events[0][1]["route"] == "agent"
        confirm = events[1][1]
        assert confirm["tool"] == "save_note" and confirm["args"]["title"] == "Plan"
        assert notes.list_notes() == "No notes saved yet."

        r = client.post(
            "/chat/confirm",
            json={"conversation_id": "e2e", "run_id": confirm["run_id"], "approved": True},
            headers=HEADERS,
        )
        assert text_of(parse_sse(r.text)) == "Note saved."
        assert notes.list_notes() == "Plan"  # written by the MCP server process, after approval

    # All three turns were recorded under this API key's namespace.
    history = ConversationMemory(db).history(f"{key_id(KEY)}:e2e")
    assert [m.role for m in history] == ["user", "assistant"] * 3
    assert history[0].content == "hello"

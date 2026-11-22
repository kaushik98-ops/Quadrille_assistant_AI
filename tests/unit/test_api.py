import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from assistant.api.documents import DocumentError
from assistant.api.main import create_app
from assistant.api.orchestrator import ConfirmEvent, RouteEvent, SourcesEvent, TextEvent
from assistant.api.services import Services
from assistant.core.auth import key_id
from assistant.core.config import Settings
from assistant.core.errors import ConfigError
from assistant.rag.service import Source

KEY1 = "test-key-one-0123456789"
KEY2 = "test-key-two-0123456789"
AUTH1 = {"Authorization": f"Bearer {KEY1}"}
AUTH2 = {"X-API-Key": KEY2}
BAD = {"Authorization": "Bearer wrong-key-0000000000"}
PDF = b"%PDF-1.4 fake but starts right"


class FakeOrchestrator:
    def __init__(self):
        self.handled, self.resumed, self.pending = [], [], set()

    async def handle(self, conversation_id, text):
        self.handled.append((conversation_id, text))
        yield RouteEvent("chat", 0.9, text, "model")
        if text == "boom":
            raise RuntimeError("secret internal detail")
        if text == "write":
            self.pending.add((conversation_id, "run1"))
            yield ConfirmEvent("run1", "save_note", {"title": "T"})
            return
        yield SourcesEvent([Source(1, "a.pdf", 2, 0.8)])
        yield TextEvent("hello ")
        yield TextEvent("world")

    def has_pending(self, conversation_id, run_id):
        return (conversation_id, run_id) in self.pending

    async def resume_agent(self, conversation_id, run_id, approved):
        self.pending.discard((conversation_id, run_id))
        self.resumed.append((conversation_id, run_id, approved))
        yield TextEvent("approved" if approved else "denied")


class FakeDocuments:
    def __init__(self):
        self.added = []

    async def add_pdf(self, name, data):
        if b"scanned" in data:
            raise DocumentError("No text could be extracted.")
        self.added.append((name, data))
        return 3

    async def sources(self):
        return {"b.pdf": 2, "a.pdf": 5}


def build(tmp_path, **overrides):
    values = dict(
        api_keys=f"{KEY1},{KEY2}", rate_limit_per_minute=100, max_upload_mb=1,
        db_path=str(tmp_path / "t.db"), pdf_dir=str(tmp_path / "pdfs"),
    )
    values.update(overrides)
    cfg = Settings(_env_file=None, **values)
    orch, docs = FakeOrchestrator(), FakeDocuments()

    @asynccontextmanager
    async def factory(_cfg):
        yield Services(orch, docs)

    return create_app(cfg, factory), orch, docs


@pytest.fixture
def env(tmp_path):
    app, orch, docs = build(tmp_path)
    with TestClient(app) as client:  # runs the lifespan, including the mounted MCP server
        yield SimpleNamespace(client=client, orch=orch, docs=docs)


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


def upload(client, name, data, headers=AUTH1):
    return client.post("/documents", files={"file": (name, data, "application/pdf")}, headers=headers)


# --- access control ---------------------------------------------------------

def test_health_is_public(env):
    r = env.client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_api_docs_are_not_exposed(env):
    assert env.client.get("/openapi.json").status_code == 401
    assert env.client.get("/docs").status_code == 401


def test_auth_is_required_on_every_route(env):
    c = env.client
    chat = {"conversation_id": "c1", "message": "hi"}
    confirm = {"conversation_id": "c1", "run_id": "run1", "approved": True}
    assert c.post("/chat", json=chat).status_code == 401
    assert c.post("/chat", json=chat, headers=BAD).status_code == 401
    assert c.post("/chat/confirm", json=confirm).status_code == 401
    assert c.get("/documents").status_code == 401
    assert c.post("/documents", files={"file": ("a.pdf", PDF)}).status_code == 401
    assert c.post("/chat", json=chat, headers=AUTH1).status_code == 200      # Authorization: Bearer
    assert c.post("/chat", json=chat, headers=AUTH2).status_code == 200      # X-API-Key
    assert len(env.orch.handled) == 2  # nothing ran for the rejected requests


def test_rate_limit_applies_per_key_but_not_to_health(tmp_path):
    app, _, _ = build(tmp_path, rate_limit_per_minute=3)
    with TestClient(app) as c:
        for _ in range(3):
            assert c.get("/documents", headers=AUTH1).status_code == 200
        r = c.get("/documents", headers=AUTH1)
        assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
        assert c.get("/documents", headers=AUTH2).status_code == 200  # another key is unaffected
        assert c.get("/health").status_code == 200
        assert c.get("/documents", headers=BAD).status_code == 401    # and bad keys never count


def test_app_refuses_to_start_without_api_keys():
    with pytest.raises(ConfigError):
        create_app(Settings(_env_file=None, api_keys=""), lambda cfg: None)


# --- chat -------------------------------------------------------------------

def test_chat_streams_sse_events(env):
    r = env.client.post("/chat", json={"conversation_id": "c1", "message": "  hi  "}, headers=AUTH1)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["x-request-id"]
    events = parse_sse(r.text)
    assert [name for name, _ in events] == ["route", "sources", "text", "text", "done"]
    assert events[1][1]["sources"][0] == {"n": 1, "source": "a.pdf", "page": 2, "score": 0.8}
    assert "".join(d["text"] for name, d in events if name == "text") == "hello world"
    assert env.orch.handled[0][1] == "hi"  # the message was trimmed


def test_conversations_are_scoped_to_the_api_key(env):
    c = env.client
    body = {"conversation_id": "shared", "message": "hi"}
    c.post("/chat", json=body, headers=AUTH1)
    c.post("/chat", json=body, headers=AUTH2)
    cid1, cid2 = env.orch.handled[0][0], env.orch.handled[1][0]
    assert cid1 == f"{key_id(KEY1)}:shared" and cid2 == f"{key_id(KEY2)}:shared"

    # Key 1 gets a pending approval. Key 2 can't answer it, even using the same ids.
    c.post("/chat", json={"conversation_id": "shared", "message": "write"}, headers=AUTH1)
    r = c.post("/chat/confirm", json={"conversation_id": "shared", "run_id": "run1", "approved": True},
               headers=AUTH2)
    assert r.status_code == 404 and env.orch.resumed == []


def test_request_validation(env):
    bad_bodies = [
        {"conversation_id": "has space", "message": "hi"},
        {"conversation_id": "../etc", "message": "hi"},
        {"conversation_id": "c" * 65, "message": "hi"},
        {"conversation_id": "c1", "message": ""},
        {"conversation_id": "c1", "message": "   "},
        {"conversation_id": "c1", "message": "x" * 4001},
        {"message": "hi"},
    ]
    for body in bad_bodies:
        assert env.client.post("/chat", json=body, headers=AUTH1).status_code == 422, body
    assert env.orch.handled == []


def test_stream_errors_do_not_leak_details(env):
    r = env.client.post("/chat", json={"conversation_id": "c1", "message": "boom"}, headers=AUTH1)
    assert r.status_code == 200
    assert [name for name, _ in parse_sse(r.text)] == ["route", "error"]
    assert "secret internal detail" not in r.text


def test_confirmation_flow(env):
    c = env.client
    r = c.post("/chat", json={"conversation_id": "c1", "message": "write"}, headers=AUTH1)
    events = parse_sse(r.text)
    assert [name for name, _ in events] == ["route", "confirm", "done"]
    assert events[1][1] == {"run_id": "run1", "tool": "save_note", "args": {"title": "T"}}

    answer = {"conversation_id": "c1", "run_id": "run1", "approved": True}
    r = c.post("/chat/confirm", json=answer, headers=AUTH1)
    assert [name for name, _ in parse_sse(r.text)] == ["text", "done"]
    assert env.orch.resumed == [(f"{key_id(KEY1)}:c1", "run1", True)]
    assert c.post("/chat/confirm", json=answer, headers=AUTH1).status_code == 404  # already answered


def test_confirm_for_unknown_run_is_404(env):
    body = {"conversation_id": "c1", "run_id": "nope", "approved": True}
    assert env.client.post("/chat/confirm", json=body, headers=AUTH1).status_code == 404


# --- documents --------------------------------------------------------------

def test_upload_ok_and_filename_is_sanitized(env):
    r = upload(env.client, "../../evil name.pdf", PDF)
    assert r.status_code == 200 and r.json() == {"filename": "evil name.pdf", "chunks": 3}
    assert env.docs.added == [("evil name.pdf", PDF)]
    assert upload(env.client, "weird<>?.PDF", PDF).json()["filename"] == "weird___.pdf"


def test_upload_rejects_bad_files(env):
    c = env.client
    assert upload(c, "notes.txt", PDF).status_code == 400
    assert upload(c, ".pdf", PDF).status_code == 400
    assert upload(c, "fake.pdf", b"<html>not a pdf</html>").status_code == 400
    assert c.post("/documents", data={"x": "y"}, headers=AUTH1).status_code == 400  # no 'file' field
    assert env.docs.added == []


def test_upload_size_limit(env):
    big = PDF + b"x" * (1024 * 1024 + 10)  # the limit is 1 MB in these tests
    assert upload(env.client, "big.pdf", big).status_code == 413
    assert env.docs.added == []


def test_upload_unusable_pdf_is_422(env):
    r = upload(env.client, "scan.pdf", PDF + b" scanned")
    assert r.status_code == 422 and "No text" in r.json()["detail"]


def test_list_documents(env):
    r = env.client.get("/documents", headers=AUTH1)
    assert r.json() == {"documents": [{"name": "a.pdf", "chunks": 5}, {"name": "b.pdf", "chunks": 2}]}


# --- MCP and UI -------------------------------------------------------------

def test_mcp_is_mounted_behind_the_same_auth(env):
    body = {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                   "clientInfo": {"name": "test", "version": "0"}},
    }
    headers = {"Accept": "application/json, text/event-stream"}
    assert env.client.post("/mcp", json=body, headers=headers).status_code == 401
    r = env.client.post("/mcp", json=body, headers={**headers, **AUTH1})
    assert r.status_code == 200 and "serverInfo" in r.text


def test_ui_is_public_and_locked_down(env):
    page = env.client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert "script-src 'self';" in page.headers["content-security-policy"]
    assert env.client.get("/app.js").status_code == 200

import asyncio
import logging
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from assistant.agent.mcp_client import child_env
from assistant.api.main import create_app
from assistant.api.services import Services
from assistant.core.config import Settings
from assistant.core.guardrails import needs_confirmation, preview_args
from assistant.core.llm.base import LLMClient, Message
from assistant.core.llm.logged import LoggedLLM
from assistant.core.retry import retry_async
from assistant.mcp_server.server import build_mcp_server
from assistant.tools.builtin import web

# --- guardrails -------------------------------------------------------------

def test_needs_confirmation_external_tools_after_untrusted_content():
    assert needs_confirmation(True, True, tainted=False) is False   # the user's own request: fine
    assert needs_confirmation(True, True, tainted=True) is True     # after reading untrusted content: ask
    assert needs_confirmation(True, None, tainted=True) is True     # undeclared reach counts as outside
    assert needs_confirmation(True, False, tainted=True) is False   # closed-world read-only tool
    assert needs_confirmation(False, False, tainted=False) is True  # writes always ask


def test_preview_args_truncates():
    assert preview_args({"a": 1}) == '{"a": 1}'
    out = preview_args({"a": "x" * 500}, limit=50)
    assert out.endswith("...") and len(out) == 53


# --- secrets ----------------------------------------------------------------

def test_child_env_drops_secrets():
    parent = {
        "PATH": "/bin", "HOME": "/home/x", "DB_PATH": "d.db",
        "ANTHROPIC_API_KEY": "sk-1", "OPENAI_API_KEY": "sk-2",
        "API_KEYS": "secret-key-xxxxxxxx", "AGENT_MCP_API_KEY": "k", "RANDOM_TOKEN": "t",
    }
    env = child_env(parent, Settings(_env_file=None, embedding_provider="local"))
    assert env == {"PATH": "/bin", "HOME": "/home/x", "DB_PATH": "d.db"}


def test_child_env_keeps_openai_key_only_for_openai_embeddings():
    parent = {"OPENAI_API_KEY": "sk-2", "ANTHROPIC_API_KEY": "sk-1"}
    env = child_env(parent, Settings(_env_file=None, embedding_provider="openai"))
    assert env == {"OPENAI_API_KEY": "sk-2"}


def test_settings_repr_hides_secrets():
    s = Settings(_env_file=None, anthropic_api_key="sk-secret-123", api_keys="k" * 20)
    for text in (repr(s), str(s), f"{s}"):
        assert "sk-secret-123" not in text and "kkkk" not in text


# --- retry ------------------------------------------------------------------

async def _no_sleep(_):
    return None


def test_retry_succeeds_after_failures():
    calls = []

    async def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("temporary")
        return "ok"

    assert asyncio.run(retry_async(flaky, attempts=3, sleep=_no_sleep)) == "ok"
    assert len(calls) == 3


def test_retry_gives_up_and_raises_the_last_error():
    calls = []

    async def always_fails():
        calls.append(1)
        raise RuntimeError(f"failure {len(calls)}")

    with pytest.raises(RuntimeError, match="failure 2"):
        asyncio.run(retry_async(always_fails, attempts=2, sleep=_no_sleep))
    assert len(calls) == 2


def test_retry_only_catches_the_given_errors():
    calls = []

    async def wrong_kind():
        calls.append(1)
        raise KeyError("not retryable")

    with pytest.raises(KeyError):
        asyncio.run(retry_async(wrong_kind, attempts=3, retry_on=(ValueError,), sleep=_no_sleep))
    assert len(calls) == 1


def test_retry_delays_double():
    delays = []

    async def record(delay):
        delays.append(delay)

    async def always_fails():
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        asyncio.run(retry_async(always_fails, attempts=4, base_delay=0.5, sleep=record))
    assert delays == [0.5, 1.0, 2.0]


# --- LLM call logging -------------------------------------------------------

class Inner(LLMClient):
    def __init__(self, fail=False):
        self.fail = fail

    async def generate(self, messages, system=None):
        if self.fail:
            raise RuntimeError("provider says: secret detail")
        return "hello"

    async def stream(self, messages, system=None):
        yield "he"
        yield "llo"


def llm_logs(caplog):
    return [r.fields for r in caplog.records if r.getMessage() == "llm_call"]


def test_logged_llm_generate(caplog):
    caplog.set_level(logging.INFO)
    out = asyncio.run(LoggedLLM(Inner(), "test").generate([Message("user", "hi")]))
    assert out == "hello"
    (fields,) = llm_logs(caplog)
    assert (fields["llm"], fields["kind"], fields["status"]) == ("test", "generate", "ok")
    assert (fields["in_chars"], fields["out_chars"]) == (2, 5)


def test_logged_llm_stream(caplog):
    caplog.set_level(logging.INFO)

    async def go():
        return [c async for c in LoggedLLM(Inner(), "test").stream([Message("user", "hi")])]

    assert "".join(asyncio.run(go())) == "hello"
    (fields,) = llm_logs(caplog)
    assert (fields["kind"], fields["status"], fields["out_chars"]) == ("stream", "ok", 5)
    assert "first_token_ms" in fields


def test_logged_llm_logs_errors_without_details(caplog):
    caplog.set_level(logging.INFO)
    with pytest.raises(RuntimeError):
        asyncio.run(LoggedLLM(Inner(fail=True), "test").generate([Message("user", "hi")]))
    (fields,) = llm_logs(caplog)
    assert fields["status"] == "error"
    assert "secret detail" not in str(caplog.records[0].__dict__)


# --- tools ------------------------------------------------------------------

def test_web_search_rejects_empty_and_oversized_queries():
    with pytest.raises(ValueError, match="too long"):
        asyncio.run(web.web_search("x" * 301))
    with pytest.raises(ValueError, match="empty"):
        asyncio.run(web.web_search("   "))


def test_tool_annotations_mark_web_search_as_open_world():
    by_name = {t.name: t for t in asyncio.run(build_mcp_server().list_tools())}
    assert by_name["web_search"].annotations.openWorldHint is True
    assert by_name["read_file"].annotations.openWorldHint is False


# --- API error handling -----------------------------------------------------

class BoomDocs:
    async def add_pdf(self, name, data):
        raise RuntimeError("secret internal detail")

    async def sources(self):
        return {}


def test_unhandled_error_returns_generic_500(tmp_path):
    key = "k" * 20
    cfg = Settings(
        _env_file=None, api_keys=key, rate_limit_per_minute=100,
        db_path=str(tmp_path / "t.db"), pdf_dir=str(tmp_path / "pdfs"),
    )

    @asynccontextmanager
    async def factory(_cfg):
        yield Services(object(), BoomDocs())

    with TestClient(create_app(cfg, factory), raise_server_exceptions=False) as client:
        r = client.post(
            "/documents",
            files={"file": ("a.pdf", b"%PDF-1.4 x", "application/pdf")},
            headers={"Authorization": f"Bearer {key}"},
        )
    assert r.status_code == 500
    body = r.json()
    assert body["detail"] == "Internal server error" and body["request_id"]
    assert "secret internal detail" not in r.text

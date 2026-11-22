import chromadb

from assistant.core.config import Settings
from assistant.rag.store import build_store


class FakeClient:
    def get_or_create_collection(self, **kwargs):
        return object()


def patch_clients(monkeypatch):
    calls = {}

    def fake_http(host, port):
        calls["http"] = (host, port)
        return FakeClient()

    def fake_persistent(path):
        calls["persistent"] = path
        return FakeClient()

    monkeypatch.setattr(chromadb, "HttpClient", fake_http)
    monkeypatch.setattr(chromadb, "PersistentClient", fake_persistent)
    return calls


def test_chroma_server_is_used_when_host_is_set(monkeypatch):
    calls = patch_clients(monkeypatch)
    build_store(Settings(_env_file=None, chroma_host="chroma", chroma_port=8123))
    assert calls == {"http": ("chroma", 8123)}


def test_embedded_chroma_is_used_when_host_is_empty(monkeypatch, tmp_path):
    calls = patch_clients(monkeypatch)
    build_store(Settings(_env_file=None, chroma_host="", chroma_path=str(tmp_path / "c")))
    assert calls == {"persistent": str(tmp_path / "c")}

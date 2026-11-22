import asyncio
import math
import re
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import assistant.api.documents as documents
from assistant.api.documents import DocumentError, RagDocuments
from assistant.rag.embeddings import Embedder
from assistant.rag.store import VectorStore


class HashEmbedder(Embedder):
    """Hashes words into a unit vector. Can be told to fail."""

    def __init__(self):
        self.fail = False

    def embed(self, texts):
        if self.fail:
            raise RuntimeError("embedding service down")
        out = []
        for t in texts:
            v = [0.0] * 128
            for w in re.findall(r"[a-z]+", t.lower()):
                v[zlib.crc32(w.encode()) % 128] += 1
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


def fake_load(path):
    """Stands in for the PDF reader: the file's bytes are its text."""
    data = Path(path).read_bytes()
    if data == b"corrupt":
        raise RuntimeError("bad pdf")
    if data == b"empty":
        return []
    return [(1, data.decode())]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(documents, "load_pdf", fake_load)
    emb = HashEmbedder()
    store = VectorStore(str(tmp_path / "chroma"), "test")
    folder = tmp_path / "pdfs"
    return SimpleNamespace(docs=RagDocuments(emb, store, folder), emb=emb, store=store, dir=folder)


def add(setup, name, data):
    return asyncio.run(setup.docs.add_pdf(name, data))


def test_add_pdf_ingests_and_saves(setup):
    assert add(setup, "a.pdf", b"alpha beta gamma") == 1
    assert (setup.dir / "a.pdf").read_bytes() == b"alpha beta gamma"
    assert asyncio.run(setup.docs.sources()) == {"a.pdf": 1}
    assert [p.name for p in setup.dir.iterdir()] == ["a.pdf"]  # no .part file left behind


def test_unreadable_pdf_is_rejected_and_nothing_is_kept(setup):
    with pytest.raises(DocumentError, match="PDF"):
        add(setup, "bad.pdf", b"corrupt")
    assert list(setup.dir.iterdir()) == []
    assert asyncio.run(setup.docs.sources()) == {}


def test_pdf_without_text_is_rejected(setup):
    with pytest.raises(DocumentError, match="No text"):
        add(setup, "scan.pdf", b"empty")
    assert list(setup.dir.iterdir()) == []


def test_failed_reupload_keeps_the_previous_version(setup):
    add(setup, "a.pdf", b"alpha beta gamma")
    with pytest.raises(DocumentError):
        add(setup, "a.pdf", b"corrupt")
    assert (setup.dir / "a.pdf").read_bytes() == b"alpha beta gamma"
    assert asyncio.run(setup.docs.sources()) == {"a.pdf": 1}


def test_embedding_failure_keeps_old_chunks(setup):
    add(setup, "a.pdf", b"alpha beta gamma")
    setup.emb.fail = True
    with pytest.raises(RuntimeError):
        add(setup, "a.pdf", b"delta epsilon zeta")
    setup.emb.fail = False
    assert (setup.dir / "a.pdf").read_bytes() == b"alpha beta gamma"
    hits = setup.store.query(setup.emb.embed(["alpha beta gamma"])[0], 3)
    assert [h.text for h in hits] == ["alpha beta gamma"]

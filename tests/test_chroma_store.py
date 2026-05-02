"""ChromaDB store: end-to-end smoke test with metadata sanitization."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from medlit.models import Chunk, Source
from medlit.storage.chroma_store import _decode_list, _sanitize


def test_sanitize_drops_none_and_encodes_lists():
    md = {
        "paper_id": "p",
        "section": None,
        "authors": ["Liu D", "Smith J"],
        "mesh_terms": [],
        "url": None,
        "chunk_index": 0,
    }
    out = _sanitize(md)
    assert "section" not in out
    assert "url" not in out
    assert out["paper_id"] == "p"
    assert out["chunk_index"] == 0
    assert isinstance(out["authors"], str) and out["authors"].startswith("__list__:")


def test_decode_list_roundtrip():
    raw = ["Liu D", "Smith J"]
    sanitized = _sanitize({"authors": raw})
    assert _decode_list(sanitized["authors"]) == raw
    # Also handles missing or non-encoded values gracefully.
    assert _decode_list(None) == []
    assert _decode_list("not a list marker") == []


@pytest.mark.skipif(
    pytest.importorskip("chromadb", reason="chromadb not installed") is None,
    reason="chromadb not installed",
)
def test_chroma_upsert_with_none_metadata(tmp_path: Path):
    from medlit.storage.chroma_store import ChromaStore

    store = ChromaStore(persist_dir=str(tmp_path / "chroma"), collection="medlit-test")
    store.ensure_collection(dimension=4)
    chunk = Chunk(
        id="p1:0",
        paper_id="p1",
        text="hello world",
        chunk_index=0,
        section=None,                      # <- None used to crash chroma
        title=None,
        journal=None,
        publication_date=None,
        authors=["Liu D"],                 # <- list also used to crash chroma
        mesh_terms=[],
        publication_types=[],
        source=Source.PUBMED,
        pmid=None,
        doi=None,
        url=None,
        citation_token="PMID:1",
    )
    vec = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    store.upsert([chunk], vec)
    got = store.get_paper_chunks("p1")
    assert len(got) == 1
    assert got[0].authors == ["Liu D"]
    assert got[0].mesh_terms == []
    assert got[0].section is None

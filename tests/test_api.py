"""API smoke tests. Mocks heavy services (embedder, store, generator)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from medlit.api import deps
from medlit.api.app import app
from medlit.generation.base import BaseGenerator, GenerationRequest
from medlit.models import Chunk, Source
from tests.conftest import FakeEmbedder, InMemoryStore


class FakeGenerator(BaseGenerator):
    model = "fake"

    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        yield "The "
        yield "answer "
        yield f"is X [{request.context[0].chunk.citation_token}]."


@pytest.fixture(autouse=True)
def _patch_deps(monkeypatch):
    embedder = FakeEmbedder()
    store = InMemoryStore()
    chunks = [
        Chunk(
            id="p1:0",
            paper_id="p1",
            text="CRISPR base editing changes single nucleotides.",
            chunk_index=0,
            section="Abstract",
            title="A primer",
            source=Source.PUBMED,
            pmid="1",
            citation_token="PMID:1",
        ),
        Chunk(
            id="p2:0",
            paper_id="p2",
            text="LNPs deliver mRNA therapeutics in vivo.",
            chunk_index=0,
            section="Abstract",
            title="LNP delivery",
            source=Source.PUBMED,
            pmid="2",
            citation_token="PMID:2",
        ),
    ]
    vecs = embedder.embed_documents([c.text for c in chunks])
    store.upsert(chunks, vecs)

    deps.get_embedder.cache_clear()
    deps.get_store.cache_clear()
    deps.get_reranker.cache_clear()
    deps.get_retrieval.cache_clear()
    deps.get_generator.cache_clear()
    deps.get_ingestion.cache_clear()

    monkeypatch.setattr(deps, "build_embedder", lambda: embedder)
    monkeypatch.setattr(deps, "build_vector_store", lambda: store)
    monkeypatch.setattr(deps, "build_generator", lambda: FakeGenerator())
    # Disable reranker for the test (would otherwise need to download a model).
    from medlit.config import get_settings
    s = get_settings()
    s.reranker.enabled = False
    yield


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_search_endpoint():
    with TestClient(app) as client:
        r = client.post("/search", json={"query": "base editing", "top_k": 2})
        assert r.status_code == 200
        body = r.json()
        assert len(body["results"]) >= 1
        assert body["results"][0]["chunk"]["text"]


def test_stats_endpoint():
    with TestClient(app) as client:
        r = client.get("/stats")
        assert r.status_code == 200
        body = r.json()
        assert body["embedding_model"] == "fake-embedder"
        assert body["store"]["vectors_count"] >= 2


def test_papers_get():
    with TestClient(app) as client:
        r = client.get("/papers/p1")
        assert r.status_code == 200
        chunks = r.json()
        assert chunks[0]["paper_id"] == "p1"


def test_papers_list():
    with TestClient(app) as client:
        r = client.get("/papers")
        assert r.status_code == 200
        rows = r.json()
        assert len(rows) >= 2
        ids = {row["paper_id"] for row in rows}
        assert {"p1", "p2"} <= ids
        # required fields populated
        for row in rows:
            assert "chunk_count" in row
            assert row["chunk_count"] >= 1
            assert "source" in row


def test_ask_streams_sse():
    with TestClient(app) as client:
        with client.stream("POST", "/ask", json={"question": "what is base editing?", "top_k": 2}) as r:
            assert r.status_code == 200
            collected = "".join(r.iter_text())
        # SSE frames separated by blank lines; we just check our tokens were emitted.
        assert "The " in collected
        assert "answer " in collected

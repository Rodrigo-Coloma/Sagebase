"""Shared fixtures: in-memory vector store and a fake embedder."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from medlit.models import Chunk, Source
from medlit.storage.base import PaperSummary, SearchFilter, VectorStore


class FakeEmbedder:
    model_name = "fake-embedder"
    dimension = 16

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        # Deterministic pseudo-embedding from text content; normalized.
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, t in enumerate(texts):
            for j, ch in enumerate(t.encode("utf-8")):
                out[i, j % self.dimension] += float(ch) / 255.0
            n = float(np.linalg.norm(out[i]))
            if n > 0:
                out[i] /= n
        return out

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class InMemoryStore(VectorStore):
    def __init__(self) -> None:
        self.chunks: dict[str, Chunk] = {}
        self.vectors: dict[str, np.ndarray] = {}

    def ensure_collection(self, *, dimension: int) -> None:
        return None

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        for c, v in zip(chunks, vectors, strict=True):
            self.chunks[c.id] = c
            self.vectors[c.id] = v

    def dense_search(
        self, query_vector: np.ndarray, *, top_k: int, filter_: SearchFilter | None = None
    ) -> list[tuple[Chunk, float]]:
        ids = list(self.chunks.keys())
        if not ids:
            return []
        mat = np.stack([self.vectors[i] for i in ids])
        scores = mat @ query_vector
        order = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[ids[i]], float(scores[i])) for i in order]

    def sparse_search(
        self, query_text: str, *, top_k: int, filter_: SearchFilter | None = None
    ) -> list[tuple[Chunk, float]]:
        # naive overlap score
        q = set(query_text.lower().split())
        scored: list[tuple[Chunk, float]] = []
        for c in self.chunks.values():
            t = set(c.text.lower().split())
            if not t:
                continue
            score = len(q & t) / (len(q) + 1)
            if score > 0:
                scored.append((c, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def delete_paper(self, paper_id: str) -> int:
        before = len(self.chunks)
        for cid in list(self.chunks):
            if self.chunks[cid].paper_id == paper_id:
                del self.chunks[cid]
                del self.vectors[cid]
        return before - len(self.chunks)

    def get_paper_chunks(self, paper_id: str) -> list[Chunk]:
        return sorted(
            (c for c in self.chunks.values() if c.paper_id == paper_id),
            key=lambda c: c.chunk_index,
        )

    def list_papers(self) -> list[PaperSummary]:
        by_paper: dict[str, Chunk] = {}
        counts: dict[str, int] = {}
        for c in self.chunks.values():
            counts[c.paper_id] = counts.get(c.paper_id, 0) + 1
            by_paper.setdefault(c.paper_id, c)
        return [
            PaperSummary(
                paper_id=pid,
                title=c.title,
                authors=list(c.authors),
                journal=c.journal,
                publication_date=c.publication_date,
                source=c.source,
                doi=c.doi,
                pmid=c.pmid,
                arxiv_id=c.arxiv_id,
                url=c.url,
                citation_token=c.citation_token,
                chunk_count=counts[pid],
            )
            for pid, c in by_paper.items()
        ]

    def stats(self) -> dict[str, object]:
        return {"vectors_count": len(self.chunks)}


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def in_memory_store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def sample_chunk() -> Chunk:
    return Chunk(
        id="paper1:0",
        paper_id="paper1",
        text="CRISPR base editors enable single-nucleotide changes without DSBs.",
        chunk_index=0,
        section="Abstract",
        source=Source.PUBMED,
        pmid="12345",
        citation_token="PMID:12345",
        title="A primer on base editing",
    )

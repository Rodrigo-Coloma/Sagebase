"""Vector store abstraction. Same interface across Qdrant, Chroma, pgvector."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from medlit.models import Chunk, Source


@dataclass(slots=True)
class SearchFilter:
    """Pre-filter applied before vector search."""

    date_from: date | None = None
    date_to: date | None = None
    journals: list[str] | None = None
    authors: list[str] | None = None
    mesh_terms: list[str] | None = None
    publication_types: list[str] | None = None
    sources: list[str] | None = None
    paper_ids: list[str] | None = None


@dataclass(slots=True)
class PaperSummary:
    """One row per indexed paper, aggregated across its chunks."""

    paper_id: str
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    journal: str | None = None
    publication_date: date | None = None
    source: Source = Source.MANUAL
    doi: str | None = None
    pmid: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    citation_token: str | None = None
    chunk_count: int = 0


class VectorStore(ABC):
    @abstractmethod
    def ensure_collection(self, *, dimension: int) -> None:
        ...

    @abstractmethod
    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        ...

    @abstractmethod
    def dense_search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int,
        filter_: SearchFilter | None = None,
    ) -> list[tuple[Chunk, float]]:
        ...

    @abstractmethod
    def sparse_search(
        self,
        query_text: str,
        *,
        top_k: int,
        filter_: SearchFilter | None = None,
    ) -> list[tuple[Chunk, float]]:
        ...

    @abstractmethod
    def delete_paper(self, paper_id: str) -> int:
        """Delete all chunks for a given paper. Returns number deleted."""
        ...

    @abstractmethod
    def get_paper_chunks(self, paper_id: str) -> list[Chunk]:
        ...

    @abstractmethod
    def list_papers(self) -> list[PaperSummary]:
        """Aggregate all chunks into one row per paper."""
        ...

    @abstractmethod
    def stats(self) -> dict[str, object]:
        ...

    def close(self) -> None:
        return None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _filter_dict(filter_: SearchFilter | None) -> dict[str, object]:
        if filter_ is None:
            return {}
        out: dict[str, object] = {}
        if filter_.date_from:
            out["date_from"] = filter_.date_from.isoformat()
        if filter_.date_to:
            out["date_to"] = filter_.date_to.isoformat()
        if filter_.journals:
            out["journals"] = list(filter_.journals)
        if filter_.authors:
            out["authors"] = list(filter_.authors)
        if filter_.mesh_terms:
            out["mesh_terms"] = list(filter_.mesh_terms)
        if filter_.publication_types:
            out["publication_types"] = list(filter_.publication_types)
        if filter_.sources:
            out["sources"] = list(filter_.sources)
        if filter_.paper_ids:
            out["paper_ids"] = list(filter_.paper_ids)
        return out

    @staticmethod
    def _chunks_payload(chunks: Iterable[Chunk]) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for c in chunks:
            out.append(
                {
                    "paper_id": c.paper_id,
                    "text": c.text,
                    "chunk_index": c.chunk_index,
                    "section": c.section,
                    "page_number": c.page_number,
                    "token_count": c.token_count,
                    "title": c.title,
                    "authors": c.authors,
                    "journal": c.journal,
                    "publication_date": c.publication_date.isoformat()
                    if c.publication_date
                    else None,
                    "doi": c.doi,
                    "pmid": c.pmid,
                    "arxiv_id": c.arxiv_id,
                    "mesh_terms": c.mesh_terms,
                    "publication_types": c.publication_types,
                    "source": c.source.value,
                    "url": c.url,
                    "citation_token": c.citation_token,
                }
            )
        return out

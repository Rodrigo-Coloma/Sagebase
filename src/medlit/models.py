"""Core domain models. All ingesters normalize to `Paper`; the chunker emits `Chunk`."""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class Source(str, Enum):
    PUBMED = "pubmed"
    PMC = "pmc"
    ARXIV = "arxiv"
    BIORXIV = "biorxiv"
    MEDRXIV = "medrxiv"
    CROSSREF = "crossref"
    UNPAYWALL = "unpaywall"
    MANUAL = "manual"


class AccessStatus(str, Enum):
    OPEN = "open"            # full text available legitimately
    CLOSED = "closed"        # abstract only
    UNKNOWN = "unknown"


class Author(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    affiliation: str | None = None
    orcid: str | None = None


class Paper(BaseModel):
    """Normalized paper representation across all ingesters."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    id: str = Field(..., description="Stable internal id (sha1 of canonical key).")
    title: str
    authors: list[Author] = Field(default_factory=list)
    abstract: str | None = None
    full_text: str | None = None
    journal: str | None = None
    publication_date: date | None = None
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    arxiv_id: str | None = None
    mesh_terms: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    publication_types: list[str] = Field(
        default_factory=list,
        description="e.g. 'Randomized Controlled Trial', 'Meta-Analysis'.",
    )
    source: Source
    url: HttpUrl | None = None
    access_status: AccessStatus = AccessStatus.UNKNOWN
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @field_validator("doi")
    @classmethod
    def _normalize_doi(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip().lower()
        for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
            if v.startswith(prefix):
                v = v[len(prefix):]
        return v or None

    @field_validator("pmid", "pmcid", "arxiv_id")
    @classmethod
    def _strip(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = v.strip()
        return s or None

    @classmethod
    def make_id(
        cls,
        *,
        doi: str | None = None,
        pmid: str | None = None,
        arxiv_id: str | None = None,
        title: str | None = None,
    ) -> str:
        """Stable id derived from the strongest available identifier."""
        canonical: str
        if doi:
            canonical = f"doi:{doi.lower()}"
        elif pmid:
            canonical = f"pmid:{pmid}"
        elif arxiv_id:
            canonical = f"arxiv:{arxiv_id}"
        else:
            canonical = f"title:{(title or '').strip().lower()}"
        return hashlib.sha1(canonical.encode("utf-8")).hexdigest()

    @property
    def citation_token(self) -> str:
        """Inline citation marker preferred for prompt context."""
        if self.pmid:
            return f"PMID:{self.pmid}"
        if self.doi:
            return f"DOI:{self.doi}"
        if self.arxiv_id:
            return f"arXiv:{self.arxiv_id}"
        return f"id:{self.id[:10]}"


class Chunk(BaseModel):
    """A retrievable unit of text with paper-level metadata propagated."""

    model_config = ConfigDict(extra="ignore")

    id: str
    paper_id: str
    text: str
    chunk_index: int
    section: str | None = None
    page_number: int | None = None
    token_count: int | None = None

    # Propagated metadata (denormalized for fast filtering)
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    journal: str | None = None
    publication_date: date | None = None
    doi: str | None = None
    pmid: str | None = None
    arxiv_id: str | None = None
    mesh_terms: list[str] = Field(default_factory=list)
    publication_types: list[str] = Field(default_factory=list)
    source: Source
    url: str | None = None
    citation_token: str | None = None

    @classmethod
    def make_id(cls, paper_id: str, chunk_index: int) -> str:
        return f"{paper_id}:{chunk_index}"


class RetrievalResult(BaseModel):
    """Single retrieved chunk + score, returned by the retrieval pipeline."""

    model_config = ConfigDict(extra="ignore")

    chunk: Chunk
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None


class IngestionJob(BaseModel):
    """State of an async ingestion job (used by the API)."""

    id: str
    status: str = "pending"   # pending | running | completed | failed
    source: Source
    query: str | None = None
    total: int = 0
    processed: int = 0
    failed: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    error: str | None = None

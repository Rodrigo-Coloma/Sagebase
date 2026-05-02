"""Request/response schemas for the REST API."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from medlit.models import IngestionJob, RetrievalResult


class PubMedIngestRequest(BaseModel):
    query: str
    max_results: int = Field(default=100, ge=1, le=10_000)
    date_from: date | None = None
    date_to: date | None = None
    with_full_text: bool = True


class ArxivIngestRequest(BaseModel):
    query: str
    max_results: int = Field(default=50, ge=1, le=2_000)


class BiorxivIngestRequest(BaseModel):
    query: str | None = None
    max_results: int = Field(default=50, ge=1, le=2_000)
    server: str = "biorxiv"
    date_from: date | None = None
    date_to: date | None = None


class DoiIngestRequest(BaseModel):
    doi: str


class IngestJobResponse(BaseModel):
    job: IngestionJob


class SearchFilterModel(BaseModel):
    date_from: date | None = None
    date_to: date | None = None
    journals: list[str] | None = None
    authors: list[str] | None = None
    mesh_terms: list[str] | None = None
    publication_types: list[str] | None = None
    sources: list[str] | None = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    use_mmr: bool = False
    rerank: bool = True
    filter: SearchFilterModel | None = None


class SearchResponse(BaseModel):
    results: list[RetrievalResult]


class AskRequest(BaseModel):
    question: str
    top_k: int = 8
    rerank: bool = True
    filter: SearchFilterModel | None = None
    max_tokens: int = 1024
    temperature: float = 0.2


class StatsResponse(BaseModel):
    store: dict[str, object]
    embedding_model: str
    generation_model: str

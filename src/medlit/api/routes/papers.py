"""Paper retrieval endpoint."""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from medlit.api.deps import get_store
from medlit.models import Chunk

router = APIRouter(prefix="/papers", tags=["papers"])


class PaperSummaryModel(BaseModel):
    paper_id: str
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    journal: str | None = None
    publication_date: date | None = None
    source: str
    doi: str | None = None
    pmid: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    citation_token: str | None = None
    chunk_count: int


@router.get("", response_model=list[PaperSummaryModel])
async def list_papers() -> list[PaperSummaryModel]:
    store = get_store()
    return [
        PaperSummaryModel(
            paper_id=p.paper_id,
            title=p.title,
            authors=p.authors,
            journal=p.journal,
            publication_date=p.publication_date,
            source=p.source.value,
            doi=p.doi,
            pmid=p.pmid,
            arxiv_id=p.arxiv_id,
            url=p.url,
            citation_token=p.citation_token,
            chunk_count=p.chunk_count,
        )
        for p in store.list_papers()
    ]


@router.get("/{paper_id}", response_model=list[Chunk])
async def get_paper(paper_id: str) -> list[Chunk]:
    store = get_store()
    chunks = store.get_paper_chunks(paper_id)
    if not chunks:
        raise HTTPException(status_code=404, detail="Paper not found.")
    return chunks


@router.delete("/{paper_id}")
async def delete_paper(paper_id: str) -> dict[str, int]:
    store = get_store()
    n = store.delete_paper(paper_id)
    return {"deleted": n}

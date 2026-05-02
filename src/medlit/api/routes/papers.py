"""Paper retrieval endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from medlit.api.deps import get_store
from medlit.models import Chunk

router = APIRouter(prefix="/papers", tags=["papers"])


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

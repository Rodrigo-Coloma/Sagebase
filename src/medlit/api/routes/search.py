"""Search endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from medlit.api.deps import get_retrieval
from medlit.api.schemas import SearchFilterModel, SearchRequest, SearchResponse
from medlit.storage.base import SearchFilter

router = APIRouter(tags=["search"])


def _to_filter(model: SearchFilterModel | None) -> SearchFilter | None:
    if model is None:
        return None
    return SearchFilter(
        date_from=model.date_from,
        date_to=model.date_to,
        journals=model.journals,
        authors=model.authors,
        mesh_terms=model.mesh_terms,
        publication_types=model.publication_types,
        sources=model.sources,
    )


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    pipe = get_retrieval()
    flt = _to_filter(req.filter)
    results = pipe.search(req.query, filter_=flt, top_k=req.top_k, use_mmr=req.use_mmr)
    return SearchResponse(results=results)

"""Stats endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from medlit.api.deps import get_embedder, get_store
from medlit.api.schemas import StatsResponse
from medlit.config import get_settings

router = APIRouter(tags=["stats"])


@router.get("/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    settings = get_settings()
    return StatsResponse(
        store=get_store().stats(),
        embedding_model=get_embedder().model_name,
        generation_model=settings.generation.model,
    )

"""RAG /ask endpoint with Server-Sent Events streaming."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from medlit.api.deps import get_generator, get_retrieval
from medlit.api.routes.search import _to_filter
from medlit.api.schemas import AskRequest
from medlit.generation.base import GenerationRequest

router = APIRouter(tags=["ask"])


@router.post("/ask")
async def ask(req: AskRequest) -> EventSourceResponse:
    pipe = get_retrieval()
    results = pipe.search(req.question, filter_=_to_filter(req.filter), top_k=req.top_k)
    if not results:
        raise HTTPException(status_code=404, detail="No relevant context found.")

    gen = get_generator()
    gen_req = GenerationRequest(
        question=req.question,
        context=results,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )

    async def event_stream() -> AsyncIterator[dict[str, str]]:
        # First event: the sources we're grounding on (so the client can render them).
        sources_payload = [
            {
                "rank": i + 1,
                "score": r.score,
                "citation": r.chunk.citation_token,
                "title": r.chunk.title,
                "url": r.chunk.url,
                "section": r.chunk.section,
                "doi": r.chunk.doi,
                "pmid": r.chunk.pmid,
                "snippet": r.chunk.text[:300],
            }
            for i, r in enumerate(results)
        ]
        yield {"event": "sources", "data": json.dumps(sources_payload)}

        async for piece in gen.stream(gen_req):
            yield {"event": "token", "data": piece}

        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_stream())

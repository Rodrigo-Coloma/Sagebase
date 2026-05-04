"""FastAPI entry point."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from medlit.api.deps import get_embedder, get_reranker, get_store
from medlit.api.routes import ask, ingest, papers, search, stats
from medlit.config import get_settings
from medlit.logging import logger, setup_logging

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    """Pre-warm the heavy singletons (embedder, store, reranker) before
    serving the first request. Without this, the first ingestion / search
    pays a 30s+ cold-start while the BGE model loads, AND if it crashes
    inside a BackgroundTasks runner the failure is invisible to the user."""
    logger.info("warming up medlit singletons (embedder, store, reranker)…")
    settings = get_settings()
    try:
        await asyncio.to_thread(get_embedder)
        await asyncio.to_thread(get_store)
        if settings.reranker.enabled:
            await asyncio.to_thread(get_reranker)
        logger.info("warm-up complete; ready to serve")
    except Exception:  # noqa: BLE001
        # Don't crash the app — first user request will retry. Just log loudly.
        logger.exception("warm-up failed; first request will retry the load")
    yield
    # No teardown — the singletons live for the life of the process.


app = FastAPI(
    title="medlit",
    description="Vectorized medical literature RAG.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(search.router)
app.include_router(ask.router)
app.include_router(papers.router)
app.include_router(stats.router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

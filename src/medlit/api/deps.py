"""Lazy-singleton service container shared across requests."""

from __future__ import annotations

from functools import lru_cache

from medlit.config import get_settings
from medlit.embeddings.factory import CachedEmbedder, build_embedder
from medlit.generation.base import BaseGenerator
from medlit.generation.factory import build_generator
from medlit.pipeline import IngestionPipeline
from medlit.retrieval.pipeline import RetrievalPipeline
from medlit.retrieval.reranker import CrossEncoderReranker
from medlit.storage.base import VectorStore
from medlit.storage.factory import build_vector_store


@lru_cache(maxsize=1)
def get_embedder() -> CachedEmbedder:
    return build_embedder()


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    embedder = get_embedder()
    store = build_vector_store()
    store.ensure_collection(dimension=embedder.dimension)
    return store


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker | None:
    settings = get_settings()
    if not settings.reranker.enabled:
        return None
    return CrossEncoderReranker(settings.reranker.model)


@lru_cache(maxsize=1)
def get_retrieval() -> RetrievalPipeline:
    return RetrievalPipeline(
        embedder=get_embedder(),
        store=get_store(),
        reranker=get_reranker(),
    )


@lru_cache(maxsize=1)
def get_generator() -> BaseGenerator:
    return build_generator()


@lru_cache(maxsize=1)
def get_ingestion() -> IngestionPipeline:
    return IngestionPipeline(embedder=get_embedder(), store=get_store())

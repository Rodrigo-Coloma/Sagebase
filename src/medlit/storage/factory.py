"""Vector store factory."""

from __future__ import annotations

from medlit.config import VectorStoreConfig, get_settings
from medlit.storage.base import VectorStore


def build_vector_store(config: VectorStoreConfig | None = None) -> VectorStore:
    cfg = config or get_settings().vector_store
    if cfg.backend == "qdrant":
        from medlit.storage.qdrant_store import QdrantStore

        return QdrantStore(
            url=cfg.qdrant.url,
            collection=cfg.collection,
            api_key=cfg.qdrant.api_key,
        )
    if cfg.backend == "chroma":
        from medlit.storage.chroma_store import ChromaStore

        return ChromaStore(persist_dir=cfg.chroma.persist_dir, collection=cfg.collection)
    if cfg.backend == "pgvector":
        from medlit.storage.pgvector_store import PgVectorStore

        return PgVectorStore(dsn=cfg.pgvector.dsn, collection=cfg.collection)
    raise ValueError(f"unknown vector backend: {cfg.backend}")

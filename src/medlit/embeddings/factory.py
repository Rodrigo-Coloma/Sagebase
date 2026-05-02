"""Factory: build the configured embedder + cached pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from medlit.config import EmbeddingConfig, get_settings
from medlit.embeddings.base import BaseEmbedder, EmbeddingError
from medlit.embeddings.cache import EmbeddingCache
from medlit.logging import logger


class CachedEmbedder:
    """Wraps a BaseEmbedder with an on-disk cache."""

    def __init__(self, embedder: BaseEmbedder, cache: EmbeddingCache) -> None:
        self.embedder = embedder
        self.cache = cache

    @property
    def dimension(self) -> int:
        return self.embedder.dimension

    @property
    def model_name(self) -> str:
        return self.embedder.model_name

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        hit_idx, hit_vec = self.cache.get_many(self.model_name, texts)
        miss_idx = [i for i in range(len(texts)) if i not in set(hit_idx)]
        miss_texts = [texts[i] for i in miss_idx]
        if miss_texts:
            new_vecs = self.embedder.embed_documents(miss_texts)
            self.cache.put_many(self.model_name, miss_texts, new_vecs)
        else:
            new_vecs = np.zeros((0, self.dimension), dtype=np.float32)
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, v in zip(hit_idx, hit_vec, strict=True):
            out[i] = v
        for i, v in zip(miss_idx, new_vecs, strict=True):
            out[i] = v
        logger.debug(
            f"embed_documents: {len(texts)} texts, {len(hit_idx)} cache hits, "
            f"{len(miss_idx)} new"
        )
        return out

    def embed_query(self, text: str) -> np.ndarray:
        # Don't cache queries - usually one-off and small.
        return self.embedder.embed_query(text)


def build_embedder(config: EmbeddingConfig | None = None) -> CachedEmbedder:
    settings = get_settings()
    cfg = config or settings.embedding

    embedder: BaseEmbedder
    if cfg.backend == "sentence_transformers":
        from medlit.embeddings.sentence_transformer import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(cfg.model, batch_size=cfg.batch_size)
    elif cfg.backend == "openai":
        from medlit.embeddings.openai import OpenAIEmbedder

        embedder = OpenAIEmbedder(cfg.model, api_key=settings.openai_api_key)
    elif cfg.backend == "voyage":
        from medlit.embeddings.voyage import VoyageEmbedder

        embedder = VoyageEmbedder(cfg.model, api_key=settings.voyage_api_key)
    else:
        raise EmbeddingError(f"unknown embedding backend: {cfg.backend}")

    cache = EmbeddingCache(Path(cfg.cache_dir))
    return CachedEmbedder(embedder, cache)

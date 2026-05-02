"""Embedder interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class EmbeddingError(RuntimeError):
    pass


class BaseEmbedder(ABC):
    model_name: str
    dimension: int

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Return shape (len(texts), dimension)."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Return shape (dimension,) — some models distinguish query from doc."""
        ...

    @property
    def cache_key(self) -> str:
        """Used in EmbeddingCache to key by (text, model)."""
        return self.model_name

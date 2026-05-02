"""sentence-transformers embedder. Default backend."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from medlit.embeddings.base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, *, batch_size: int = 32, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        self.dimension = int(self._model.get_sentence_embedding_dimension() or 0)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vecs = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > self.batch_size,
            normalize_embeddings=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        # BGE models recommend a query prefix; we apply it for bge-* models only.
        if self.model_name.lower().startswith("baai/bge"):
            text = "Represent this sentence for searching relevant passages: " + text
        vec = self._model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
        return np.asarray(vec, dtype=np.float32)

"""Voyage AI embeddings backend (optional dep)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from medlit.embeddings.base import BaseEmbedder, EmbeddingError

_MODEL_DIMS = {
    "voyage-3": 1024,
    "voyage-3-lite": 512,
    "voyage-large-2": 1536,
}


class VoyageEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str = "voyage-3",
        *,
        api_key: str | None = None,
        batch_size: int = 128,
    ) -> None:
        try:
            import voyageai
        except ImportError as e:  # pragma: no cover - optional dep
            raise EmbeddingError("Install medlit[voyage] to use VoyageEmbedder.") from e
        self.model_name = model_name
        self.batch_size = batch_size
        self._client = voyageai.Client(api_key=api_key)
        self.dimension = _MODEL_DIMS.get(model_name, 1024)

    def _embed(self, texts: Sequence[str], *, input_type: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            resp = self._client.embed(batch, model=self.model_name, input_type=input_type)
            out.extend(resp.embeddings)
        arr = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return arr / norms

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(texts, input_type="document")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], input_type="query")[0]

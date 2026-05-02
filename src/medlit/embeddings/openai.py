"""OpenAI embeddings backend (optional dep)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from medlit.embeddings.base import BaseEmbedder, EmbeddingError

# Approximate model -> dimension. text-embedding-3-* allow shrinking via `dimensions`.
_MODEL_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        *,
        api_key: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 256,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover - optional dep
            raise EmbeddingError("Install medlit[openai] to use OpenAIEmbedder.") from e
        self.model_name = model_name
        self.batch_size = batch_size
        self._dimensions = dimensions
        self._client = OpenAI(api_key=api_key)
        self.dimension = dimensions or _MODEL_DIMS.get(model_name, 1536)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            kwargs: dict[str, object] = {"model": self.model_name, "input": batch}
            if self._dimensions:
                kwargs["dimensions"] = self._dimensions
            resp = self._client.embeddings.create(**kwargs)  # type: ignore[arg-type]
            out.extend(d.embedding for d in resp.data)
        arr = np.asarray(out, dtype=np.float32)
        # Normalize for cosine similarity to behave consistently across backends.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return arr / norms

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

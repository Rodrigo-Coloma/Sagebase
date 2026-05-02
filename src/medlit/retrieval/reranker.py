"""Cross-encoder reranker using sentence-transformers' CrossEncoder."""

from __future__ import annotations

from collections.abc import Sequence

from medlit.models import Chunk, RetrievalResult


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-large", *, max_length: int = 512) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._model = CrossEncoder(model_name, max_length=max_length)

    def rerank(
        self,
        query: str,
        results: Sequence[RetrievalResult],
        *,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        if not results:
            return []
        pairs = [(query, r.chunk.text) for r in results]
        scores = self._model.predict(pairs, show_progress_bar=False)
        out: list[RetrievalResult] = []
        for r, s in zip(results, scores, strict=True):
            out.append(
                RetrievalResult(
                    chunk=r.chunk,
                    score=float(s),
                    dense_score=r.dense_score,
                    sparse_score=r.sparse_score,
                    rerank_score=float(s),
                )
            )
        out.sort(key=lambda x: x.score, reverse=True)
        return out[:top_k] if top_k else out

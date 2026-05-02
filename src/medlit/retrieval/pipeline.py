"""End-to-end retrieval pipeline: dense + sparse + RRF + (optional) rerank + MMR."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from medlit.config import RetrievalConfig, get_settings
from medlit.logging import logger
from medlit.models import RetrievalResult
from medlit.retrieval.hybrid import reciprocal_rank_fusion
from medlit.retrieval.mmr import maximal_marginal_relevance
from medlit.storage.base import SearchFilter, VectorStore

if TYPE_CHECKING:
    from medlit.embeddings.factory import CachedEmbedder
    from medlit.retrieval.reranker import CrossEncoderReranker


class RetrievalPipeline:
    def __init__(
        self,
        *,
        embedder: "CachedEmbedder",
        store: VectorStore,
        reranker: "CrossEncoderReranker | None" = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.config = config or get_settings().retrieval

    def search(
        self,
        query: str,
        *,
        filter_: SearchFilter | None = None,
        top_k: int | None = None,
        use_mmr: bool | None = None,
    ) -> list[RetrievalResult]:
        cfg = self.config
        top_k = top_k or cfg.rerank_top_k
        use_mmr = cfg.use_mmr if use_mmr is None else use_mmr

        # Dense
        qvec = self.embedder.embed_query(query)
        dense = self.store.dense_search(qvec, top_k=cfg.top_k_dense, filter_=filter_)
        # Sparse
        sparse = self.store.sparse_search(query, top_k=cfg.top_k_sparse, filter_=filter_)
        logger.debug(f"retrieval: dense={len(dense)} sparse={len(sparse)}")

        # Fuse
        fused = reciprocal_rank_fusion([dense, sparse], k=cfg.rrf_k)

        # Build initial RetrievalResult list with constituent scores
        dense_scores = {c.id: s for c, s in dense}
        sparse_scores = {c.id: s for c, s in sparse}
        results = [
            RetrievalResult(
                chunk=c,
                score=score,
                dense_score=dense_scores.get(c.id),
                sparse_score=sparse_scores.get(c.id),
            )
            for c, score in fused
        ]

        # Rerank
        if self.reranker is not None and results:
            results = self.reranker.rerank(query, results, top_k=top_k * 2 if use_mmr else top_k)

        # MMR for diversity
        if use_mmr and results:
            cand_texts = [r.chunk.text for r in results]
            cand_embs = self.embedder.embed_documents(cand_texts)
            picks = maximal_marginal_relevance(
                qvec,
                list(zip([r.chunk for r in results], cand_embs, strict=True)),
                k=top_k,
                lambda_mult=cfg.mmr_lambda,
            )
            picked_ids = {c.id for c in picks}
            results = [r for r in results if r.chunk.id in picked_ids][:top_k]
        else:
            results = results[:top_k]

        return results

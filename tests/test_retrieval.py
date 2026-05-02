from __future__ import annotations

import numpy as np

from medlit.models import Chunk, RetrievalResult, Source
from medlit.retrieval.hybrid import reciprocal_rank_fusion
from medlit.retrieval.mmr import maximal_marginal_relevance
from medlit.retrieval.pipeline import RetrievalPipeline


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(
        id=cid, paper_id="p", text=text, chunk_index=0, source=Source.PUBMED, citation_token="PMID:1"
    )


def test_rrf_promotes_chunks_in_both_lists():
    a, b, c = _chunk("a", "x"), _chunk("b", "y"), _chunk("c", "z")
    dense = [(a, 0.9), (b, 0.7), (c, 0.5)]
    sparse = [(b, 5.0), (c, 4.0), (a, 1.0)]
    fused = reciprocal_rank_fusion([dense, sparse])
    ids = [ch.id for ch, _ in fused]
    # b appears at rank 2 dense and rank 1 sparse — should be #1 fused.
    assert ids[0] == "b"


def test_mmr_picks_diverse():
    # 3 query-similar items, 2 of which are near-duplicates.
    q = np.array([1.0, 0.0, 0.0])
    a_emb = np.array([0.9, 0.1, 0.0])
    a_emb /= np.linalg.norm(a_emb)
    a2_emb = a_emb + 1e-3
    a2_emb /= np.linalg.norm(a2_emb)
    b_emb = np.array([0.6, 0.0, 0.4])
    b_emb /= np.linalg.norm(b_emb)
    chunks = [_chunk("a", "a"), _chunk("a2", "a2"), _chunk("b", "b")]
    embs = [a_emb, a2_emb, b_emb]
    picks = maximal_marginal_relevance(q, list(zip(chunks, embs, strict=True)), k=2, lambda_mult=0.3)
    picked_ids = [c.id for c in picks]
    # First pick is a (most similar to q). Second should be b, not the near-dup a2.
    assert picked_ids[0] == "a"
    assert picked_ids[1] == "b"


def test_retrieval_pipeline_end_to_end(fake_embedder, in_memory_store):  # type: ignore[no-untyped-def]
    chunks = [
        _chunk("c1", "CRISPR base editing converts adenine to guanine without DSBs."),
        _chunk("c2", "AAV vectors are widely used for gene therapy delivery."),
        _chunk("c3", "Lipid nanoparticles deliver mRNA to hepatocytes."),
    ]
    vecs = fake_embedder.embed_documents([c.text for c in chunks])
    in_memory_store.upsert(chunks, vecs)

    pipe = RetrievalPipeline(embedder=fake_embedder, store=in_memory_store, reranker=None)
    results = pipe.search("base editing adenine", top_k=2)
    assert results
    assert results[0].chunk.id == "c1"
    # Each result has a fused score, and at least one component score.
    assert all(isinstance(r, RetrievalResult) for r in results)
